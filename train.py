from typing import NamedTuple
from functools import partial
import mctx
import jax
from jax import numpy as jnp
import pgx
import optax
import wandb
from save_ckpt import save_ckpt

from model import TransformerModel, tokenize_board

import wandb
wandb.login(key="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")


# === ハイパーパラメータ ===
seed = 2323
batch_size = 2048
max_steps = 64
num_simulations = 32
learning_rate = 1e-3
num_iterations = 100000
eval_interval = 100
num_eval_games = 4096
save_every = 5000

# === デバイス ===
devices = jax.local_devices()
num_devices = len(devices)
per_device_batch = 4096
per_device_eval = num_eval_games // num_devices

# === 初期化 ===
key = jax.random.PRNGKey(seed)
model = TransformerModel(num_heads=2, head_dim=128, num_layers=6, emb_dim=128)
env = pgx.make('othello')
optimizer = optax.adam(learning_rate)
baseline = pgx.make_baseline_model("othello_v0")

wandb.init(
    project="othello-transformer",
    config={
        "seed": seed,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "num_simulations": num_simulations,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "num_devices": num_devices,
        "num_heads": 4,
        "head_dim": 128,
        "num_layers": 6,
    },
)


# === データ構造 ===
class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    action_weights: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray


# === MCTS内の展開関数 ===
def recurrent_fn(params, rng_key, action, embedding):
    state = embedding
    next_state = jax.vmap(env.step)(state, action)

    tokens = tokenize_board(next_state.observation)
    logits, values = model.apply(params, tokens)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(next_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = next_state.rewards[jnp.arange(per_device_batch), state.current_player]
    value = jnp.where(next_state.terminated, 0.0, values.reshape(-1))
    discount = jnp.where(next_state.terminated, 0.0, -1.0 * jnp.ones(per_device_batch))

    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return output, next_state


# === 自己対局（各デバイスで実行） ===
@jax.pmap
def selfplay(params, rng_key) -> SelfplayOutput:

    def step(state, key):
        key1, key2 = jax.random.split(key)

        tokens = tokenize_board(state.observation)
        logits, values = model.apply(params, tokens)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=values.reshape(-1),
            embedding=state,
        )

        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=8,
        )

        obs = state.observation
        actor = state.current_player
        state = jax.vmap(env.step)(state, policy_output.action)

        reward = state.rewards[jnp.arange(per_device_batch), actor]
        discount = jnp.where(state.terminated, 0.0, -1.0 * jnp.ones(per_device_batch))

        output = SelfplayOutput(
            obs=obs,
            action_weights=policy_output.action_weights,
            reward=reward,
            discount=discount,
        )
        return state, output

    keys = jax.random.split(rng_key, per_device_batch + 1)
    state = jax.vmap(env.init)(keys[1:])
    key_seq = jax.random.split(keys[0], max_steps)
    _, data = jax.lax.scan(step, state, key_seq)
    return data


# === ターゲット計算（各デバイスで実行） ===
@jax.pmap
def compute_targets(data: SelfplayOutput):
    def body_fn(carry, t):
        idx = max_steps - t - 1
        v = data.reward[idx] + data.discount[idx] * carry
        return v, v

    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(per_device_batch), jnp.arange(max_steps))
    value_tgt = value_tgt[::-1]

    terminated = (data.discount == 0.0)
    mask = jnp.cumsum(terminated, axis=0) <= 1
    mask = mask.astype(jnp.float32)

    return value_tgt, mask


# === 学習（各デバイスで実行、勾配を平均） ===
@partial(jax.pmap, axis_name="devices")
def train_step(params, opt_state, obs, policy_tgt, value_tgt, mask):
    def loss_fn(params):
        tokens = tokenize_board(obs.reshape(-1, 8, 8, 2))
        logits, values = model.apply(params, tokens)

        policy_loss = optax.softmax_cross_entropy(logits, policy_tgt.reshape(-1, 65))
        policy_loss = policy_loss.reshape(max_steps, per_device_batch)

        value_loss = optax.l2_loss(values.reshape(max_steps, per_device_batch), value_tgt)

        policy_loss = (policy_loss * mask).sum() / mask.sum()
        value_loss = (value_loss * mask).sum() / mask.sum()

        return policy_loss + value_loss, (policy_loss, value_loss)

    grads, (policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, policy_loss, value_loss


# === 評価（各デバイスで実行） ===
@jax.pmap
def evaluate(params, rng_key):
    my_player = 0
    keys = jax.random.split(rng_key, per_device_eval)
    state = jax.vmap(env.init)(keys)

    def recurrent_fn_eval(params, rng_key, action, embedding):
        state = embedding
        next_state = jax.vmap(env.step)(state, action)

        tokens = tokenize_board(next_state.observation)
        logits, values = model.apply(params, tokens)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(next_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        reward = next_state.rewards[jnp.arange(per_device_eval), state.current_player]
        value = jnp.where(next_state.terminated, 0.0, values.reshape(-1))
        discount = jnp.where(next_state.terminated, 0.0, -1.0 * jnp.ones(per_device_eval))

        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return output, next_state

    def body_fn(val):
        key, state, R = val

        # 自分のターン: MCTSで行動選択
        tokens = tokenize_board(state.observation)
        my_logits, my_value = model.apply(params, tokens)
        my_logits = my_logits - jnp.max(my_logits, axis=-1, keepdims=True)
        my_logits = jnp.where(state.legal_action_mask, my_logits, jnp.finfo(my_logits.dtype).min)

        root = mctx.RootFnOutput(
            prior_logits=my_logits,
            value=my_value.reshape(-1),
            embedding=state,
        )

        key, subkey = jax.random.split(key)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=subkey,
            root=root,
            recurrent_fn=recurrent_fn_eval,
            num_simulations=num_simulations,
            max_num_considered_actions=8,
        )
        my_action = policy_output.action

        # 相手のターン: baselineでサンプリング
        opp_logits, _ = baseline(state.observation)
        key, subkey = jax.random.split(key)
        opp_action = jax.random.categorical(subkey, opp_logits, axis=-1)

        # 手番に応じて行動を選択
        is_my_turn = state.current_player == my_player
        action = jnp.where(is_my_turn, my_action, opp_action)

        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(per_device_eval), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~x[1].terminated.all(),
        body_fn,
        (rng_key, state, jnp.zeros(per_device_eval)),
    )
    return R



def compute_elo(win_rate):
    if win_rate <= 0:
        return 1000.0
    if win_rate >= 1:
        return 2000.0
    elo_diff = -400 * jnp.log10(1.0 / win_rate - 1.0)
    return 1500.0 + float(elo_diff)


# === モデル初期化 & デバイスに複製 ===
key, subkey = jax.random.split(key)
dummy_keys = jax.random.split(subkey, per_device_batch)
dummy_state = jax.vmap(env.init)(dummy_keys)
params = model.init(key, tokenize_board(dummy_state.observation))
opt_state = optimizer.init(params)

# 全デバイスに複製
params = jax.device_put_replicated(params, devices)
opt_state = jax.device_put_replicated(opt_state, devices)

# === メインループ ===
for i in range(num_iterations):
    # 自己対局（デバイスごとに異なるkey）
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_devices)
    data = selfplay(params, keys)

    # ターゲット計算
    value_tgt, mask = compute_targets(data)

    # 学習
    params, opt_state, policy_loss, value_loss = train_step(
        params, opt_state, data.obs, data.action_weights, value_tgt, mask
    )

    # ログ（デバイス0の値を使う）
    pl = float(policy_loss[0])
    vl = float(value_loss[0])
    log = {
        "train/policy_loss": pl,
        "train/value_loss": vl,
        "train/total_loss": pl + vl,
        "iteration": i,
    }

    # 評価
    if i % eval_interval == 0:
        key, subkey = jax.random.split(key)
        eval_keys = jax.random.split(subkey, num_devices)
        R = evaluate(params, eval_keys)
        R = R.reshape(-1)  # (num_devices, per_device_eval) → (num_eval_games,)
        win_rate = float((R == 1).mean())
        draw_rate = float((R == 0).mean())
        lose_rate = float((R == -1).mean())
        elo = compute_elo(win_rate)

        log.update({
            "eval/win_rate": win_rate,
            "eval/draw_rate": draw_rate,
            "eval/lose_rate": lose_rate,
            "eval/elo": elo,
        })
        print(
            f"iter {i:4d} | loss: {pl + vl:.4f} "
            f"| win: {win_rate:.2%} | elo: {elo:.0f}"
        )
    
    # save
    if i % save_every == 0:
        save_ckpt(params, opt_state, i)

    wandb.log(log)

wandb.finish()
print("学習完了")
