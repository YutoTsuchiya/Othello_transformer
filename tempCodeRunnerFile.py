from typing import NamedTuple
import mctx
import jax
jax.config.update("jax_platform_name", "cpu")
from jax import numpy as jnp
import pgx

from model import TransformerModel, tokenize_board

seed = 2323
key = jax.random.PRNGKey(seed)
batch_size = 2048
max_steps = 64
num_simulations = 128

model = TransformerModel(num_heads=4, head_dim=128, num_layers=6)
env = pgx.make('othello')
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray          # (max_steps, B, 8, 8, 2)
    action_weights: jnp.ndarray  # (max_steps, B, 65)
    reward: jnp.ndarray       # (max_steps, B)
    discount: jnp.ndarray     # (max_steps, B)


def recurrent_fn(params, rng_key, action, embedding):
    state = embedding
    next_state = jax.vmap(env.step)(state, action)

    tokens = tokenize_board(next_state.observation)
    logits, values = model.apply(params, tokens)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(next_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = next_state.rewards[jnp.arange(batch_size), state.current_player]
    value = jnp.where(next_state.terminated, 0.0, values.reshape(-1))
    discount = jnp.where(next_state.terminated, 0.0, -1.0 * jnp.ones(batch_size))

    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return output, next_state


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
        state = step_fn(state, policy_output.action)

        reward = state.rewards[jnp.arange(batch_size), actor]
        discount = jnp.where(state.terminated, 0.0, -1.0 * jnp.ones(batch_size))

        output = SelfplayOutput(
            obs=obs,
            action_weights=policy_output.action_weights,
            reward=reward,
            discount=discount,
        )
        return state, output

    # 初期化
    keys = jax.random.split(rng_key, batch_size + 1)
    state = init_fn(keys[1:])
    key_seq = jax.random.split(keys[0], max_steps)

    # 固定ステップ数で回す
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


# 実行
key, subkey = jax.random.split(key)
dummy_state = init_fn(jax.random.split(subkey, batch_size))
params = model.init(key, tokenize_board(dummy_state.observation))

key, subkey = jax.random.split(key)
data = selfplay(params, subkey)

print(f"obs: {data.obs.shape}")
print(f"action_weights: {data.action_weights.shape}")
print(f"reward: {data.reward.shape}")
print(f"discount: {data.discount.shape}")
