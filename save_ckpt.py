# save_ckpt.py
import pickle
import os
import jax


def save_ckpt(params, opt_state, iteration, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)

    params_cpu = jax.tree_util.tree_map(lambda x: x[0], params)
    opt_state_cpu = jax.tree_util.tree_map(lambda x: x[0], opt_state)

    ckpt = {
        "params": jax.device_get(params_cpu),
        "opt_state": jax.device_get(opt_state_cpu),
        "iteration": iteration,
    }
    path = os.path.join(ckpt_dir, f"ckpt_{iteration:06d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved: {path}")


def load_ckpt(path, devices):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put_replicated(ckpt["params"], devices)
    opt_state = jax.device_put_replicated(ckpt["opt_state"], devices)
    return params, opt_state, ckpt["iteration"]
