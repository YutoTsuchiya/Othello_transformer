import jax
jax.config.update("jax_platform_name", "cpu")

from jax import numpy as jnp
import pgx

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 256)
env = pgx.make('othello')

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

state = init_fn(keys)
print(state.observation[0,:,:,0])


next_state = step_fn(state, jnp.full(256, 19))

actions = jnp.full(256, 19, dtype=jnp.int32)
print(actions.dtype, actions.shape)

next_state = step_fn(state, actions)
print(next_state.observation[0,:,:,0])
print(next_state.observation[0,:,:,1])

state_single = env.init(jax.random.PRNGKey(0))
step_single = jax.jit(env.step)
next_single = step_single(state_single, jnp.int32(19))

print("ch0:", next_single.observation[:,:,0])
print("ch1:", next_single.observation[:,:,1])
