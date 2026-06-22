import mctx
import jax
import jax.numpy as jnp

from model import TransformerModel


model = TransformerModel(num_heads=4,
                         head_dim=128,
                         num_layers=6
                         )

print(model.)