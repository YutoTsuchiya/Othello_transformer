import jax
from jax import numpy as jnp
from flax import linen as nn

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (B:batch, C:context, E:embedding)
        B, C, E = x.shape
        H, D = self.num_heads, self.head_dim

        Q = nn.Dense(H*D, use_bias=False)(x) # (B, C, HD)
        K = nn.Dense(H*D, use_bias=False)(x) # (B, C, HD)
        V = nn.Dense(H*D, use_bias=False)(x) # (B, C, HD)

        Q = Q.reshape(B, C, H, D).transpose(0, 2, 1, 3) # (B, H, C, D)
        K = K.reshape(B, C, H, D).transpose(0, 2, 1, 3) # (B, H, C, D)
        V = V.reshape(B, C, H, D).transpose(0, 2, 1, 3) # (B, H, C, D)

        scores = Q @ K.transpose(0, 1, 3, 2)
        scores /= D ** 0.5

        weights = nn.softmax(scores, axis=-1)
        hidden = weights @ V # (B, H, C, D)
        hidden = hidden.transpose(0, 2, 1, 3).reshape(B, C, H*D)

        output = nn.Dense(E, use_bias=False)(hidden)

        return output
    

if __name__ == '__main__':
    dummy_input = jnp.ones(shape=(1, 64, 128))
    key = jax.random.PRNGKey(0)
    
    multihead_attention = MultiHeadAttention(4, 128)
    params = multihead_attention.init(key, dummy_input)

    out = multihead_attention.apply(params, dummy_input)
    print(out.shape)
