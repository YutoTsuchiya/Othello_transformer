import jax
from jax import numpy as jnp
from flax import linen as nn

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jax.Array:
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
    

class LayerNorm(nn.Module):
    eps:float=1e-5
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        norm = (x - mean) / jnp.sqrt(var + self.eps)

        gamma = self.param('gamma', nn.initializers.ones, (x.shape[-1],))
        beta = self.param('beta', nn.initializers.zeros, (x.shape[-1],))

        return gamma * norm + beta


class FFN(nn.Module):
    hidden_dim: int | None = None

    @nn.compact
    def __call__(self, x):
        hidden_dim = 4 * x.shape[-1] if self.hidden_dim is None else self.hidden_dim
        x_dim = x.shape[-1]
        x = nn.Dense(hidden_dim, use_bias=False)(x)
        x = nn.gelu(x)
        out = nn.Dense(x_dim)(x)
        return out
    

if __name__ == '__main__':
    input_shape = (1, 64, 128)
    dummy_input = jnp.ones(shape=input_shape)
    key = jax.random.PRNGKey(0)
    
    multihead_attention = MultiHeadAttention(4, 128)
    mha_params = multihead_attention.init(key, dummy_input)

    layernorm = LayerNorm(128)
    ln_params = layernorm.init(key, dummy_input)

    ffn = FFN(128)
    ffn_params = ffn.init(key, dummy_input)

    mha_out = multihead_attention.apply(mha_params, dummy_input)
    print("MultiHead Attention OK" if input_shape==mha_out.shape else "MultiHead Attention Fail")

    ln_out = layernorm.apply(ln_params, dummy_input)
    print("LayerNorm OK" if ln_out.shape == dummy_input.shape else "LayerNorm Fail")

    ffn_out = ffn.apply(ffn_params, dummy_input)
    print("FFN OK" if ffn_out.shape == dummy_input.shape else "FFN Fail")