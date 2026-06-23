import jax
from jax import numpy as jnp
from flax import linen as nn
from modules import TransformerBlock

class TransformerModel(nn.Module):
    num_heads: int
    head_dim: int
    num_layers: int
    ff_dim: int = None
    policy_size: int = 65
    emb_dim:int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.emb_dim)(x)
        B, C, E = x.shape
        pos_emb = self.param('pos_emb', 
                             nn.initializers.normal(0.02), 
                             (C, E))
        x = x + pos_emb

        for _ in range(self.num_layers):
            x = TransformerBlock(self.head_dim, 
                                 self.num_heads,
                                 self.ff_dim)(x)
        
        policy = nn.Dense(1, name='policy_affin')(x)
        policy = policy.reshape(B, -1)
        policy = nn.gelu(policy)
        policy = nn.Dense(self.policy_size)(policy)

        value = nn.Dense(1, name='value_affin')(x)
        value = value.reshape(B, -1)
        value = nn.gelu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value


def tokenize_board(boards: jnp.ndarray):
    #boards: (B, 8, 8, 2)
    boards = boards.transpose(0, 3, 1, 2)
    B = boards.shape[0]
    idx = jnp.arange(0, 8)

    rows = boards.reshape(B, 2*8, 8)
    columns = boards.transpose(0, 1, 3, 2).reshape(B, 2*8, 8)
    right_diag = boards[:, :, idx, idx]
    left_diag = boards[:, :, idx, idx[::-1]]

    tokens = jnp.concat([rows, columns, right_diag, left_diag], axis=1)
    return tokens


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(shape=[1, 64, 128])

    model = TransformerModel(num_heads=4,
                             head_dim=64,
                             num_layers=4)
    
    params = model.init(key, dummy_input)

    policy, value = model.apply(params, dummy_input)
    
    print(f"policy shape: {policy.shape}")
    print(f"value shape: {value.shape}")