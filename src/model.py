import math
from os import supports_bytes_environ

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1024
# LENGTH OF FEN - TBD
BLOCK_SIZE = 77


class GLU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.c_attn = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.c_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
        )
        self.n_head = n_head

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]

        Q, K, V = self.c_attn(x).split(EMBED_DIM, dim=2)
        Q = Q.view(
            batch_size, seq_length, self.n_head, EMBED_DIM // self.n_head
        ).transpose(1, 2)
        K = K.view(
            batch_size, seq_length, self.n_head, EMBED_DIM // self.n_head
        ).transpose(1, 2)
        V = V.view(
            batch_size, seq_length, self.n_head, EMBED_DIM // self.n_head
        ).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(EMBED_DIM // self.n_head)
        att = att.masked_fill(
            self.bias[:, :, :seq_length, :seq_length] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, EMBED_DIM)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention(n_head)
        self.ln_2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
                c_proj=nn.Linear(4 * EMBED_DIM, EMBED_DIM),
                act=GLU(),
                dropout=nn.Dropout(0.1),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class AutoRegressiveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(20, EMBED_DIM),
                wpe=nn.Embedding(BLOCK_SIZE, EMBED_DIM),
                drop=nn.Dropout(0.1),
                h=nn.ModuleList([Block(self, 8) for _ in range(16)]),
                ln_f=nn.LayerNorm(EMBED_DIM),
            )
        )
        self.lm_head = nn.Linear(EMBED_DIM, 20, bias=False)
