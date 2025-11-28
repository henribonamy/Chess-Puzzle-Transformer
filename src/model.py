import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1024
BLOCK_SIZE = 84
VOCAB_SIZE = 41


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
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlpf(x))
        return x


class AutoRegressiveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(VOCAB_SIZE, EMBED_DIM),
                wpe=nn.Embedding(BLOCK_SIZE, EMBED_DIM),
                drop=nn.Dropout(0.1),
                h=nn.ModuleList([Block(8) for _ in range(16)]),
                ln_f=nn.LayerNorm(EMBED_DIM),
            )
        )
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        seq_length = idx.shape[1]
        pos = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )

        return logits, loss
