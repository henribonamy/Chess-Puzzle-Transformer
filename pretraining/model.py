import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1024
BLOCK_SIZE = 84
VOCAB_SIZE = 41

class Block(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=n_head,
            dropout=0.1,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(EMBED_DIM)
        self.fc1 = nn.Linear(EMBED_DIM, 2 * EMBED_DIM)
        self.fc2 = nn.Linear(2 * EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attn_mask, need_weights=False)
        x = x + attn_out

        mlp_out = self.fc1(self.ln_2(x))
        mlp_out = F.silu(mlp_out)
        mlp_out = self.fc2(mlp_out)
        mlp_out = self.dropout(mlp_out)
        x = x + mlp_out
        return x


class AutoRegressiveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.wpe = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(8) for _ in range(16)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

        # Weight tying: share embedding weights with output head
        self.lm_head.weight = self.wte.weight

        # Cache positional encodings
        self.register_buffer("pos", torch.arange(0, BLOCK_SIZE, dtype=torch.long))

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print("Model Parameter Count: %.2fM" % (n_params / 1e6,))

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
        seq_length = idx.shape[1]

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(self.pos[:seq_length].unsqueeze(0))
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

    def get_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for the input token sequence."""
        seq_length = idx.shape[1]
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(self.pos[:seq_length].unsqueeze(0))
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def compute_log_probs(self, sequences):
        logits = self.get_logits(sequences[:, :-1])  
        log_probs_all = F.log_softmax(logits, dim=-1)  
        targets = sequences[:, 1:]
        log_probs = torch.gather(
            log_probs_all, 
            dim=-1, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        return log_probs

    def compute_sequence_log_prob(self, sequences):
        token_log_probs = self.compute_log_probs(sequences) 
        return token_log_probs.sum(dim=-1)

    def compute_entropy(self, sequences):
        logits = self.get_logits(sequences[:, :-1])
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1) 
        return token_entropy.mean(dim=-1)

    def generate_with_log_probs(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        device = idx.device
        batch_size = idx.shape[0]
        
        all_log_probs = []
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits = self.get_logits(idx_cond)
            logits = logits[:, -1, :] / temperature  
            log_probs_step = F.log_softmax(logits, dim=-1)  
            probs = torch.exp(log_probs_step)
            idx_next = torch.multinomial(probs, num_samples=1)
            log_prob_selected = torch.gather(
                log_probs_step, 
                dim=-1, 
                index=idx_next
            ).squeeze(-1)
            all_log_probs.append(log_prob_selected)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        log_probs = torch.stack(all_log_probs, dim=1)
        return idx, log_probs
