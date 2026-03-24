# Chess Puzzle Generation via RL

Generating hard, counter-intuitive chess puzzles using a three-stage pipeline: pretraining, supervised fine-tuning, and reinforcement learning. Based on [Puzzle Generation via RL (arXiv 2510.23881)](https://arxiv.org/pdf/2510.23881v1).

---

## Pipeline

### 1. Pretraining

A 134M parameter autoregressive transformer is trained from scratch to generate chess positions in FEN (Forsyth-Edwards Notation) format. The model learns the token distribution of legal chess positions by predicting the next character in a FEN string, trained on a large corpus of positions extracted from real games.

Output: a model that can generate syntactically valid FEN strings, with no notion of puzzle quality.

### 2. Supervised Fine-tuning

The pretrained model is fine-tuned on ~2.6M Lichess puzzles with a human difficulty rating above 1500. This shifts the model's prior toward positions that are tactically rich — positions that arise after a blunder, where one side has a forced winning continuation.

Effect: validity jumps to ~98-100%, and the model generates far more positions with a unique best move (UniqueWin per batch increases from 1-4 to 8-16).

### 3. Reinforcement Learning (PPO)

The fine-tuned model is further trained with Proximal Policy Optimization (PPO). At each step, the model generates a batch of FEN positions, each is evaluated by Stockfish, and rewards are assigned based on puzzle quality criteria.

---

## Puzzle Quality Criteria

A generated position qualifies as a puzzle only if it passes all of the following filters:

| Filter | Definition |
|--------|-----------|
| **Valid FEN** | Parseable by python-chess as a legal position |
| **Realistic piece count** | Each side: ≤16 total, ≤8 pawns, =1 king, ≤1 queen, ≤2 rooks, ≤2 bishops, ≤2 knights |
| **Not game over** | Position is not already checkmate or stalemate |
| **Entropy** | Token entropy ≥ τ_ent (0.6) — prevents degenerate repetitive outputs |
| **Uniqueness** | gap = w_deep − w2_deep ≥ τ_uni (0.1): the best move at depth-6 must be significantly better than the second-best move (paper Eq. 1) |
| **No trivial capture** | Rejects positions where the best move captures an undefended piece, or captures a piece worth ≥3 more than the attacker |
| **No mate-in-1** | Rejects positions where depth-1 Stockfish already finds a forced mate |
| **Novelty** | Levenshtein distance from all positions in the replay buffer ≥ τ_board (5) for position, ≥ τ_pv (3) for best line |

where `w_deep` is the winning probability at depth-6 and `w2_deep` is the winning probability of the second-best move, computed via the formula `w = 1 / (1 + exp(-cp / 400))`.

---

## Reward Function

The reward combines two signals:

**Option A — Uniqueness reward**: `gap_reward = min(0.7, gap × 4)`

Rewards positions where the best move is unambiguous. Higher gap = more puzzle-like (one solution, not ambiguous). This signal is always non-zero for qualifying positions (gap ≥ 0.1), giving PPO a consistent gradient.

**Option B — Balanced position bonus**: `+0.3 if w_shallow < 0.55`

Rewards positions that look roughly equal at depth-1 (`w_shallow` = winning probability at depth-1). These are the positions where deep calculation reveals a hidden winning advantage — the definition of a counter-intuitive puzzle. This nudges the model toward the regime where genuine counter-intuitiveness is possible.

**Total reward**: `min(1.0, gap_reward + balance_bonus)`

Counter-intuitiveness (`r_cnt = w_deep − w_shallow`) is also tracked in logs as `r_cnt_uniq` and used as a secondary diagnostic signal.

---

## Training Setup

```
Model:          134M autoregressive transformer
Optimizer:      AdamW, lr=1e-6
PPO epochs:     4 per step
PPO clip:       ε=0.2
KL penalty:     0.3 × token-level KL from reference model
SL warmup:      SL coefficient 1.0→0.1 over 200 steps (prevents forgetting)
Batch size:     64
Tactical depth: 6 (Stockfish)
Replay buffer:  10,000 positions (novelty filter)
```

---

## Key Metrics (logged per step)

- `puzzle_rate`: fraction of batch positions that qualify and receive reward > 0
- `r_cnt_uniq`: average counter-intuitiveness (w_deep − w_shallow) across UniqueWin positions — the primary signal that RL is improving position quality
- `mean_gap`: average uniqueness gap for qualifying positions
- `n_balanced`: number of qualifying positions with w_shallow < 0.55
- `UniqueWin`: positions where the best move is unique AND the side to move is winning
