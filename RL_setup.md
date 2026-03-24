# RL Training Setup

Based on the DeepMind paper "Generating Creative Chess Puzzles" (arXiv 2510.23881).

## Task
An autoregressive transformer (134M params) generates FEN strings token-by-token from a zero start token. RL fine-tunes the pretrained model to produce positions that are tactically rich and novel.

## Reward
Binary: **-2** (illegal FEN), **0** (valid but fails checks), **+1** (qualifying puzzle).

A position qualifies as +1 if it passes all of:
1. Valid FEN with realistic piece counts
2. **Uniqueness** (`r_uni >= tau_uni = 0.3`): the best move's winning chance must exceed the second-best by ≥ 0.3, computed via Stockfish `multipv=2` at depth 10. This ensures there is one clearly winning move — the definition of a good puzzle.
3. **Entropy** (`>= tau_ent = 0.0`): disabled — a confident pretrained model has naturally low per-token entropy (0.1–0.3 nats), so any positive threshold silently blocks all positions.
4. **Novelty**: not an exact match to any of the 10k training positions seeded into the replay buffer, and Levenshtein distance ≥ 5 on board string / ≥ 3 on PV from any previously discovered position.

## Training Loop (PPO)
- Generate 32 sequences per step from the current policy (eval mode, temperature=1.0).
- Compute rewards and KL-shaped advantages: `adv = (reward - KL_penalty) - EMA_baseline`.
- PPO clipped objective (1 epoch per step) on generated sequences only.
- SL loss on 32 random dataset positions to prevent forgetting.
- Token-level KL penalty against frozen reference model keeps policy close to pretrained distribution.

## Replay Buffer
- **Seeded** at startup with 10k training positions (exact board strings). Generated positions that exactly match any seed get reward 0 — the model must produce genuinely new positions.
- Discovered qualifying positions are added to the buffer and checked with Levenshtein distance to prevent near-repetition.
- Buffer capacity: 10k discovered positions (FIFO).

## Key Hyperparameters
| Param | Value |
|---|---|
| LR | 1e-6 |
| Batch size | 32 |
| PPO clip ε | 0.2 |
| KL coeff | 0.1 |
| SL coeff | 0.1 |
| EMA α (baseline) | 0.1 |
| tau_uni | 0.1 |
| tau_board | 5 |
| tau_pv | 3 |
| tau_ent | 0.5 |
| Stockfish depth | 10 |

## What We Omit vs the Paper
- **Counter-intuitiveness** (`r_cnt`): the paper's secondary reward signal measuring how deep Stockfish must search before agreeing with the solution. Omitted for simplicity.
- **200M model**: we use 134M.
