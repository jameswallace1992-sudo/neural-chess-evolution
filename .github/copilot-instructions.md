# Neural Chess Evolution — Permanent Development Rules

## Non-negotiable principles

1. This is a genuine self-learning chess neural network. Never substitute handcrafted chess intelligence for the requested learning system.
2. No Stockfish, Leela Chess Zero, other chess engines, opening books, tablebases, pretrained chess models, or expert-game training data, with ONE user-approved exception (amendment 2026-11-24): Stockfish may be used as a graded *opponent* in a separate, clearly labelled experiment (see rule 3). It must never supply evaluations, move choices for the network, policy/value targets, or be used inside MCTS.
3. The main line's only source of strategic learning is the project's own self-play experience. The Stockfish-opponent experiment is a separate, recorded history (own state/games directories, never mixed into the self-play main line) in which the network learns from the games it plays against Stockfish, using only the game outcome and its own MCTS policy targets.
4. Use python-chess for game legality and PGN handling, not engine-assisted evaluation.
5. All strategic predictions must originate from our own trainable PyTorch policy-value network, with MCTS used to improve move selection.
6. Preserve the main network's architecture across training generations.
7. Never reset learned weights or optimiser state when resuming training.
8. Never destroy historical checkpoints, PGNs or experiment results.
9. Never replace the current champion without an appropriately evaluated promotion decision.
10. Never treat a single win, loss, decreasing training loss or higher generation number as proof of improved playing strength.
11. Every training configuration and experimental change must be recorded.
12. Training must work locally, without an internet connection, and support both CUDA and CPU operation.
13. Before implementing major changes, explain whether they affect the network architecture, learning process, search algorithm or evaluation methodology.
14. Add tests for every important new feature, especially learning, move legality, data integrity and checkpoint recovery.
15. Do not implement future milestones before the current milestone works.
16. Prefer reliable, understandable code over unnecessary abstraction.

## Educational approach

Explain technical design decisions in accessible language, including what the neural network is actually learning and what remains hardcoded.

Distinguish improvements in search strength from improvements in learned network weights.

When discussing changes, explain the expected benefits, possible disadvantages and how we can measure their effect.

Treat this as a scientific experiment, not merely an application to complete.
