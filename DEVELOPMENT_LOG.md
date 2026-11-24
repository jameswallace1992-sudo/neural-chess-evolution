# Development log

## Milestone 1 — Foundation

The environment enforces chess legality and recognizes outcomes using
python-chess. The network receives a board and its available history, then
produces move scores and an outcome estimate. The action mask excludes illegal
moves before probabilities are calculated. Piece identities, chess rules and
the encoding are hardcoded; the network's numerical predictions are not
handcrafted. Its weights are random and it has learned nothing yet.

The fixed policy format covers castling, en passant and all four promotion
choices. The input retains up to 150 earlier board identities for draw
context; this increases computation and cannot reconstruct history omitted
from a FEN. Python-chess remains the source of truth for termination.

**Verify:** run `python -m unittest discover -s tests -v` and
`python -m neural_chess`. The latter should report 20 legal starting moves
whose probabilities sum to one. Random preferences and values do not demonstrate
playing strength, and each run starts from fresh random weights.

**Experiment:** run the demonstration several times. Compare the top moves and
value; they change because different random weights are used, not because
the network learned. Later, compare separately saved generations in balanced
matches to investigate learning rather than interpreting these random outputs.

## Milestone 2 — Search and self-play

PUCT search explores legal continuations using the network's move probabilities
and value estimates. Values are flipped when backing up across a move, so a
good outcome for one player is bad for the other. Terminal outcomes and legal
moves are rules; the network supplies all non-terminal strategic predictions.
Search changes move selection, not the network weights. Root noise and early
sampling encourage variety but can also make play less consistent; extra
simulations cost computation and are not guaranteed to help a random network.

Complete games and their per-colour budgets are archived as PGNs; session
settings are saved separately. Each session currently starts with a new
random model, and its hash identifies but does not restore the weights.
Claimable draws are not claimed. In-progress game recovery, model checkpoints,
and learning are not implemented.

**Verify:** run `python -m unittest discover -s tests -v` and the self-play
command in the README. Load the resulting PGN in chess software or replay it
with python-chess. Tests check perspective backup, a checkmate, legal PGN
replay, configuration recording and preservation of existing files.

**Experiment:** generate several games using symmetric and asymmetric
budgets, alternating the larger budget between colours. Compare results and
game lengths, recording computation as well as outcomes. At this stage the
network is random, and a few games cannot establish learning or strength.

## Milestone 3 — Diagnostic learning

Self-play now retains the search visit counts at every position in a completed
game. The replay buffer converts those counts into policy targets and assigns
the final result from the perspective of the player to move. The optimizer
adjusts the existing network's weights to predict those targets, with legal
move masking; it does not learn from an external engine or human games.

The initial and final model, optimizer state, retained replay examples, game
identifiers, configuration and losses are preserved for a successful
diagnostic run. The run is not resumable, and interruptions before the final
save can lose in-memory learning progress. Every new diagnostic initializes
fresh weights rather than continuing a prior run. Milestone 4 must
implement reliable resumption before this becomes a long-term training
workflow.

**Verify:** run the tests and the diagnostic command in the README. A completed
GPU run produced a legal 54-ply draw and four updates; all 35 parameter tensors
changed, and the final optimizer and replay were loaded on CPU. A separate CPU
diagnostic also completed. Synthetic tests check that optimization improves its
target predictions and that outcome signs alternate correctly.

**Experiment:** compare the same fixed diagnostic batch before and after a few
updates to see how its target probabilities and values change. More updates
may fit the small sample without making the engine stronger. Training loss is
not playing strength; evaluate that later using separate balanced games.

## Milestone 4 — Persistence

The selected GPU diagnostic became the main history: its model weights,
Adam state and replay buffer were imported without resetting them. The original
random weights are saved as generation 0 and the diagnostic result as
generation 1. The first resumed cycle produced generation 2; it does not
become a champion and is not proof of stronger chess play.

Completed self-play games are journaled before checkpoint advancement. Each
checkpoint preserves the model, optimizer, replay, progress, configuration,
and available random states. Checkpoints and generation snapshots are
non-overwriting and integrity-checked; a process lock prevents concurrent
writers. Ctrl+C or a stop request waits for the current game or update.
Crashes can lose an in-progress game or updates since the last checkpoint,
and hardware changes need not reproduce the same sequence. Backups remain
essential. The earlier diagnostic did not save its CUDA RNG state.

**Verify:** `python -m unittest discover -s tests -v` checks recovery,
optimizer continuity, corruption handling and CPU/CUDA portability. The
actual imported GPU run was checked against its source: all 35 weight
tensors, 54 replay positions and optimizer state matched. After resuming, its
update count increased from 4 to 5; the resulting checkpoint also loaded and
performed an update on CPU without changing the saved files.

**Experiment:** stop a short session at a safe boundary, inspect `status`,
then resume it without configuration overrides. Compare the saved update and
game counts before and after. The README documents commands and transfer steps.

## Exploration and reward experiment

The existing network architecture and optimizer are unchanged. The configured
experiment uses 25 simulations per side, compared with one in the original
diagnostic configuration. In a three-seed GPU probe of the initial position,
search visited 4–6 distinct root moves. This shows more varied search targets,
not stronger chess; substantially more computation is required per move.

For 20 completed games, root noise is 0.5 and move selection samples at
temperature 1.25 for up to 10,000 plies. Baseline noise and temperature
resume afterward, while the simulation budget and reward scale remain.
Exploration can produce greater variety but also weaker moves. The count and
actual settings are recorded; a cycle already underway finishes under its
original settings before the experiment starts.

Win/draw/loss values are scaled to +2/0/−2 in search and value loss. The
network output remains normalized and its architecture is unchanged. Scaling
changes the PUCT value/exploration balance and multiplies the squared value
loss and its gradient by four, including for draws; it does not penalize draws
relative to decisive outcomes. This may destabilize learning, and cannot
create a winning signal from drawn games. Track outcomes, repetition, game
length, search diversity, time and loss separately; balanced evaluation games
are needed before claiming stronger learned weights.

New checkpoints reference immutable, checksummed replay shards instead of
duplicating the whole buffer. Migration preserves the old checkpoint files
and copies the existing replay once. Model and optimizer snapshots still
consume disk space, and the previous 40 GiB of history remains in place.
Back up the entire training state, including replay shards, with the PGNs.

## Separate fresh history

A fresh history can be explicitly initialized in a new directory without
touching prior games or generation snapshots. It starts at generation 0 with
random weights, an empty Adam optimizer and empty replay. Its seed and initial
configuration are recorded, and its architecture is identical to the earlier
network. The 25-simulation exploration profile can then be configured before
training. Previous learned weights are not carried over; this is a restart,
not a resume or evidence that the earlier network was weaker.

## Self-play performance and visibility

Position encoding now builds the same 321 input planes without thousands of
individual tensor writes, and legal move probabilities are copied from the
GPU together rather than one at a time. These implementation changes do not
alter the network architecture, learning targets, search rules or chess rules.
Long games now report their elapsed time and half-move count between moves,
approximately every 15 seconds; this does not save an unfinished game.

On the laptop GPU, paired 25-simulation probes at a 150-ply position took
about 0.72 seconds with the original encoding and 0.25 seconds with the new
encoding. Replaying all 55 plies of the first archived game with 25 simulations
per move took 8.6 seconds. These are local samples, not guaranteed game times
or evidence of stronger play. Longer games still require more computation.

## Parallel self-play

Added `resume --workers N`: games within a cycle are played by worker processes using CPU copies of the unchanged network (weights are fixed while a cycle plays games). Single-position GPU inference was latency-bound (about 19.5 ms per evaluation on the laptop GPU versus 3.5 ms on one CPU thread in a short probe), so GPU batching was not used. Architecture, PUCT search, targets and reward scale are unchanged; per-game seeds derive from the cycle seed and game index. Settings overrides are now deferred to the next cycle rather than rejected. Throughput results below are local samples, not evidence of stronger play.

## Complete-game learning

Added the opt-in `fresh_pass` learning mode and `configure-learning` command.
Every new game's positions receive one shuffled training pass; each update
also samples up to an equal number of older positions from the replay buffer.
Game shards supply new examples even when the rolling buffer cannot hold an
entire cycle. The shuffled order, progress, optimizer and RNG states are
checkpointed. The default configuration increases replay capacity to 8,192
and checkpoints every eight updates; safe stops save immediately, while a
crash can lose up to seven updates. An explicit option can include an active
cycle only while it is still generating games and before any training updates.
Existing replay history is retained, but previously evicted positions are not
automatically added back to the buffer.

Parallel workers now publish each completed game journal themselves, even if
an earlier game has not finished. Journals are applied in game-index order.
Stopping waits for running workers; unjournaled games may be lost in a crash.
Serial and parallel games do not share the same random-sequence scheme.
These changes do not alter the network architecture, search, game rules,
reward scale or evaluation. One pass over new examples is an experiment, not
proof that the network learns stronger chess. GPU batching and controlled
generation-versus-generation evaluation remain separate future work.

Fresh-history initialization accepts explicit per-side simulation and reward
scale settings without requiring the temporary exploration profile. These
settings are recorded in the initial configuration; existing histories are
not modified.


### Experiment: draw claims (threefold repetition and 50-move rule)

Option `claim_draws` (default off, recorded in each cycle's self-play
configuration; `resume --claim-draws` / `--no-claim-draws`, applied at the
next cycle). When on, a game ends as a draw the moment a position occurs for
the third time or the halfmove clock reaches 100 (FIDE claimable draws, always
claimed immediately). Checkmate and stalemate take priority. The same rule is
used inside MCTS, so a claimable draw is scored 0 by search. Automatic fivefold
and 75-move draws remain in force either way.

Effect category: game rules / search terminal states, so it changes the data the
network learns from (shorter drawn games). It does NOT change the network
architecture, weights or optimiser. Expected benefit: faster learning per minute
and less shuffling. Possible disadvantage: repetition becomes an easier route
to a draw. Measure: decisive-game fraction, game length and checkmate rate per
10 games, before versus after the switch. Histories started without the flag
are unchanged; games from before and after are not directly comparable.


### Milestone: generation-versus-generation evaluation

`neural_chess/evaluation.py` plays colour-balanced matches between two saved
generation snapshots of one history. Affects evaluation methodology only: no
change to the architecture, learning, weights, optimiser or training search,
and no champion exists or is replaced (promotion is not implemented).

Design: each side searches with its own network and an equal simulation budget,
Dirichlet noise off, the first 8 plies sampled from visit counts (so games
differ) and later plies most-visited. The candidate plays white in even games
and black in odd games (an even game count is required). Snapshots are read
only, verified by integrity envelope and model hash. Results (config.json,
result.json, one PGN per game) go to a new `evaluations\<id>` directory and
are never overwritten. Statistics: score, 95% bootstrap interval, Elo
difference (infinite at 0% or 100%), colour split, termination counts. The
verdict says only whether the interval excludes equal strength.

Limits: evaluation uses CPU workers and competes with training for CPU. Small
matches have wide intervals; a result applies to the settings used (simulations,
claim_draws, seed) and does not prove general strength. The search budget and
reward scale are recorded with every match.

### Experiment: Stockfish as a graded opponent (separate history)

User-approved amendment to rules 2–3 (2026-11-24, recorded in
`.github\copilot-instructions.md`). Affects the learning process (data source
for a separate history) only: architecture, MCTS algorithm, loss and reward
(+1/0/-1) unchanged; the self-play main line is not modified.

Design: `initialize-opponent` copies a generation's weights and Adam state into
a new state directory (empty replay buffer; lineage in `initialization.json`).
The network plays alternate colours against a Stockfish ladder
(`neural_chess/opponent.py`, levels 0-9, nominal and uncalibrated). Stockfish
chooses only the opponent's moves; opponent plies carry empty visit counts and
are never training targets. The ladder advances at >=55% over the last 10 games
at a level. Each PGN records OpponentLevel/NetworkColour; ladder results are in
the checkpoint. Tests: `tests\test_opponent.py`.

Smoke test (temporary state, since deleted): generation 88, level 0, 2 games,
2 workers; the network lost the game inspected, so the earlier "misses free
pieces" observation is real even against a heavily degraded opponent.
Expected benefit: punishment for blunders. Risks: overfitting to this opponent's
style; a bad sampling of one-game cycles. Measure with the ladder score and
generation matches against the self-play line.

### Experiment: value-head reactivation

A sampled set of opponent-game positions showed the first value-head ReLU
inactive, with identical value predictions. The `neural_chess.value_repair`
command creates a separate history from a completed generation and adjusts only
the first value-head convolution's bias, based on up to 128 evenly sampled
replay positions. It retains the architecture, all other weights, optimizer
state and replay, and records the source, adjustment and measured value range.
It refuses an active head, empty replay or an unsuccessful reactivation. This
changes learned parameters and therefore predictions, but not the search or
evaluation rules. Continued training may deactivate it again; compare held-out
value predictions and balanced match results rather than assuming improvement.

On 2026-11-24, generation 128 from `training_state\stockfish-opponent` was
forked into `training_state\value-repair-20261124`. The bias changed from
-0.098066 to 2.000659; predictions varied by 0.008946 across the 128 sampled
replay positions. An in-memory training step changed the repaired bias without
writing to either history. This does not establish that the repair will remain
active during training or improve play.
