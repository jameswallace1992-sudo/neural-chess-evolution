# Neural Chess Evolution

An offline chess-learning research project. **Milestones 1–4** provide
standard chess rules, position and move encodings, a trainable PyTorch
policy-value network, PUCT search, archived self-play games and resumable
training. Evaluation tournaments and a dashboard are not implemented yet.

Training histories, checkpoints and saved games are not included. A new clone
starts without trained weights; use the fresh-history instructions below.

## Structure and planned design

- `neural_chess/environment.py`: legal moves and outcomes via python-chess.
- `neural_chess/encoding.py`: stable position and action formats.
- `neural_chess/network.py`: one shared residual network with policy and value heads.
- `neural_chess/mcts.py`: PUCT selection, expansion and value backup.
- `neural_chess/selfplay.py`: self-play and non-overwriting game archives.
- `neural_chess/training.py`: bounded replay and a diagnostic training cycle.
- `neural_chess/persistence.py`: resumable main training history.
- `neural_chess/__main__.py`: random-network demonstration.
- `tests/`: foundation tests.
- `DEVELOPMENT_LOG.md`: milestone decisions and experiments.

Later milestones will add evaluation and experiments in that order. Search
selects between network predictions and actual game
outcomes; it does not change network weights. Training uses self-play data and
an optimizer to update those weights.

## Setup and demonstration

Use Python 3.10 or later. In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m neural_chess
```

Installing dependencies requires access to packages, but the demonstration and
tests do not require internet access or an external chess engine once installed.
The optional Stockfish opponent smoke test is skipped unless `STOCKFISH_PATH`
names an available executable.
The demonstration chooses CUDA if the installed PyTorch build can use it;
otherwise it uses CPU. For CUDA, install an appropriate Windows PyTorch build
for the local driver and GPU; the default installation is not guaranteed to
include CUDA. Each demonstration starts with new random weights, so its outputs
are not trained chess assessments and vary between runs.

## Self-play

```powershell
python -m neural_chess.selfplay --games 1 --white-simulations 1 --black-simulations 1
```

Each self-play-only invocation creates **new, random and untrained weights** shared by both
players for that session. This standalone command does not save or resume its model.
The model identifier in each PGN is a SHA-256 hash of the serialized state
dict, not a saved checkpoint; it identifies what was used but cannot restore
it. Do not interpret these games as training or evidence of improved strength.
Use the resumable command below to continue the main training history.

Each game is played from the standard starting position until an automatic
chess outcome, without a move limit or external engine. Claimable draws are
not claimed; automatic fivefold repetition and 75-move draws still apply.
Games can take a long time, especially with larger search budgets. Completed
games are written separately to `games\selfplay` as PGNs, with UUIDs, model
identifier, UTC finish time, result, ply count and actual per-colour budgets.
A session JSON file records the seed and all search settings. Both file types
are published without overwriting existing files, using a flushed temporary
file and an atomic hard link on a filesystem that supports hard links.
Archives should be backed up; hard links and filesystem caches do not guarantee
durability in every power failure. Incomplete standalone games are not saved.

Use `--white-simulations 50 --black-simulations 400 --alternate-budgets` to
swap those budgets between colours on successive games. The defaults are
symmetric and deliberately small. `--exploration`, `--noise-fraction`,
`--dirichlet-alpha`, `--temperature` and `--temperature-plies` configure PUCT
and self-play exploration. `--seed` records the random seed; changing devices
or library versions can prevent exact reproduction. Use `--output-dir` to
select another archive folder. These options configure self-play, not
training sessions.

Each search first evaluates the root, then runs the requested number of
simulations, each visiting one root move. Network priors guide expansion;
PUCT balances estimated outcomes against exploring under-visited moves.
Values change sign at each ply because the players alternate. Terminal
values come from actual outcomes. Root Dirichlet noise encourages exploration,
and root visit counts determine move probabilities; after the configured
number of plies, the most-visited move is selected deterministically.
The learning cycle below records the unmodified root visit counts, independently
of the temperature used to select the move.

## Diagnostic learning cycle

```powershell
.\.venv\Scripts\python.exe -m neural_chess.training --games 1 --updates 4 --batch-size 16
```

This command starts a **new randomly initialized model every time**. It is a
diagnostic, not a way to continue an existing training history. Use the
resumable command below for the main network instead.
Each run saves its starting weights, configuration, completed-game PGNs,
and—after the last update—its final weights, Adam optimizer state, retained
replay examples and loss history in a unique `training_runs` folder.
These artifacts are not yet resumable checkpoints. If interrupted before the
final save, completed PGNs and the starting model remain, but completed
updates and in-memory replay may be lost. Completed diagnostic artifacts can
be imported explicitly into a new main history. Only load trusted PyTorch files.

For each completed game, replay stores positions, legal actions, raw MCTS
visit counts and the outcome from the player-to-move perspective: +1 for a
win, 0 for a draw, −1 for a loss. Each policy target is the visit counts
normalized to sum to one. The training step minimizes cross-entropy against
that policy over **legal moves only**, plus mean squared error against the
outcome; Adam updates both heads and the shared body by backpropagation.
The buffer keeps the most recent `--replay-capacity` positions; older PGNs
remain archived, but evicted diagnostic visit counts are not retained. Batches sample
from the buffer without replacement within each update.

Use `--games`, `--updates`, `--batch-size`, `--replay-capacity`,
`--learning-rate`, `--white-simulations`, `--black-simulations`,
`--alternate-budgets`, and `--seed` to configure a diagnostic. Other
self-play settings use their recorded defaults. A single simulation and four
updates are only a pipeline check, not a useful training regimen. Losses are
measured on sampled training batches; they are not validation metrics and
cannot establish stronger play. Separate balanced matches are still needed.

## Resume the main network

The main history was initialized from the selected GPU diagnostic; that
diagnostic's trained weights, Adam state and replay buffer were preserved.
The original random model is saved as generation 0, and the diagnostic's
trained model is generation 1. Generation 2 was produced by the first
resumed cycle. The other diagnostic remains separate and untouched.

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence status
.\.venv\Scripts\python.exe -m neural_chess.persistence resume
```

`resume` refuses to create a new model when no checkpoint exists. By default
it finishes an interrupted cycle, or runs one new cycle if the previous one
finished. For an approximately 30-minute session:

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence resume --minutes 30
```

It repeats cycles until the time limit, then stops at the next safe boundary:
a long game can exceed the requested time. Press Ctrl+C once to request the
same safe stop, or in a second PowerShell window run:

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence stop
```

Each completed game is journaled with its MCTS visit counts and RNG state,
then archived as PGN before the checkpoint advances. If interrupted after
the journal was published, resuming recovers that game rather than playing
it again. A game interrupted before its journal is committed cannot be
recovered. The current game is not checkpointed mid-play.

During longer games, progress is printed between moves approximately every
15 seconds, showing elapsed time and plies (half-moves). A particularly slow
move can delay the next message. This is progress reporting, not a game
checkpoint or a limit on game length.

Checkpoints contain network weights, Adam state, replay references, counters,
configuration, and available RNG states. Replay examples are saved once in
immutable, checksummed shards and referenced by subsequent checkpoints.
The existing embedded-replay checkpoints are still readable and are not
deleted. The first migration copies the existing replay once; later completed
games each add one shard. A referenced checkpoint requires its shards, so
back up the entire `training_state` directory. There is no scheduler yet. A
checkpoint is written after each game, after each update by default, and at
a safe stop. Historical checkpoints and generation snapshots are never
overwritten. Main checkpoints, game journals and generation checkpoints use
checksums and atomic non-overwriting publication; the imported generation 0
is a model-only copy of the original diagnostic. These files
require hard-link support and do not replace backups or guarantee
survival of every sudden power failure. Disk use grows with training.

To configure a **new** cycle, pass options such as:

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence resume --games 2 --updates 4 --batch-size 16 --white-simulations 10 --black-simulations 10 --checkpoint-interval 1
```

`--checkpoint-interval` is measured in optimizer updates; after a crash a
larger interval can lose updates since the last checkpoint. The default is
one. Game checkpoints and safe stops are always saved. You can also set
`--learning-rate` and `--alternate-budgets`; use `--fixed-budgets` to
stop alternating. Set both budgets equal for symmetric search. Settings are
recorded per cycle.
An unfinished cycle normally finishes with its saved settings; configuration
overrides given while one is unfinished are applied when the next cycle starts
in the same invocation. If that invocation stops before the next cycle starts,
pass the overrides again.
`--minutes` and `--device cpu` or `--device cuda` can be used when resuming.

### Learning from complete games

The original `sampled` mode runs a fixed number of updates on random replay
positions. To schedule `fresh_pass` mode and enlarge the replay buffer:

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence configure-learning --state-dir training_state\restart
```

By default this applies to the next cycle. To include an unfinished cycle that
is still generating games and has performed no training updates, add
`--include-untrained-cycle`. This records the change and keeps the existing
weights, optimizer and completed games. It refuses to modify a cycle that has
already entered training.

In `fresh_pass` mode each position from every new game in the cycle appears
exactly once in a shuffled training pass, even if the replay buffer is too small
to hold the whole cycle. Each update uses up to `--batch-size` new positions
and samples up to the same number of older positions still in the replay buffer.
It may therefore use twice the specified batch size. The number of updates is
determined by the number of new positions, not `--updates`. Older positions
already evicted from the replay buffer are not automatically reintroduced,
although their archived shards are preserved. No playing-strength improvement
is guaranteed.

The default capacity is 8,192 positions, configurable with
`--replay-capacity` on `configure-learning`; it cannot be reduced through that
command. Encoded positions alone can use about 642 MiB at full capacity, plus
other memory and disk overhead. The default checkpoint interval for the new
mode is eight updates, configurable using `--checkpoint-interval`: a crash
can lose up to seven updates, whereas a safe stop saves progress. Existing
checkpoints and replay sources are never deleted.

### Parallel self-play

`resume --workers N` (default: half the CPU cores, up to 8) plays up to N games
of a cycle at once in separate processes. Weights do not change while a cycle
plays games, so each worker uses a CPU copy of the same network; the GPU/CPU
workers journal completed games independently; the main process applies them
in order and trains afterwards. Parallel games derive their seeds from the cycle
seed and game index. `--workers 1` uses the serial path, which uses a different
random sequence and may use a different inference device, so switching paths
does not guarantee identical games. The architecture, PUCT search, targets and
reward scale are unchanged. On a safe stop the process waits for running
workers to finish; their journals are recovered on resume. A sudden process or
power failure can still lose games that had not yet been journaled.

If the latest checkpoint is corrupt, normal resume fails rather than silently
starting again. Review and back up the history before explicitly using
`resume --recover-previous`: this may roll back progress since the older
checkpoint, while preserving the corrupt file. A process lock prevents two
training processes writing the same history at once.

For another computer, stop training cleanly and copy `training_state`,
`games`, and `training_runs` together, preserving their relative paths.
Install dependencies there, then run `resume`; CPU is used automatically
when CUDA is unavailable. Checkpoints load tensors through CPU before moving
the model and optimizer to the selected device. Hardware changes do not
guarantee identical random sequences or results. The original diagnostic did
not save CUDA RNG state, so its first resumed CUDA run warns about that.
Only load trusted checkpoint files. A separate champion and promotion
decisions are not implemented; training never replaces a champion.

### Starting a separate fresh history

Use this only when intentionally starting over, not when resuming learned
weights. The destination must not already exist. Initialization creates
random generation 0 weights, a new empty optimizer and empty replay buffer;
it records the seed and configuration. It does not change older histories.
Use `--simulations 25 --reward-scale 2` on `initialize-fresh` to set those
search and reward settings from the beginning without scheduling a special
exploration phase. Defaults remain one simulation and reward scale 1.

```powershell
.\.venv\Scripts\python.exe -m neural_chess.persistence initialize-fresh --state-dir training_state\restart
.\.venv\Scripts\python.exe -m neural_chess.persistence configure-exploration --state-dir training_state\restart
.\.venv\Scripts\python.exe -m neural_chess.persistence resume --state-dir training_state\restart --minutes 30
```

For this history, use `--state-dir training_state\restart` with `status`,
`stop`, and every subsequent `resume`. Games are still archived in the
shared `games\selfplay` directory with unique identifiers. `--seed` on
`initialize-fresh` optionally records a specific initialization seed.

### Generation-versus-generation matches

Measure whether a newer generation beats an older one. It only reads saved
snapshots and never changes weights, histories or any champion; it does not
promote anything. Safe to run while training (it uses CPU workers, so it
slows training somewhat).

```powershell
.\.venv\Scripts\python.exe -m neural_chess.evaluation --state-dir training_state\fresh-20261124 --candidate 48 --baseline 0 --games 40 --simulations 25 --workers 4 --claim-draws
```

Both sides use their own network with equal simulations and no noise; the
first 8 plies are sampled so games differ; colours alternate (games must be
even). Results go to `evaluations\<id>` (config, result.json, PGNs). Use the
same `--claim-draws` setting as training. Read the 95% interval: if it
contains 0.5, the match did not show a difference. A single match is limited
evidence, not proof of strength.

### Draw claims

By default games end only on automatic draws (fivefold repetition, 75 moves),
as in FIDE rules. `resume --claim-draws` makes self-play and MCTS also end the
game as a draw at threefold repetition or the 50-move rule, claimed as soon as
available (checkmate takes priority). It is recorded in the cycle
configuration and applies from the next cycle; weights are untouched.
`--no-claim-draws` switches it off again. This changes the training data, not
the architecture; compare decisive-game fraction and game length before and
after.

### Value-head repair experiment

If the value head is inactive on all sampled replay positions, a separate
history can be created from a completed generation:

```
python -m neural_chess.value_repair --source-state-dir training_state\stockfish-opponent --generation N --state-dir training_state\value-repair
```

This keeps the source history unchanged. It copies its optimizer, replay and
other weights, then increases the value head's first bias enough to activate
all sampled positions. The adjustment and source are recorded in
`initialization.json`. It refuses to run on an empty replay or an already
active head. The change can alter predictions immediately and does not prove
stronger play; monitor value predictions and compare playing strength.

### Stockfish-opponent experiment (separate history)

By user decision (rules 2–3 amended 2026-11-24), Stockfish may be a graded
*opponent* in a separate history. The main self-play history is never touched.

```
python -m neural_chess.persistence initialize-opponent --source-state-dir training_state\fresh-20261124 --generation 88 --state-dir training_state\stockfish-opponent --stockfish-path "<path to stockfish.exe>" --level 0
python -m neural_chess.persistence resume --state-dir training_state\stockfish-opponent --games-dir games\stockfish-opponent --minutes 30 --claim-draws
```

Replace the example generation number and source directory with a completed
generation in your own training history.

What it is: the new history starts from an exact copy of that generation's
weights and Adam state, with an empty replay buffer. The network plays white in
even games and black in odd games, searching with its own MCTS. Stockfish only
chooses the opponent's moves: it gives no evaluations, no targets, and is never
used inside MCTS. Only the network's own plies become training examples, using
its own visit counts and the game result (+1/0/-1). Opponent plies are stored
in the PGN but never learned from as targets (the network still sees them in
its input history).

Ladder (nominal, uncalibrated; Stockfish 18 cannot go below 1320 Elo): levels
0-2 use Skill Level 0 with one search node and 60%/30%/10% random moves;
levels 3-9 use UCI_Elo 1320, 1500, 1700, 1900, 2100, 2300, 2500 at 0.05 s per
move. The level rises by one when the network's last 10 games at the current
level score at least 55% (win 1, draw 0.5). Results and level changes are
stored in the checkpoint (`opponent_results`, `opponent_ladder_history`) and
every PGN records `OpponentLevel` and `NetworkColour`. `--opponent-level N`
forces a level (it also overrides the automatic advance for that cycle).

Limits: this changes the learning process (data source). Results measure
strength against this opponent only; use generation matches to compare with
the self-play line. Stockfish's Elo labels do not transfer exactly to this
ladder or to other rating systems.

## Exploration and reward experiment

The configured profile uses **25 simulations for each side**. For 20 completed
games it raises root noise from 0.25 to 0.5 and samples visit counts at
temperature 1.25 for the first 10,000 plies. After those 20 games it
automatically returns to the recorded baseline noise and temperature settings;
the 25-simulation budget and reward scale remain. The count survives safe
stops and recovery, and each game records its actual mode and settings. If a
cycle was already underway when configured, it first finishes with its old
settings; the exploration count starts with the next cycle.

The experiment uses **+2 for a win, 0 for a draw, −2 for a loss**. The
network's architecture and its normalized value output in [−1, 1] are
unchanged: search multiplies its prediction and terminal outcome by 2, and
training applies the same scale to predictions and targets before calculating
value loss. Consequently value loss is four times the unscaled squared error;
losses from the two reward scales are not directly comparable. This also
changes the balance between PUCT's value and exploration terms and can make
training less stable. It increases the weight of the value loss for draws too;
it does not penalize draws or change their value relative to wins and losses.
Scaling outcomes cannot teach wins when all games are draws; it is a recorded
experiment, not evidence of improvement. The higher simulation count can make
each game much slower.

The existing 40 GiB of historical checkpoints is deliberately retained.
New checkpoints no longer duplicate the full replay buffer, but model,
optimizer, and generation snapshots still take space. Monitor disk usage.

## Encodings and architecture

Only standard chess is supported, not Chess960 or variants. The input is
`[batch, 321, 8, 8]`, with rank 1 at row 0 and file a at column 0; the board
is never flipped. Channels 0–11 are the current position's white then black
pawn, knight, bishop, rook, queen and king occupancy. Each of the next 150
pairs represents one earlier board, newest first:

1. Signed piece type divided by 8 on each occupied square, white positive,
   black negative, empty zero.
2. Zero if that historical board is unavailable; otherwise a constant
   `1 + castling_rights / 16`, with 2 added at the legal en-passant target
   square if one exists. Castling bits 0–3 are white kingside, white queenside,
   black kingside, black queenside. The side to move is inferred from the
   current side and history age.

The final nine planes are side to move (1 for white), the four current
castling rights in the same order, current en-passant target as a one-hot
plane, halfmove clock divided by 150 (clipped at 150), and current twofold
and threefold repetition indicators. The 150-position window includes the
reversible history relevant before the automatic 75-move draw in a normally
played game. The full python-chess move stack remains authoritative for rules.
If a board is loaded from FEN without its earlier moves, those moves and their
repetition rights cannot be recovered: use a complete move stack for games
where repetition matters. Claimable draws are distinct from automatic draws;
the environment checks claims only when explicitly requested. Self-play does
not claim them; optional draw claims are not move actions.

There are 20,480 policy logits: `((destination * 5 + promotion) * 64 + origin)`,
where squares use python-chess numbering (a1 = 0), and promotion is none,
knight, bishop, rook or queen in that order. Castling and en passant use their
normal from/to squares. Unused and illegal entries are masked before softmax.
The null move's index is unused; decoding an index alone does not establish
legality. For the starting position the mask admits exactly 20 moves.

The network has **506,050 trainable parameters**, a 64-channel convolutional
stem, four 64-channel residual blocks and separate policy and value heads.
Output shapes are `[batch, 20480]` logits and `[batch, 1]` values in [-1, 1].
The value is intended as the outcome from the current player's perspective.
Group normalization is independent of batch size. The network's numbers come
from its weights, not a handcrafted piece-value or positional evaluation.

The 150-board history avoids silently omitting draw-relevant positions but
increases memory, input conversion work and inference cost. This must be
measured in later self-play; do not silently change the architecture once
training histories exist.

## Important future engineering risks

- **MCTS performance:** evaluating individual positions can underutilize the
  GPU. Measure throughput before choosing batching or multiprocessing.
- **Windows:** worker processes use spawn; future entry points must protect
  process startup and avoid loading redundant GPU model copies.
- **Recovery:** a future checkpoint must atomically preserve model, optimizer,
  replay data and progress; keep known-good snapshots and explicitly fail
  when a requested resume is missing or corrupt. File moves, interruptions,
  and CPU/CUDA portability need dedicated tests.
- **Experimental integrity:** future matches must use equal search budgets,
  alternating colours and enough games to justify promotion. Lower loss or
  increased search alone does not prove that learned weights are stronger.

See the development log for an experiment to explore this milestone.
