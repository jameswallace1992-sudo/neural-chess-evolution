"""Optional Stockfish opponent ladder for the separate opponent experiment.

Stockfish only chooses the moves of the opposing side. It never evaluates positions for
the network, never supplies policy or value targets, and is never used inside MCTS.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random

import chess
import chess.engine

LADDER_WINDOW = 10
LADDER_THRESHOLD = 0.55


@dataclass(frozen=True)
class Level:
    random_probability: float  # chance the opponent plays a uniformly random legal move
    elo: int | None            # UCI_Elo when Stockfish's own strength limiter is used
    seconds: float             # thinking time per move for Elo-limited levels


# Nominal, uncalibrated strength. Stockfish 18 cannot be limited below 1320 Elo, so the
# lowest levels use Skill Level 0 with one search node and random moves mixed in.
LADDER = [
    Level(0.60, None, 0.0), Level(0.30, None, 0.0), Level(0.10, None, 0.0),
    Level(0.0, 1320, 0.05), Level(0.0, 1500, 0.05), Level(0.0, 1700, 0.05),
    Level(0.0, 1900, 0.05), Level(0.0, 2100, 0.05), Level(0.0, 2300, 0.05),
    Level(0.0, 2500, 0.05),
]


def validate_level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(LADDER):
        raise ValueError(f"opponent_level must be an integer from 0 to {len(LADDER) - 1}")


def network_score(result: str, network_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if (result == "1-0") == network_is_white else 0.0


def next_level(level: int, results: list) -> int:
    """Advance one level once the last LADDER_WINDOW games at this level score well enough."""
    scores = [entry[1] for entry in results if entry[0] == level][-LADDER_WINDOW:]
    if (level + 1 < len(LADDER) and len(scores) == LADDER_WINDOW
            and sum(scores) / LADDER_WINDOW >= LADDER_THRESHOLD):
        return level + 1
    return level


@contextmanager
def open_opponent(level: int, path: str):
    """Yield a function choosing the opponent's move for a board."""
    validate_level(level)
    settings = LADDER[level]
    executable = Path(path)
    if not executable.is_file():
        raise FileNotFoundError(f"Stockfish executable not found: {path}")
    engine = chess.engine.SimpleEngine.popen_uci(str(executable))
    try:
        if settings.elo is not None:
            engine.configure({"Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": settings.elo})
            limit = chess.engine.Limit(time=settings.seconds)
        else:
            engine.configure({"Threads": 1, "Skill Level": 0})
            limit = chess.engine.Limit(nodes=1)

        def choose(board: chess.Board, rng: random.Random) -> chess.Move:
            if rng.random() < settings.random_probability:
                return rng.choice(list(board.legal_moves))
            return engine.play(board, limit).move

        yield choose
    finally:
        engine.quit()
