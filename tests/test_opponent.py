import contextlib
import io
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import torch

from neural_chess.opponent import LADDER, LADDER_WINDOW, network_score, next_level, open_opponent
from neural_chess.persistence import (
    initialize_fresh, initialize_from_generation, load_latest, resume,
)
from neural_chess.selfplay import SelfPlayConfig, network_colour_for, play_game
from neural_chess.training import ReplayBuffer

STOCKFISH = Path(os.environ["STOCKFISH_PATH"]) if os.environ.get("STOCKFISH_PATH") else None


def fools_mate_predictor(board):
    sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
    moves = list(board.legal_moves)
    target = chess.Move.from_uci(sequence[len(board.move_stack)])
    return {move: float(move == target) for move in moves}, 0.0


def scripted_white(board, rng):
    return chess.Move.from_uci(("f2f3", "e7e5", "g2g4")[len(board.move_stack)])


@contextlib.contextmanager
def random_opponent(level, path):
    yield lambda board, rng: rng.choice(list(board.legal_moves))


class OpponentGameTests(unittest.TestCase):
    def config(self, level=0):
        return SelfPlayConfig(noise_fraction=0, temperature=0, opponent_level=level, opponent_path="x")

    def test_opponent_plies_are_recorded_but_never_become_learning_targets(self):
        positions = []
        # The network is black in odd games; the scripted opponent plays white's moves.
        game = play_game(fools_mate_predictor, self.config(), random.Random(1), 1, 1, positions,
                         game_index=1, opponent_move=scripted_white)
        self.assertEqual(game.headers["Result"], "0-1")
        self.assertEqual(game.headers["NetworkColour"], "black")
        self.assertEqual([bool(p.visits) for p in positions], [False, True, False, True])
        buffer = ReplayBuffer(10)
        buffer.add_game(game, positions)
        self.assertEqual(len(buffer.examples), 2)
        self.assertEqual([e.value for e in buffer.examples], [1.0, 1.0])

    def test_opponent_plies_carrying_visits_are_rejected(self):
        positions = []
        game = play_game(fools_mate_predictor, self.config(), random.Random(1), 1, 1, positions,
                         game_index=1, opponent_move=scripted_white)
        positions[0].visits = {game.variations[0].move: 1}
        with self.assertRaises(ValueError):
            ReplayBuffer(10).add_game(game, positions)

    def test_colour_alternates_and_self_play_is_unchanged(self):
        config = self.config()
        self.assertEqual([network_colour_for(config, i) for i in range(3)],
                         [chess.WHITE, chess.BLACK, chess.WHITE])
        self.assertIsNone(network_colour_for(SelfPlayConfig(), 0))

    def test_opponent_settings_are_validated(self):
        with self.assertRaises(ValueError):
            SelfPlayConfig(opponent_level=len(LADDER))
        with self.assertRaises(ValueError):
            SelfPlayConfig(opponent_level=0)  # path missing

    def test_ladder_advances_only_after_a_full_window_at_the_current_level(self):
        good = [[2, 1.0, str(i)] for i in range(LADDER_WINDOW)]
        self.assertEqual(next_level(2, good[:-1]), 2)
        self.assertEqual(next_level(2, good), 3)
        self.assertEqual(next_level(3, good), 3)
        weak = [[2, 0.0, str(i)] for i in range(LADDER_WINDOW)]
        self.assertEqual(next_level(2, weak), 2)
        self.assertEqual(next_level(len(LADDER) - 1, [[len(LADDER) - 1, 1.0, "x"]] * 20), len(LADDER) - 1)
        self.assertEqual(network_score("0-1", False), 1.0)
        self.assertEqual(network_score("1/2-1/2", True), 0.5)


class OpponentHistoryTests(unittest.TestCase):
    def test_copy_generation_then_learn_from_opponent_games(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            initialize_fresh(root / "source", seed=1)
            source, _ = load_latest(root / "source")
            initialize_from_generation(root / "source", 0, root / "opp", opponent_path="x",
                                       opponent_level=1, seed=2)
            state, _ = load_latest(root / "opp")
            self.assertEqual(state["model_id"], source["model_id"])
            self.assertEqual(state["replay_refs"], [])
            with self.assertRaises(FileExistsError):
                initialize_from_generation(root / "source", 0, root / "opp", opponent_path="x")
            with patch("neural_chess.selfplay.open_opponent", random_opponent):
                resume(root / "opp", root / "games", torch.device("cpu"),
                       overrides={"games": 1, "batch_size": 4}, workers=1)
            after, _ = load_latest(root / "opp")
            self.assertEqual(after["opponent_results"][0][0], 1)
            self.assertEqual(after["last_selfplay_config"]["opponent_level"], 1)
            self.assertNotEqual(after["model_id"], state["model_id"])
            self.assertTrue(list((root / "opp" / "generations").glob("*.pt")))
            self.assertEqual(len(list((root / "source" / "generations").glob("*.pt"))), 1)


@unittest.skipUnless(STOCKFISH is not None and STOCKFISH.is_file(),
                     "Set STOCKFISH_PATH to run the optional engine smoke test")
class StockfishSmokeTests(unittest.TestCase):
    def test_every_level_returns_a_legal_move(self):
        board = chess.Board()
        for level in (0, 3):
            with open_opponent(level, str(STOCKFISH)) as choose:
                self.assertIn(choose(board, random.Random(level)), board.legal_moves)


if __name__ == "__main__":
    unittest.main()
