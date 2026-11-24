import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import chess
import torch

from neural_chess.evaluation import (
    candidate_score, elo_from_score, load_generation, play_match_game, run_match, summarise,
)
from neural_chess.persistence import _model_id_for_state, _pack, _read, initialize_fresh
from neural_chess.selfplay import write_new_file


def scripted(board):
    sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
    target = chess.Move.from_uci(sequence[len(board.move_stack)])
    return {move: float(move == target) for move in board.legal_moves}, 0.0


class StatisticsTests(unittest.TestCase):
    def test_scores_follow_colour_and_result(self):
        self.assertEqual(candidate_score("1-0", True), 1.0)
        self.assertEqual(candidate_score("1-0", False), 0.0)
        self.assertEqual(candidate_score("0-1", False), 1.0)
        self.assertEqual(candidate_score("1/2-1/2", False), 0.5)
        self.assertAlmostEqual(elo_from_score(0.5), 0.0)
        self.assertEqual(elo_from_score(1.0), float("inf"))

    def test_small_samples_are_inconclusive_and_clear_samples_are_not(self):
        few = summarise([1.0, 0.5, 0.0, 1.0], [True, False, True, False], ["CHECKMATE"] * 4)
        self.assertEqual((few["candidate_wins"], few["draws"], few["candidate_losses"]), (2, 1, 1))
        self.assertAlmostEqual(few["score"], 0.625)
        self.assertTrue(few["verdict"].startswith("inconclusive"))
        many = summarise([1.0] * 30, [True, False] * 15, ["CHECKMATE"] * 30)
        self.assertTrue(many["verdict"].startswith("candidate scored higher"))
        lost = summarise([0.0] * 30, [True, False] * 15, ["CHECKMATE"] * 30)
        self.assertTrue(lost["verdict"].startswith("candidate scored lower"))
        with self.assertRaises(ValueError):
            summarise([], [], [])


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_dir = self.root / "state"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(self.state_dir, seed=5, simulations=1)
        first = self.state_dir / "generations" / "generation-00000000.pt"
        state = copy.deepcopy(_read(first))
        generator = torch.Generator().manual_seed(1)
        state["model"] = {name: tensor + 0.05 * torch.randn(tensor.shape, generator=generator)
                          if tensor.is_floating_point() and "running" not in name else tensor
                          for name, tensor in state["model"].items()}
        state["generation"] = 1
        state["model_id"] = _model_id_for_state(state["model"])
        self.second = self.state_dir / "generations" / "generation-00000001.pt"
        write_new_file(self.second, _pack(state))

    def test_scripted_game_uses_each_side_predictor_and_is_legal(self):
        game = play_match_game(scripted, scripted, 1, seed=1, temperature_plies=0)
        self.assertEqual(game.headers["Result"], "0-1")
        self.assertEqual(game.headers["PlyCount"], "4")

    def test_snapshot_loading_checks_integrity(self):
        self.assertEqual(load_generation(self.state_dir, 1)["generation"], 1)
        with self.assertRaises(ValueError):
            load_generation(self.state_dir, 7)
        tampered = copy.deepcopy(_read(self.second))
        first_name = next(name for name, value in tampered["model"].items() if value.is_floating_point())
        tampered["model"][first_name] = tampered["model"][first_name] + 1
        other = self.state_dir / "generations" / "generation-00000002.pt"
        tampered["generation"] = 2
        write_new_file(other, _pack(tampered))
        with self.assertRaises(ValueError):
            load_generation(self.state_dir, 2)

    def test_match_is_balanced_recorded_reproducible_and_read_only(self):
        before = {path: path.read_bytes() for path in (self.state_dir / "generations").glob("*.pt")}
        evaluations = self.root / "evaluations"
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_match(self.state_dir, 1, 0, games=2, simulations=1, seed=3, claim_draws=True,
                              evaluations_dir=evaluations)
            second = run_match(self.state_dir, 1, 0, games=2, simulations=1, seed=3, claim_draws=True,
                               evaluations_dir=evaluations)
        result = json.loads((first / "result.json").read_text())
        config = json.loads((first / "config.json").read_text())
        self.assertEqual(result["games"], 2)
        self.assertEqual(result["candidate_wins"] + result["draws"] + result["candidate_losses"], 2)
        self.assertEqual(config["candidate_generation"], 1)
        self.assertIn("candidate_model_id", config)
        self.assertTrue(config["claim_draws"])
        pgns = sorted((first / "games").glob("*.pgn"))
        self.assertEqual(len(pgns), 2)
        self.assertIn('[White "Generation 1"]', pgns[0].read_text())
        self.assertIn('[White "Generation 0"]', pgns[1].read_text())
        for one, two in zip(pgns, sorted((second / "games").glob("*.pgn"))):
            self.assertEqual(one.read_text().split("\n\n", 1)[1], two.read_text().split("\n\n", 1)[1])
        self.assertNotEqual(first, second)
        after = {path: path.read_bytes() for path in (self.state_dir / "generations").glob("*.pt")
                 if path in before}
        self.assertEqual(before, after)

    def test_rejects_unbalanced_or_identical_matches(self):
        for kwargs in ({"games": 3}, {"games": 0}):
            with self.assertRaises(ValueError):
                run_match(self.state_dir, 1, 0, simulations=1, evaluations_dir=self.root / "e", **kwargs)
        with self.assertRaises(ValueError):
            run_match(self.state_dir, 1, 1, games=2, simulations=1, evaluations_dir=self.root / "e")


if __name__ == "__main__":
    unittest.main()
