import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn

from neural_chess.mcts import search, terminal_outcome, visit_policy
from neural_chess.network import PolicyValueNetwork
from neural_chess.selfplay import SelfPlayConfig, budgets_for_game, play_game, run_session, write_new_file
from neural_chess.training import ReplayBuffer


def fixed_predictor(board):
    moves = list(board.legal_moves)
    return {move: 1.0 / len(moves) for move in moves}, 0.6


def fools_mate_predictor(board):
    sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
    moves = list(board.legal_moves)
    target = chess.Move.from_uci(sequence[len(board.move_stack)])
    return {move: float(move == target) for move in moves}, 0.0


class SearchTests(unittest.TestCase):
    def test_backup_alternates_value_perspective(self):
        board = chess.Board()
        root = search(board, fixed_predictor, 1, exploration=0)
        child = next(child for child in root.children.values() if child.visits)
        self.assertAlmostEqual(child.mean_value, 0.6)
        self.assertAlmostEqual(root.mean_value, -0.6)
        def focused_predictor(position):
            moves = list(position.legal_moves)
            return {move: float(move == moves[0]) for move in moves}, 0.6

        root = search(board, focused_predictor, 2, exploration=100)
        child = next(child for child in root.children.values() if child.visits)
        self.assertEqual(root.visits, 2)
        self.assertEqual(sum(child.visits for child in root.children.values()), 2)
        self.assertEqual(child.visits, 2)
        self.assertAlmostEqual(child.value_sum, 0)
        self.assertAlmostEqual(root.value_sum, 0)
        self.assertEqual(board.fen(), chess.Board().fen())

    def test_terminal_checkmate_uses_actual_outcome(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        mate = chess.Move.from_uci("f7g7")

        def predictor(position):
            self.assertFalse(position.is_game_over())
            return {move: float(move == mate) for move in position.legal_moves}, 0

        root = search(board, predictor, 1)
        self.assertEqual(root.children[mate].visits, 1)
        self.assertEqual(root.children[mate].mean_value, -1)
        self.assertEqual(root.mean_value, 1)
        scaled = search(board, predictor, 1, value_scale=2)
        self.assertEqual(scaled.children[mate].mean_value, -2)
        self.assertEqual(scaled.mean_value, 2)

    def test_scaled_nonterminal_value_preserves_perspective(self):
        root = search(chess.Board(), fixed_predictor, 1, value_scale=2)
        child = next(child for child in root.children.values() if child.visits)
        self.assertAlmostEqual(child.mean_value, 1.2)
        self.assertAlmostEqual(root.mean_value, -1.2)

    def test_scaled_draw_still_has_zero_value(self):
        board = chess.Board()
        sequence = ("g1f3", "g8f6", "f3g1", "f6g8") * 4
        for uci in sequence[:-1]:
            board.push_uci(uci)
        target = chess.Move.from_uci(sequence[-1])

        def predictor(position):
            return {move: float(move == target) for move in position.legal_moves}, 0.6

        root = search(board, predictor, 1, value_scale=2)
        self.assertEqual(root.children[target].visits, 1)
        self.assertEqual(root.mean_value, 0)

    def test_noise_and_visit_policy(self):
        board = chess.Board()
        root = search(board, fixed_predictor, 30, noise_fraction=0.25, rng=random.Random(5))
        repeat = search(board, fixed_predictor, 30, noise_fraction=0.25, rng=random.Random(5))
        self.assertEqual([n.prior for n in root.children.values()], [n.prior for n in repeat.children.values()])
        self.assertAlmostEqual(sum(n.prior for n in root.children.values()), 1)
        self.assertNotEqual(len({n.prior for n in root.children.values()}), 1)
        for temperature in (0, 1, 0.00000000000000001):
            policy = visit_policy(root, temperature)
            self.assertEqual(set(policy), set(board.legal_moves))
            self.assertAlmostEqual(sum(policy.values()), 1)
        self.assertEqual(sum(p == 1 for p in visit_policy(root, 0).values()), 1)

    def test_rejects_bad_inputs_and_predictor(self):
        board = chess.Board()
        for simulations in (0, -1):
            with self.assertRaises(ValueError):
                search(board, fixed_predictor, simulations)
        with self.assertRaises(ValueError):
            search(board, lambda b: ({}, 0), 1)
        with self.assertRaises(ValueError):
            search(board, fixed_predictor, 1, noise_fraction=float("nan"))
        with self.assertRaises(ValueError):
            search(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), fixed_predictor, 1)


class DrawClaimTests(unittest.TestCase):
    @staticmethod
    def shuffling_predictor(board):
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        moves = list(board.legal_moves)
        target = chess.Move.from_uci(cycle[len(board.move_stack) % 4])
        return {move: float(move == target) for move in moves}, 0.0

    def test_claimed_threefold_ends_the_game_while_default_rules_continue(self):
        claimed = play_game(self.shuffling_predictor,
                            SelfPlayConfig(noise_fraction=0, temperature=0, claim_draws=True),
                            random.Random(1), 1, 1)
        self.assertEqual(claimed.headers["Result"], "1/2-1/2")
        self.assertEqual(claimed.headers["Termination"], "THREEFOLD_REPETITION")
        self.assertEqual(claimed.headers["PlyCount"], "8")
        automatic = play_game(self.shuffling_predictor,
                              SelfPlayConfig(noise_fraction=0, temperature=0),
                              random.Random(1), 1, 1)
        self.assertEqual(automatic.headers["Termination"], "FIVEFOLD_REPETITION")

    def test_search_scores_claimable_draw_as_zero_and_fifty_move_claim_is_available(self):
        board = chess.Board("7k/8/8/8/8/8/8/K5R1 w - - 99 80")
        for claim, expected in ((True, 0.0), (False, 0.5)):
            def predictor(position):
                return {move: 1.0 / position.legal_moves.count() for move in position.legal_moves}, 0.5
            root = search(board, predictor, 1, exploration=0, claim_draws=claim)
            child = next(child for child in root.children.values() if child.visits)
            self.assertAlmostEqual(child.mean_value, expected)

    def test_checkmate_takes_priority_over_a_claim_and_replay_accepts_claimed_game(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 99 80")
        self.assertEqual(terminal_outcome(board, True), None)
        board.push(chess.Move.from_uci("f7g7"))
        self.assertTrue(terminal_outcome(board, True).winner is chess.WHITE)
        positions = []
        game = play_game(self.shuffling_predictor,
                         SelfPlayConfig(noise_fraction=0, temperature=0, claim_draws=True),
                         random.Random(1), 1, 1, positions)
        replay = ReplayBuffer(100)
        with self.assertRaises(ValueError):
            replay.add_game(game, positions)
        replay.add_game(game, positions, claim_draws=True)
        self.assertEqual(len(replay.examples), 8)


class SelfPlayTests(unittest.TestCase):
    def test_long_game_reports_progress_without_changing_result(self):
        config = SelfPlayConfig(noise_fraction=0, temperature=0)
        output = io.StringIO()
        with patch("neural_chess.selfplay.time.monotonic", side_effect=[0, 5, 16, 20, 32]):
            with contextlib.redirect_stdout(output):
                game = play_game(fools_mate_predictor, config, random.Random(1), 1, 1)
        self.assertIn("2 plies (16 seconds elapsed)", output.getvalue())
        self.assertIn("4 plies (32 seconds elapsed)", output.getvalue())
        self.assertEqual(game.headers["Result"], "0-1")

    def test_scripted_complete_game_is_legal_and_has_correct_result(self):
        config = SelfPlayConfig(noise_fraction=0, temperature=0)
        game = play_game(fools_mate_predictor, config, random.Random(1), 1, 1)
        board = game.board()
        for move in game.mainline_moves():
            self.assertIn(move, board.legal_moves)
            board.push(move)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(game.headers["Result"], board.result())
        self.assertEqual(game.headers["PlyCount"], "4")

    def test_archives_games_with_configuration_and_alternating_budgets(self):
        config = SelfPlayConfig(white_simulations=1, black_simulations=2,
                                alternate_budgets=True, noise_fraction=0, temperature=0)
        self.assertEqual(budgets_for_game(config, 0), (1, 2))
        self.assertEqual(budgets_for_game(config, 1), (2, 1))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("neural_chess.selfplay.predict_legal_moves", side_effect=lambda network, board: fools_mate_predictor(board)):
                with contextlib.redirect_stdout(io.StringIO()):
                    paths = run_session(PolicyValueNetwork(), config, 2, output, 7)
            self.assertEqual(len(set(paths)), 2)
            sessions = list(output.glob("session-*.json"))
            self.assertEqual(len(sessions), 1)
            metadata = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["config"]["black_simulations"], 2)
            self.assertFalse(metadata["model_saved"])
            for index, path in enumerate(paths):
                with path.open(encoding="utf-8") as file:
                    game = chess.pgn.read_game(file)
                self.assertIsNotNone(game)
                self.assertEqual(game.errors, [])
                self.assertEqual(game.headers["SessionId"], metadata["session_id"])
                self.assertEqual(game.headers["ModelId"], metadata["model_id"])
                self.assertEqual((int(game.headers["WhiteSimulations"]), int(game.headers["BlackSimulations"])),
                                 budgets_for_game(config, index))
                board = game.board()
                for move in game.mainline_moves():
                    self.assertIn(move, board.legal_moves)
                    board.push(move)
                self.assertEqual(board.result(), game.headers["Result"])
                self.assertTrue(board.is_checkmate())

    def test_existing_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.pgn"
            write_new_file(path, "original")
            with self.assertRaises(FileExistsError):
                write_new_file(path, "replacement")
            self.assertEqual(path.read_text(encoding="utf-8"), "original")

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"white_simulations": 0}, {"temperature": -1},
                       {"noise_fraction": 2}, {"dirichlet_alpha": 0}):
            with self.assertRaises(ValueError):
                SelfPlayConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
