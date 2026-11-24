import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import torch

from neural_chess.encoding import encode_move
from neural_chess.network import PolicyValueNetwork, predict_legal_moves
from neural_chess.selfplay import PositionVisits, SelfPlayConfig, play_game
from neural_chess.training import ReplayBuffer, TrainingConfig, run_diagnostic, train_step


def scripted_predictor(board):
    sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
    target = chess.Move.from_uci(sequence[len(board.move_stack)])
    return {move: float(move == target) for move in board.legal_moves}, 0.0


def scripted_game():
    positions = []
    game = play_game(scripted_predictor, SelfPlayConfig(noise_fraction=0, temperature=0),
                     random.Random(1), 1, 1, positions)
    return game, positions


class ReplayTests(unittest.TestCase):
    def test_self_play_value_signs_and_visit_targets(self):
        game, positions = scripted_game()
        board = positions[0].board
        positions[0].visits = {chess.Move.from_uci("f2f3"): 3,
                               chess.Move.from_uci("e2e4"): 1}
        replay = ReplayBuffer(10)
        replay.add_game(game, positions)
        self.assertEqual([example.value for example in replay.examples], [-1, 1, -1, 1])
        first = replay.examples[0]
        self.assertEqual(first.visit_counts[encode_move(chess.Move.from_uci("f2f3"))], 3)
        self.assertEqual(first.visit_counts[encode_move(chess.Move.from_uci("e2e4"))], 1)
        self.assertEqual(set(first.legal_actions), {encode_move(move) for move in board.legal_moves})

    def test_draws_and_bounded_buffer(self):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        positions = []
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 4:
            move = chess.Move.from_uci(uci)
            positions.append(PositionVisits(board.copy(stack=True), {move: 1}))
            node = node.add_variation(move)
            board.push(move)
        self.assertTrue(board.is_game_over())
        game.headers["Result"] = board.result()
        replay = ReplayBuffer(3)
        replay.add_game(game, positions)
        self.assertEqual(len(replay.examples), 3)
        self.assertTrue(all(example.value == 0 for example in replay.examples))
        self.assertEqual(len(replay.sample(10, random.Random(1))), 3)

    def test_bad_game_or_visits_do_not_modify_buffer(self):
        game, positions = scripted_game()
        replay = ReplayBuffer(10)
        positions[0].visits = {chess.Move.from_uci("e2e5"): 1}
        with self.assertRaises(ValueError):
            replay.add_game(game, positions)
        self.assertEqual(len(replay.examples), 0)
        game, positions = scripted_game()
        game.headers["Result"] = "1-0"
        with self.assertRaises(ValueError):
            replay.add_game(game, positions)
        self.assertEqual(len(replay.examples), 0)


class TrainingTests(unittest.TestCase):
    def test_synthetic_optimizer_improves_target_predictions(self):
        torch.manual_seed(17)
        game, positions = scripted_game()
        replay = ReplayBuffer(10)
        replay.add_game(game, positions)
        example = replay.examples[0]
        network = PolicyValueNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
        move = chess.Move.from_uci("f2f3")
        before, value_before = predict_legal_moves(network, positions[0].board)
        original = network.stem[0].weight.detach().clone()
        for _ in range(12):
            train_step(network, optimizer, [example])
        after, value_after = predict_legal_moves(network, positions[0].board)
        self.assertFalse(torch.equal(original, network.stem[0].weight))
        self.assertGreater(after[move], before[move])
        self.assertLess(abs(value_after - example.value), abs(value_before - example.value))
        self.assertTrue(optimizer.state)

    def test_reward_scale_changes_value_loss_without_changing_policy_loss(self):
        game, positions = scripted_game()
        replay = ReplayBuffer(10)
        replay.add_game(game, positions)
        first = PolicyValueNetwork()
        second = PolicyValueNetwork()
        second.load_state_dict(first.state_dict())
        first_optimizer = torch.optim.Adam(first.parameters(), lr=0.001)
        second_optimizer = torch.optim.Adam(second.parameters(), lr=0.001)
        policy1, value1 = train_step(first, first_optimizer, list(replay.examples), value_scale=1)
        policy2, value2 = train_step(second, second_optimizer, list(replay.examples), value_scale=2)
        self.assertAlmostEqual(policy1, policy2, places=6)
        self.assertAlmostEqual(value2, 4 * value1, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_updates_weights(self):
        game, positions = scripted_game()
        replay = ReplayBuffer(10)
        replay.add_game(game, positions)
        network = PolicyValueNetwork().to("cuda")
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        before = network.stem[0].weight.detach().clone()
        losses = train_step(network, optimizer, replay.sample(4, random.Random(1)))
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
        self.assertFalse(torch.equal(before, network.stem[0].weight))

    def test_diagnostic_archives_weights_optimizer_replay_and_game(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = SelfPlayConfig(noise_fraction=0, temperature=0)
            training = TrainingConfig(games=1, updates=2, batch_size=4, replay_capacity=10)
            with patch("neural_chess.selfplay.predict_legal_moves",
                       side_effect=lambda network, board: scripted_predictor(board)):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_dir = run_diagnostic(PolicyValueNetwork(), config, training, 7,
                                             base / "runs", base / "games")
            initial = torch.load(run_dir / "initial.pt", map_location="cpu", weights_only=True)
            final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=True)
            metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["resumable"])
            self.assertEqual(final["updates"], 2)
            self.assertEqual(len(final["replay"]), 4)
            self.assertEqual(final["positions_generated"], 4)
            self.assertEqual(len(final["metrics"]), 2)
            self.assertTrue(final["optimizer"]["state"])
            self.assertFalse(torch.equal(initial["model"]["stem.0.weight"],
                                         final["model"]["stem.0.weight"]))
            self.assertEqual(final["metadata"]["run_id"], metadata["run_id"])
            self.assertEqual(len(final["game_ids"]), 1)
            path = base / "games" / f"game-{final['game_ids'][0]}.pgn"
            with path.open(encoding="utf-8") as file:
                game = chess.pgn.read_game(file)
            self.assertEqual(game.headers["SessionId"], metadata["run_id"])
            self.assertEqual(game.headers["ModelId"], metadata["initial_model_id"])

    def test_invalid_training_configuration(self):
        for kwargs in ({"games": 0}, {"batch_size": 0}, {"updates": -1},
                       {"learning_rate": float("nan")}, {"replay_capacity": 0}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
