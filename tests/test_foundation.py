import unittest

import chess
import torch

from neural_chess.encoding import (
    ACTION_SIZE, HISTORY_LENGTH, POSITION_CHANNELS,
    decode_move, encode_move, encode_position, legal_action_mask,
)
from neural_chess.environment import ChessEnvironment
from neural_chess.network import PolicyValueNetwork, predict_legal_moves


class EnvironmentTests(unittest.TestCase):
    def test_illegal_moves_are_rejected_and_reset_works(self):
        env = ChessEnvironment()
        with self.assertRaises(ValueError):
            env.push(chess.Move.from_uci("e2e5"))
        env.push(chess.Move.from_uci("e2e4"))
        self.assertEqual(len(env.board.move_stack), 1)
        env.reset()
        self.assertEqual(env.board.fen(), chess.Board().fen())

    def test_checkmate_and_draws(self):
        env = ChessEnvironment()
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            env.push(chess.Move.from_uci(uci))
        self.assertEqual(env.outcome().winner, chess.BLACK)
        self.assertEqual(env.outcome().termination, chess.Termination.CHECKMATE)
        with self.assertRaises(ValueError):
            env.push(chess.Move.from_uci("e2e4"))
        env.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(env.outcome().termination, chess.Termination.STALEMATE)
        env.board = chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1")
        self.assertEqual(env.outcome().termination, chess.Termination.INSUFFICIENT_MATERIAL)

    def test_claimable_draw_is_explicit(self):
        env = ChessEnvironment()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            env.push(chess.Move.from_uci(uci))
        self.assertIsNone(env.outcome())
        self.assertEqual(env.outcome(claim_draw=True).termination, chess.Termination.THREEFOLD_REPETITION)
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            env.push(chess.Move.from_uci(uci))
        self.assertEqual(env.outcome().termination, chess.Termination.FIVEFOLD_REPETITION)

    def test_move_count_draws(self):
        env = ChessEnvironment()
        env.board = chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 100 51")
        self.assertIsNone(env.outcome())
        self.assertEqual(env.outcome(claim_draw=True).termination, chess.Termination.FIFTY_MOVES)
        env.board.halfmove_clock = 150
        self.assertEqual(env.outcome().termination, chess.Termination.SEVENTYFIVE_MOVES)


class EncodingTests(unittest.TestCase):
    def test_encoding_matches_original_for_full_histories_and_special_rules(self):
        def original_encoding(board):
            encoded = torch.zeros((POSITION_CHANNELS, 8, 8), dtype=torch.float32)
            history = board.copy(stack=True)
            for square, piece in board.piece_map().items():
                color_offset = 0 if piece.color == chess.WHITE else 6
                encoded[color_offset + piece.piece_type - 1, chess.square_rank(square),
                        chess.square_file(square)] = 1
            rights_order = ((chess.WHITE, True), (chess.WHITE, False),
                            (chess.BLACK, True), (chess.BLACK, False))
            for age in range(HISTORY_LENGTH):
                if not history.move_stack:
                    break
                history.pop()
                offset = 12 + age * 2
                for square, piece in history.piece_map().items():
                    sign = 1 if piece.color == chess.WHITE else -1
                    encoded[offset, chess.square_rank(square), chess.square_file(square)] = sign * piece.piece_type / 8
                rights = sum(1 << index for index, (color, kingside) in enumerate(rights_order)
                             if (history.has_kingside_castling_rights(color) if kingside
                                 else history.has_queenside_castling_rights(color)))
                encoded[offset + 1].fill_(1 + rights / 16)
                if history.has_legal_en_passant():
                    encoded[offset + 1, chess.square_rank(history.ep_square),
                            chess.square_file(history.ep_square)] += 2
            offset = 12 + HISTORY_LENGTH * 2
            encoded[offset].fill_(float(board.turn == chess.WHITE))
            for index, (color, kingside) in enumerate(rights_order):
                has_right = (board.has_kingside_castling_rights(color) if kingside
                             else board.has_queenside_castling_rights(color))
                encoded[offset + 1 + index].fill_(float(has_right))
            if board.ep_square is not None:
                encoded[offset + 5, chess.square_rank(board.ep_square),
                        chess.square_file(board.ep_square)] = 1
            encoded[offset + 6].fill_(min(board.halfmove_clock, 150) / 150)
            encoded[offset + 7].fill_(float(board.is_repetition(2)))
            encoded[offset + 8].fill_(float(board.is_repetition(3)))
            return encoded

        boards = [chess.Board(), chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 149 1")]
        board = chess.Board()
        for uci in ("e2e4", "a7a6", "e4e5", "d7d5", "g1f3"):
            board.push_uci(uci)
        boards.append(board)
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
                    "b5a4", "g8f6", "e1g1", "f8e7"):
            board.push_uci(uci)
        boards.append(board)
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 40:
            board.push_uci(uci)
        boards.append(board)
        for board in boards:
            with self.subTest(fen=board.fen(), plies=len(board.move_stack)):
                self.assertTrue(torch.equal(encode_position(board), original_encoding(board)))

    def test_legal_moves_round_trip_and_mask_special_moves(self):
        boards_and_moves = (
            (chess.Board(), ["e2e4", "g1f3"]),
            (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), ["e1g1", "e1c1"]),
            (chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"), ["e5d6"]),
            (chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
             ["a7a8n", "a7a8b", "a7a8r", "a7a8q"]),
        )
        for board, examples in boards_and_moves:
            mask = legal_action_mask(board)
            moves = list(board.legal_moves)
            actions = [encode_move(move) for move in moves]
            self.assertEqual(len(set(actions)), len(moves))
            self.assertEqual(mask.sum().item(), len(moves))
            for move in moves:
                self.assertEqual(decode_move(encode_move(move)), move)
                self.assertTrue(mask[encode_move(move)])
            for uci in examples:
                self.assertIn(chess.Move.from_uci(uci), moves)

    def test_all_move_encodings_are_unique_and_round_trip(self):
        for action in range(ACTION_SIZE):
            if action == 0:
                with self.assertRaises(ValueError):
                    decode_move(action)
                continue
            self.assertEqual(encode_move(decode_move(action)), action)
        for invalid in (-1, ACTION_SIZE):
            with self.assertRaises(ValueError):
                decode_move(invalid)
        with self.assertRaises(ValueError):
            encode_move(chess.Move.null())

    def test_history_and_rule_planes(self):
        board = chess.Board()
        position = encode_position(board)
        self.assertEqual(tuple(position.shape), (POSITION_CHANNELS, 8, 8))
        self.assertEqual(position[0, 1, 4].item(), 1)  # white pawn at e2
        self.assertEqual(position[13, 0, 0].item(), 0)
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(uci)
        position = encode_position(board)
        offset = 12 + HISTORY_LENGTH * 2
        self.assertEqual(position[offset + 7, 0, 0].item(), 1)
        self.assertEqual(position[offset + 8, 0, 0].item(), 1)
        self.assertEqual(position[12 + 7 * 2 + 1, 0, 0].item(), 1 + 15 / 16)
        self.assertEqual(position[12 + 8 * 2 + 1, 0, 0].item(), 0)
        self.assertAlmostEqual(position[offset + 6, 0, 0].item(), 8 / 150)
        board.push_uci("e2e4")
        position = encode_position(board)
        self.assertEqual(position[offset, 0, 0].item(), 0)
        self.assertEqual(position[offset + 5, 2, 4].item(), 1)
        self.assertEqual(position[offset + 6, 0, 0].item(), 0)

    def test_history_extends_beyond_recent_moves(self):
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
                    "b5a4", "g8f6", "e1g1", "f8e7"):
            board.push_uci(uci)
        position = encode_position(board)
        self.assertNotEqual(position[12 + 9 * 2 + 1, 0, 0].item(), 0)
        self.assertEqual(position[12 + 10 * 2 + 1, 0, 0].item(), 0)
        without_history = encode_position(chess.Board(board.fen()))
        self.assertEqual(without_history[13, 0, 0].item(), 0)
        self.assertNotEqual(position[13, 0, 0].item(), 0)

    def test_chess960_is_rejected(self):
        board = chess.Board(chess960=True)
        with self.assertRaises(ValueError):
            encode_position(board)
        with self.assertRaises(ValueError):
            legal_action_mask(board)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
        cls.network = PolicyValueNetwork()

    def test_shapes_range_and_gradient(self):
        inputs = torch.stack([encode_position(chess.Board())] * 2)
        policy, value = self.network(inputs)
        self.assertEqual(tuple(policy.shape), (2, ACTION_SIZE))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.all(value.abs() <= 1))
        (policy.mean() + value.mean()).backward()
        self.assertIsNotNone(self.network.stem[0].weight.grad)
        self.assertGreater(self.network.stem[0].weight.grad.abs().sum().item(), 0)

    def test_predictions_are_legal_and_normalized(self):
        board = chess.Board()
        probabilities, value = predict_legal_moves(self.network, board)
        self.assertEqual(set(probabilities), set(board.legal_moves))
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))
        self.assertTrue(-1 <= value <= 1)
        with torch.inference_mode():
            logits, expected_value = self.network(encode_position(board).unsqueeze(0))
            expected = torch.softmax(logits[0].masked_fill(~legal_action_mask(board), -torch.inf), dim=0)
        for move, probability in probabilities.items():
            self.assertEqual(probability, expected[encode_move(move)].item())
        self.assertEqual(value, expected_value.item())
        terminal = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        with self.assertRaises(ValueError):
            predict_legal_moves(self.network, terminal)

    def test_move_preferences_depend_on_network_weights(self):
        network = PolicyValueNetwork()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        before, _ = predict_legal_moves(network, board)
        with torch.no_grad():
            network.policy.bias[encode_move(move) // 64] += 20
        after, _ = predict_legal_moves(network, board)
        self.assertGreater(after[move], before[move])
        self.assertGreater(after[move], 0.99)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_prediction(self):
        network = PolicyValueNetwork().to("cuda")
        probabilities, _ = predict_legal_moves(network, chess.Board())
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=6)


if __name__ == "__main__":
    unittest.main()
