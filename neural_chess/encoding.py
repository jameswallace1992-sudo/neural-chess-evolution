"""Fixed encodings for standard chess positions and moves."""

import chess
import torch

HISTORY_LENGTH = 150
POSITION_CHANNELS = 12 + HISTORY_LENGTH * 2 + 9
PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
ACTION_SIZE = 64 * len(PROMOTIONS) * 64


def _require_standard_chess(board: chess.Board) -> None:
    if board.chess960:
        raise ValueError("Chess960 is not supported")


def encode_position(board: chess.Board) -> torch.Tensor:
    """Encode the current board and up to 150 earlier boards as [C, 8, 8]."""
    _require_standard_chess(board)
    encoded = [0.0] * (POSITION_CHANNELS * 64)
    history = board.copy(stack=True)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        encoded[(color_offset + piece.piece_type - 1) * 64 + square] = 1

    for age in range(HISTORY_LENGTH):
        if not history.move_stack:
            break
        history.pop()
        offset = 12 + age * 2
        for square, piece in history.piece_map().items():
            sign = 1 if piece.color == chess.WHITE else -1
            encoded[offset * 64 + square] = sign * piece.piece_type / 8
        rights = sum(
            (1 << index) for index, (color, kingside) in enumerate(
                ((chess.WHITE, True), (chess.WHITE, False),
                 (chess.BLACK, True), (chess.BLACK, False))
            )
            if (history.has_kingside_castling_rights(color) if kingside
                else history.has_queenside_castling_rights(color))
        )
        encoded[(offset + 1) * 64:(offset + 2) * 64] = [1 + rights / 16] * 64
        if history.has_legal_en_passant():
            square = history.ep_square
            assert square is not None
            encoded[(offset + 1) * 64 + square] += 2

    offset = 12 + HISTORY_LENGTH * 2
    encoded[offset * 64:(offset + 1) * 64] = [float(board.turn == chess.WHITE)] * 64
    for index, (color, kingside) in enumerate(
        ((chess.WHITE, True), (chess.WHITE, False),
         (chess.BLACK, True), (chess.BLACK, False))
    ):
        has_right = (
            board.has_kingside_castling_rights(color)
            if kingside else board.has_queenside_castling_rights(color)
        )
        start = (offset + 1 + index) * 64
        encoded[start:start + 64] = [float(has_right)] * 64
    if board.ep_square is not None:
        encoded[(offset + 5) * 64 + board.ep_square] = 1
    for channel, value in (
        (offset + 6, min(board.halfmove_clock, 150) / 150),
        (offset + 7, float(board.is_repetition(2))),
        (offset + 8, float(board.is_repetition(3))),
    ):
        encoded[channel * 64:(channel + 1) * 64] = [value] * 64
    return torch.tensor(encoded, dtype=torch.float32).view(POSITION_CHANNELS, 8, 8)


def encode_move(move: chess.Move) -> int:
    """Map a standard move to (destination, promotion, origin)."""
    if move.drop is not None or not move:
        raise ValueError("Drops and null moves are not supported")
    try:
        promotion = PROMOTIONS.index(move.promotion)
    except ValueError as exc:
        raise ValueError(f"Unsupported promotion: {move.promotion}") from exc
    return (move.to_square * len(PROMOTIONS) + promotion) * 64 + move.from_square


def decode_move(action: int) -> chess.Move:
    """Decode an action; legality must be checked against the current board."""
    if not isinstance(action, int) or not 0 <= action < ACTION_SIZE:
        raise ValueError(f"Action must be an integer in [0, {ACTION_SIZE})")
    destination_promotion, origin = divmod(action, 64)
    destination, promotion = divmod(destination_promotion, len(PROMOTIONS))
    move = chess.Move(origin, destination, promotion=PROMOTIONS[promotion])
    if not move:
        raise ValueError("Action represents an unsupported null move")
    return move


def legal_action_mask(board: chess.Board) -> torch.Tensor:
    _require_standard_chess(board)
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[encode_move(move)] = True
    return mask
