"""Chess rules are supplied by python-chess, not by an evaluation engine."""

import chess


class ChessEnvironment:
    def __init__(self) -> None:
        self.board = chess.Board()

    def reset(self) -> None:
        self.board.reset()

    def push(self, move: chess.Move) -> None:
        if self.board.is_game_over():
            raise ValueError("Cannot play a move after the game is over")
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")
        self.board.push(move)

    def outcome(self, *, claim_draw: bool = False) -> chess.Outcome | None:
        return self.board.outcome(claim_draw=claim_draw)
