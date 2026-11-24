"""PUCT search using only network predictions and chess rules."""

from dataclasses import dataclass, field
import math
import random
from typing import Callable

import chess

Predictor = Callable[[chess.Board], tuple[dict[chess.Move, float], float]]


@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def terminal_outcome(board: chess.Board, claim_draws: bool = False) -> chess.Outcome | None:
    """Game result under the automatic FIDE rules, plus claimed draws if enabled.

    With claim_draws, a draw by threefold repetition or the 50-move rule is
    taken as soon as it becomes available. Checkmate and stalemate come first.
    """
    outcome = board.outcome()
    if outcome is None and claim_draws:
        if board.is_repetition(3):
            return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
        if board.is_fifty_moves():
            return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return outcome


def _terminal_value(board: chess.Board, outcome: chess.Outcome, value_scale: float) -> float:
    if outcome.winner is None:
        return 0.0
    return value_scale if outcome.winner == board.turn else -value_scale


def _expand(node: Node, board: chess.Board, predict: Predictor) -> float:
    priors, value = predict(board)
    legal = set(board.legal_moves)
    if set(priors) != legal:
        raise ValueError("Predictor must provide exactly the legal moves")
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Predictor value must be finite and between -1 and 1")
    if any(not math.isfinite(p) or p < 0 for p in priors.values()):
        raise ValueError("Predictor priors must be finite and non-negative")
    total = sum(priors.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Predictor priors must have positive finite sum")
    node.children = {move: Node(prior / total) for move, prior in priors.items()}
    return value

def search(
    board: chess.Board,
    predict: Predictor,
    simulations: int,
    *,
    exploration: float = 1.5,
    noise_fraction: float = 0.0,
    dirichlet_alpha: float = 0.3,
    value_scale: float = 1.0,
    rng: random.Random | None = None,
    claim_draws: bool = False,
) -> Node:
    """Run exactly `simulations` root-edge visits; node values are side-to-move."""
    if board.chess960 or terminal_outcome(board, claim_draws) is not None:
        raise ValueError("Search requires an unfinished standard chess position")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError("exploration must be finite and non-negative")
    if not math.isfinite(noise_fraction) or not 0 <= noise_fraction <= 1:
        raise ValueError("noise_fraction must be between 0 and 1")
    if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive and finite")
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be positive and finite")
    rng = rng if rng is not None else random.Random()
    root = Node(1.0)
    _expand(root, board, predict)
    if noise_fraction:
        noise = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in root.children]
        total = sum(noise)
        if total <= 0 or not math.isfinite(total):
            raise ValueError("Unable to sample Dirichlet noise")
        for child, sample in zip(root.children.values(), noise):
            child.prior = (1 - noise_fraction) * child.prior + noise_fraction * sample / total

    for _ in range(simulations):
        position = board.copy(stack=True)
        node = root
        path = [node]
        while node.children:
            scale = math.sqrt(max(1, node.visits))
            move, node = max(
                node.children.items(),
                key=lambda item: -item[1].mean_value
                + exploration * item[1].prior * scale / (1 + item[1].visits),
            )
            position.push(move)
            path.append(node)
        outcome = terminal_outcome(position, claim_draws)
        value = (_terminal_value(position, outcome, value_scale) if outcome is not None
                 else _expand(node, position, predict) * value_scale)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    return root


def visit_policy(root: Node, temperature: float) -> dict[chess.Move, float]:
    """Convert root visits to a sampling policy; zero temperature is deterministic."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not root.children or not any(child.visits for child in root.children.values()):
        raise ValueError("Search must have at least one root-edge visit")
    if temperature == 0:
        winner = max(root.children, key=lambda move: root.children[move].visits)
        return {move: float(move == winner) for move in root.children}
    largest = max(child.visits for child in root.children.values())
    weights = {
        move: math.exp(math.log(child.visits / largest) / temperature)
        for move, child in root.children.items() if child.visits
    }
    total = sum(weights.values())
    return {move: weights.get(move, 0.0) / total for move in root.children}
