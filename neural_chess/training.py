"""A single, non-resumable diagnostic self-play and learning cycle."""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import random
import uuid

import chess
import chess.pgn
import torch
from torch import nn
from torch.nn import functional as F

from .encoding import ACTION_SIZE, encode_move, encode_position
from .mcts import terminal_outcome
from .network import PolicyValueNetwork
from .selfplay import PositionVisits, SelfPlayConfig, network_identifier, run_session, write_new_file


@dataclass(frozen=True)
class TrainingConfig:
    games: int = 1
    updates: int = 4
    batch_size: int = 32
    replay_capacity: int = 512
    learning_rate: float = 0.001
    learning_mode: str = "sampled"

    def __post_init__(self) -> None:
        for name in ("games", "updates", "batch_size", "replay_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.learning_mode not in ("sampled", "fresh_pass"):
            raise ValueError("learning_mode must be sampled or fresh_pass")


@dataclass
class ReplayExample:
    position: torch.Tensor
    legal_actions: list[int]
    visit_counts: dict[int, int]
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.examples: deque[ReplayExample] = deque(maxlen=capacity)

    def add_game(self, game: chess.pgn.Game, positions: list[PositionVisits],
                 claim_draws: bool = False) -> None:
        result = game.headers["Result"]
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("Training requires a completed game")
        moves = list(game.mainline_moves())
        if len(positions) != len(moves):
            raise ValueError("Position count does not match the game")
        board = game.board()
        network_colour = {"white": chess.WHITE, "black": chess.BLACK}.get(game.headers.get("NetworkColour"))
        examples = []
        for position, move in zip(positions, moves):
            if position.board.fen() != board.fen() or position.board.move_stack != board.move_stack:
                raise ValueError("Position history does not match the game")
            legal = set(board.legal_moves)
            if network_colour is not None and board.turn != network_colour:
                # Opponent plies are never learning targets.
                if position.visits or move not in legal:
                    raise ValueError("Opponent plies must be legal and carry no visit counts")
                board.push(move)
                continue
            if move not in legal or not position.visits or any(
                action not in legal or isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for action, count in position.visits.items()
            ):
                raise ValueError("Visits and played moves must be legal with positive counts")
            if result == "1/2-1/2":
                value = 0.0
            else:
                white_won = result == "1-0"
                value = 1.0 if board.turn == white_won else -1.0
            examples.append(ReplayExample(
                position=encode_position(board),
                legal_actions=[encode_move(action) for action in legal],
                visit_counts={encode_move(action): count for action, count in position.visits.items()},
                value=value,
            ))
            board.push(move)
        outcome = terminal_outcome(board, claim_draws)
        if outcome is None or outcome.result() != result:
            raise ValueError("Recorded game result does not match its actual outcome")
        self.examples.extend(examples)

    def sample(self, size: int, rng: random.Random) -> list[ReplayExample]:
        if size < 1 or not self.examples:
            raise ValueError("Cannot sample an empty buffer or a non-positive batch size")
        return rng.sample(list(self.examples), min(size, len(self.examples)))


def train_step(
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    examples: list[ReplayExample],
    *,
    value_scale: float = 1.0,
) -> tuple[float, float]:
    if not examples:
        raise ValueError("Training batch must not be empty")
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be positive and finite")
    device = next(network.parameters()).device
    positions = torch.stack([example.position for example in examples]).to(device)
    legal = torch.zeros((len(examples), ACTION_SIZE), dtype=torch.bool, device=device)
    targets = torch.zeros((len(examples), ACTION_SIZE), device=device)
    for row, example in enumerate(examples):
        legal[row, example.legal_actions] = True
        total = sum(example.visit_counts.values())
        if total <= 0 or any(action not in example.legal_actions or count <= 0
                             for action, count in example.visit_counts.items()):
            raise ValueError("Invalid policy target")
        for action, count in example.visit_counts.items():
            targets[row, action] = count / total
    values = torch.tensor([example.value for example in examples], device=device)
    network.train()
    optimizer.zero_grad(set_to_none=True)
    logits, predictions = network(positions)
    log_probs = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
    log_probs = log_probs.masked_fill(~legal, 0)
    policy_loss = -(targets * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(predictions.squeeze(1) * value_scale, values * value_scale)
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss; update aborted")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
        raise ValueError("Non-finite gradients; update aborted")
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _save_torch(path: Path, state: dict) -> None:
    buffer = io.BytesIO()
    torch.save(_cpu_state(state), buffer)
    write_new_file(path, buffer.getvalue())


def run_diagnostic(
    network: PolicyValueNetwork,
    selfplay: SelfPlayConfig,
    training: TrainingConfig,
    seed: int,
    runs_dir: Path = Path("training_runs"),
    games_dir: Path = Path("games") / "selfplay",
) -> Path:
    """Run one fresh model only; this deliberately does not support resuming."""
    run_id = uuid.uuid4().hex
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.Adam(network.parameters(), lr=training.learning_rate)
    replay = ReplayBuffer(training.replay_capacity)
    metadata = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "initial_model_id": network_identifier(network),
        "selfplay": asdict(selfplay),
        "training": asdict(training),
        "device": str(next(network.parameters()).device),
        "torch_version": str(torch.__version__),
        "resumable": False,
    }
    write_new_file(run_dir / "config.json", json.dumps(metadata, indent=2) + "\n")
    _save_torch(run_dir / "initial.pt", {"model": network.state_dict(), "metadata": metadata})
    positions_generated = 0

    def record_game(game: chess.pgn.Game, positions: list[PositionVisits]) -> None:
        nonlocal positions_generated
        replay.add_game(game, positions, selfplay.claim_draws)
        positions_generated += len(positions)

    paths = run_session(
        network, selfplay, training.games, games_dir, seed,
        session_id=run_id, model_saved=True, on_game=record_game,
    )
    rng = random.Random(seed)
    metrics = []
    for update in range(training.updates):
        policy_loss, value_loss = train_step(
            network, optimizer, replay.sample(training.batch_size, rng), value_scale=selfplay.value_scale
        )
        metrics.append({"update": update + 1, "policy_loss": policy_loss, "value_loss": value_loss})
        print(f"Update {update + 1}: policy loss {policy_loss:.4f}, value loss {value_loss:.4f}", flush=True)
    final = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay": [
            {"position": example.position, "legal_actions": example.legal_actions,
             "visit_counts": example.visit_counts, "value": example.value}
            for example in replay.examples
        ],
        "metadata": metadata,
        "final_model_id": network_identifier(network),
        "game_ids": [path.stem.removeprefix("game-") for path in paths],
        "metrics": metrics,
        "updates": training.updates,
        "positions_generated": positions_generated,
        "torch_rng_state": torch.get_rng_state(),
        "sampling_rng_state": rng.getstate(),
    }
    _save_torch(run_dir / "final.pt", final)
    print(f"Saved diagnostic results to {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fresh, non-resumable diagnostic training cycle.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--white-simulations", type=int, default=1)
    parser.add_argument("--black-simulations", type=int, default=1)
    parser.add_argument("--alternate-budgets", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        training = TrainingConfig(args.games, args.updates, args.batch_size,
                                  args.replay_capacity, args.learning_rate)
        selfplay = SelfPlayConfig(white_simulations=args.white_simulations,
                                  black_simulations=args.black_simulations,
                                  alternate_budgets=args.alternate_budgets)
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be between 0 and 2**63 - 1")
    except ValueError as exc:
        parser.error(str(exc))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fresh diagnostic model on {device}; seed {seed}. This run cannot be resumed.", flush=True)
    run_diagnostic(PolicyValueNetwork().to(device), selfplay, training, seed)


if __name__ == "__main__":
    main()
