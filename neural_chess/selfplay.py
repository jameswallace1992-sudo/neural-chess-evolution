"""Play and archive complete games using a single, untrained network."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable
import uuid

import chess
import chess.pgn
import torch

from .mcts import Predictor, search, terminal_outcome, visit_policy
from .network import PolicyValueNetwork, predict_legal_moves
from .opponent import open_opponent, validate_level


@dataclass(frozen=True)
class SelfPlayConfig:
    white_simulations: int = 1
    black_simulations: int = 1
    alternate_budgets: bool = False
    exploration: float = 1.5
    noise_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    temperature: float = 1.0
    temperature_plies: int = 30
    value_scale: float = 1.0
    claim_draws: bool = False
    opponent_level: int | None = None
    opponent_path: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.claim_draws, bool):
            raise ValueError("claim_draws must be true or false")
        if self.opponent_level is not None:
            validate_level(self.opponent_level)
            if not isinstance(self.opponent_path, str) or not self.opponent_path:
                raise ValueError("opponent_path is required when opponent_level is set")
        for name in ("white_simulations", "black_simulations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and non-negative")
        if not math.isfinite(self.noise_fraction) or not 0 <= self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be between 0 and 1")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive and finite")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(self.temperature_plies, bool) or not isinstance(self.temperature_plies, int) or self.temperature_plies < 0:
            raise ValueError("temperature_plies must be a non-negative integer")
        if not math.isfinite(self.value_scale) or self.value_scale <= 0:
            raise ValueError("value_scale must be positive and finite")


@dataclass
class PositionVisits:
    board: chess.Board
    visits: dict[chess.Move, int]


def budgets_for_game(config: SelfPlayConfig, game_index: int) -> tuple[int, int]:
    if config.alternate_budgets and game_index % 2:
        return config.black_simulations, config.white_simulations
    return config.white_simulations, config.black_simulations


def network_colour_for(config: SelfPlayConfig, game_index: int) -> bool | None:
    """The network plays white in even games against an opponent; None means self-play."""
    if config.opponent_level is None:
        return None
    return chess.WHITE if game_index % 2 == 0 else chess.BLACK


def play_game(
    predict: Predictor,
    config: SelfPlayConfig,
    rng: random.Random,
    white_simulations: int,
    black_simulations: int,
    positions: list[PositionVisits] | None = None,
    game_index: int = 0,
    opponent_move: Callable[[chess.Board, random.Random], chess.Move] | None = None,
) -> chess.pgn.Game:
    """Play one game. With an opponent level, the opponent chooses the other side's moves
    and those plies are recorded with empty visit counts (they are never learning targets)."""
    colour = network_colour_for(config, game_index)
    if colour is not None and opponent_move is None:
        with open_opponent(config.opponent_level, config.opponent_path) as choose:
            return play_game(predict, config, rng, white_simulations, black_simulations,
                             positions, game_index, choose)
    board = chess.Board()
    game = chess.pgn.Game()
    if colour is not None:
        game.headers["OpponentLevel"] = str(config.opponent_level)
        game.headers["NetworkColour"] = "white" if colour == chess.WHITE else "black"
    node = game
    started = time.monotonic()
    last_progress = started
    while terminal_outcome(board, config.claim_draws) is None:
        if colour is not None and board.turn != colour:
            move = opponent_move(board, rng)
            if positions is not None:
                positions.append(PositionVisits(board.copy(stack=True), {}))
            node = node.add_variation(move)
            board.push(move)
            continue
        simulations = white_simulations if board.turn == chess.WHITE else black_simulations
        root = search(
            board, predict, simulations,
            exploration=config.exploration,
            noise_fraction=config.noise_fraction,
            dirichlet_alpha=config.dirichlet_alpha,
            value_scale=config.value_scale,
            rng=rng,
            claim_draws=config.claim_draws,
        )
        if positions is not None:
            positions.append(PositionVisits(
                board.copy(stack=True),
                {move: child.visits for move, child in root.children.items() if child.visits},
            ))
        temperature = config.temperature if len(board.move_stack) < config.temperature_plies else 0
        policy = visit_policy(root, temperature)
        if temperature == 0:
            move = max(policy, key=policy.get)
        else:
            move = rng.choices(list(policy), weights=list(policy.values()), k=1)[0]
        node = node.add_variation(move)
        board.push(move)
        now = time.monotonic()
        if now - last_progress >= 15:
            print(f"Game in progress: {len(board.move_stack)} plies "
                  f"({int(now - started)} seconds elapsed).", flush=True)
            last_progress = now
    outcome = terminal_outcome(board, config.claim_draws)
    assert outcome is not None
    game.headers["Result"] = outcome.result()
    game.headers["Termination"] = outcome.termination.name
    game.headers["PlyCount"] = str(len(board.move_stack))
    return game


def write_new_file(path: Path, contents: str | bytes) -> None:
    """Publish a complete file without replacing an existing archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(contents.encode("utf-8") if isinstance(contents, str) else contents)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def network_identifier(network: PolicyValueNetwork) -> str:
    buffer = io.BytesIO()
    torch.save(network.state_dict(), buffer)
    return "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()


def label_game(
    game: chess.pgn.Game,
    game_id: str,
    session_id: str,
    model_id: str,
    generation: int,
    game_index: int,
    white: int,
    black: int,
) -> None:
    finished = datetime.now(timezone.utc)
    network_name = f"Network generation {generation}"
    white_name = black_name = network_name
    event = "Neural Chess Evolution self-play"
    if "OpponentLevel" in game.headers:
        opponent = f"Stockfish opponent level {game.headers['OpponentLevel']}"
        network_white = game.headers["NetworkColour"] == "white"
        white_name, black_name = (network_name, opponent) if network_white else (opponent, network_name)
        event = "Neural Chess Evolution Stockfish-opponent experiment"
    game.headers.update({
        "Event": event,
        "Site": "Local",
        "Date": finished.strftime("%Y.%m.%d"),
        "Round": str(game_index + 1),
        "White": white_name,
        "Black": black_name,
        "GameId": game_id,
        "SessionId": session_id,
        "FinishedUTC": finished.isoformat(),
        "Generation": str(generation),
        "ModelId": model_id,
        "WhiteSimulations": str(white),
        "BlackSimulations": str(black),
        "ExperimentId": session_id,
    })


def run_session(
    network: PolicyValueNetwork,
    config: SelfPlayConfig,
    games: int,
    output_dir: Path,
    seed: int,
    *,
    session_id: str | None = None,
    model_saved: bool = False,
    on_game: Callable[[chess.pgn.Game, list[PositionVisits]], None] | None = None,
) -> list[Path]:
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ValueError("games must be a positive integer")
    session_id = session_id if session_id is not None else uuid.uuid4().hex
    model_id = network_identifier(network)
    started = datetime.now(timezone.utc)
    write_new_file(
        output_dir / f"session-{session_id}.json",
        json.dumps({
            "session_id": session_id,
            "started_utc": started.isoformat(),
            "generation": 0,
            "model_id": model_id,
            "model_saved": model_saved,
            "seed": seed,
            "games_requested": games,
            "config": asdict(config),
        }, indent=2) + "\n",
    )
    rng = random.Random(seed)
    saved = []
    for game_index in range(games):
        white, black = budgets_for_game(config, game_index)
        positions: list[PositionVisits] | None = [] if on_game is not None else None
        game = play_game(
            lambda board: predict_legal_moves(network, board), config, rng, white, black, positions
        )
        game_id = uuid.uuid4().hex
        label_game(game, game_id, session_id, model_id, 0, game_index, white, black)
        path = output_dir / f"game-{game_id}.pgn"
        write_new_file(path, str(game) + "\n\n")
        saved.append(path)
        print(f"Saved {path} ({game.headers['Result']}, {game.headers['PlyCount']} plies)", flush=True)
        if on_game is not None:
            assert positions is not None
            on_game(game, positions)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate complete untrained self-play games.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--white-simulations", type=int, default=1)
    parser.add_argument("--black-simulations", type=int, default=1)
    parser.add_argument("--alternate-budgets", action="store_true")
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("games") / "selfplay")
    args = parser.parse_args()
    try:
        config = SelfPlayConfig(
            white_simulations=args.white_simulations, black_simulations=args.black_simulations,
            alternate_budgets=args.alternate_budgets, exploration=args.exploration,
            noise_fraction=args.noise_fraction, dirichlet_alpha=args.dirichlet_alpha,
            temperature=args.temperature, temperature_plies=args.temperature_plies,
        )
        if args.games < 1:
            raise ValueError("games must be a positive integer")
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be between 0 and 2**63 - 1")
    except ValueError as exc:
        parser.error(str(exc))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; seed {seed}; playing {args.games} game(s).", flush=True)
    run_session(PolicyValueNetwork().to(device), config, args.games, args.output_dir, seed)


if __name__ == "__main__":
    main()
