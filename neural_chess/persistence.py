"""Durable, single-process training history with safe-step checkpoints."""

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import pickle
from pathlib import Path
import random
import re
import signal
import sys
import time
import uuid

import chess
import chess.pgn
import torch

from .network import PolicyValueNetwork, predict_legal_moves
from .opponent import network_score, next_level
from .parallel import GamePool, game_seed
from .selfplay import (
    PositionVisits, SelfPlayConfig, budgets_for_game, label_game, play_game, write_new_file,
)
from .training import ReplayBuffer, ReplayExample, TrainingConfig, _cpu_state, train_step

FORMAT_VERSION = 1
REFERENCED_FORMAT_VERSION = 2
ARCHITECTURE = "policy-value-321-64x4-20480-v1"
DEFAULT_STATE = Path("training_state") / "main"
DEFAULT_GAMES = Path("games") / "selfplay"


def _serialize(value: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(_cpu_state(value), buffer)
    return buffer.getvalue()


def _pack(value: dict) -> bytes:
    payload = _serialize(value)
    return _serialize({"payload": payload, "sha256": hashlib.sha256(payload).hexdigest(),
                       "envelope_version": 1})


def _read(path: Path) -> dict:
    result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    if "envelope_version" in result:
        if result["envelope_version"] != 1 or not isinstance(result.get("payload"), bytes):
            raise ValueError(f"Invalid integrity envelope in {path}")
        payload = result["payload"]
        if hashlib.sha256(payload).hexdigest() != result.get("sha256"):
            raise ValueError(f"Integrity check failed for {path}")
        result = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a state dictionary in {path}")
    return result


def _model_id_for_state(model: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.items()):
        tensor = value.detach().to("cpu").contiguous().clone()
        description = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        digest.update(description.encode("utf-8") + b"\0")
        digest.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return "sha256-tensor-v1:" + digest.hexdigest()


def model_id(network: PolicyValueNetwork) -> str:
    """Hash named tensor contents independent of device and serialization."""
    return _model_id_for_state(network.state_dict())


def _replay_data(replay: ReplayBuffer) -> list[dict]:
    return [
        {"position": example.position, "legal_actions": example.legal_actions,
         "visit_counts": example.visit_counts, "value": example.value}
        for example in replay.examples
    ]


def _load_replay(data: list[dict], capacity: int) -> ReplayBuffer:
    if len(data) > capacity:
        raise ValueError("Saved replay buffer exceeds its recorded capacity")
    replay = ReplayBuffer(capacity)
    replay.examples.extend(ReplayExample(**example) for example in data)
    return replay


def _read_replay_source(state_dir: Path, source: str) -> list[dict]:
    if not isinstance(source, str) or re.fullmatch(r"(origin|game)-[0-9a-f]{32}", source) is None:
        raise ValueError("Invalid replay source identifier")
    saved = _read(state_dir / "replay_sources" / f"{source}.pt")
    if saved.get("source_id") != source or not isinstance(saved.get("examples"), list):
        raise ValueError(f"Invalid replay source: {source}")
    return saved["examples"]


def _load_replay_refs(state_dir: Path, state: dict) -> ReplayBuffer:
    refs = state["replay_refs"]
    if not isinstance(refs, list) or len(refs) > state["replay_capacity"]:
        raise ValueError("Invalid replay references")
    sources: dict[str, list[dict]] = {}
    data = []
    for ref in refs:
        if not isinstance(ref, (list, tuple)) or len(ref) != 2:
            raise ValueError("Invalid replay reference")
        source, index = ref
        if not isinstance(source, str):
            raise ValueError("Invalid replay source identifier")
        if source not in sources:
            sources[source] = _read_replay_source(state_dir, source)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(sources[source]):
            raise ValueError("Invalid replay source index")
        data.append(sources[source][index])
    return _load_replay(data, state["replay_capacity"])


def _migrate_replay(state_dir: Path, state: dict, replay: ReplayBuffer) -> None:
    if state["format_version"] != FORMAT_VERSION:
        return
    source = "origin-" + uuid.uuid4().hex
    write_new_file(state_dir / "replay_sources" / f"{source}.pt",
                   _pack({"source_id": source, "examples": _replay_data(replay)}))
    state["replay_refs"] = [[source, index] for index in range(len(replay.examples))]
    del state["replay"]
    state["format_version"] = REFERENCED_FORMAT_VERSION
    _new_checkpoint(state_dir, state)


def _checkpoint_files(state_dir: Path) -> list[Path]:
    directory = state_dir / "checkpoints"
    if not directory.exists():
        return []
    return sorted(directory.glob("checkpoint-[0-9]" + "[0-9]" * 11 + ".pt"))


@contextmanager
def _training_lock(state_dir: Path):
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Training history does not exist: {state_dir}")
    with (state_dir / "training.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is using this training history") from exc
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another process is using this training history") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_checkpoint(state: dict) -> None:
    if state.get("format_version") not in (FORMAT_VERSION, REFERENCED_FORMAT_VERSION) or state.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported checkpoint format or network architecture")
    for key in ("sequence", "generation", "model", "optimizer", "replay_capacity",
                "torch_rng_state", "sampling_rng_state", "selfplay_rng_state", "cycle",
                "total_updates", "run_id", "model_id"):
        if key not in state:
            raise ValueError(f"Checkpoint is missing {key}")
    replay_key = "replay" if state["format_version"] == FORMAT_VERSION else "replay_refs"
    if replay_key not in state:
        raise ValueError(f"Checkpoint is missing {replay_key}")
    if _model_id_for_state(state["model"]) != state["model_id"]:
        raise ValueError("Checkpoint model identifier does not match its weights")


def load_latest(state_dir: Path, *, recover_previous: bool = False,
                validate_replay: bool = False) -> tuple[dict, Path]:
    files = _checkpoint_files(state_dir)
    if not files:
        raise FileNotFoundError(f"No training checkpoint found in {state_dir}; no model was initialized")
    failures = []
    for path in reversed(files):
        try:
            state = _read(path)
            _validate_checkpoint(state)
            if state["sequence"] != int(path.stem.split("-")[-1]):
                raise ValueError("Checkpoint sequence does not match its filename")
            if validate_replay and state["format_version"] == REFERENCED_FORMAT_VERSION:
                _load_replay_refs(state_dir, state)
            if failures:
                print(f"WARNING: recovering {path}; newer checkpoints failed: {', '.join(failures)}",
                      file=sys.stderr, flush=True)
            return state, path
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError, ValueError, KeyError) as exc:
            failures.append(f"{path.name}: {exc}")
            if not recover_previous:
                raise ValueError(
                    f"Latest checkpoint {path} is unreadable or incompatible; "
                    "it was not replaced. Use --recover-previous only after reviewing the risk."
                ) from exc
    raise ValueError(f"No usable checkpoint found: {'; '.join(failures)}")


def _new_checkpoint(state_dir: Path, state: dict) -> Path:
    files = _checkpoint_files(state_dir)
    last_file_sequence = int(files[-1].stem.split("-")[-1]) if files else 0
    state["sequence"] = max(state["sequence"], last_file_sequence) + 1
    path = state_dir / "checkpoints" / f"checkpoint-{state['sequence']:012d}.pt"
    write_new_file(path, _pack(state))
    return path


def _generation_snapshot(state_dir: Path, state: dict, network: PolicyValueNetwork) -> None:
    path = state_dir / "generations" / f"generation-{state['generation']:08d}.pt"
    if path.exists():
        saved = _read(path)
        _validate_checkpoint(saved)
        if saved.get("model_id") != model_id(network) or saved["generation"] != state["generation"]:
            raise ValueError(f"Existing generation snapshot conflicts with current state: {path}")
    else:
        write_new_file(path, _pack(state))


def _sync_state(
    state: dict,
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    sample_rng: random.Random,
    play_rng: random.Random,
) -> None:
    state["model"] = _cpu_state(network.state_dict())
    state["model_id"] = model_id(network)
    state["optimizer"] = _cpu_state(optimizer.state_dict())
    if state["format_version"] == FORMAT_VERSION:
        state["replay"] = _replay_data(replay)
    elif len(state["replay_refs"]) != len(replay.examples):
        raise ValueError("Replay references do not match the in-memory buffer")
    state["torch_rng_state"] = torch.get_rng_state()
    state["sampling_rng_state"] = sample_rng.getstate()
    state["selfplay_rng_state"] = play_rng.getstate()
    device = next(network.parameters()).device
    state["cuda_rng_states"] = torch.cuda.get_rng_state_all() if device.type == "cuda" else None


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def initialize_fresh(state_dir: Path, seed: int | None = None, *,
                     simulations: int = 1, value_scale: float = 1.0) -> None:
    """Create a separate history with random weights and no previous experience."""
    if state_dir.exists():
        raise FileExistsError(f"Training history already exists at {state_dir}; it will not be replaced")
    if seed is None:
        seed = random.SystemRandom().randrange(2**63)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer between 0 and 2**63 - 1")
    selfplay = SelfPlayConfig(white_simulations=simulations, black_simulations=simulations,
                              value_scale=value_scale)
    training = TrainingConfig(games=1, updates=1, batch_size=16, replay_capacity=512)
    torch.manual_seed(seed)
    network = PolicyValueNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=training.learning_rate)
    replay = ReplayBuffer(training.replay_capacity)
    sample_rng = random.Random(seed)
    play_rng = random.Random(seed)
    state = {
        "format_version": REFERENCED_FORMAT_VERSION, "architecture": ARCHITECTURE,
        "run_id": uuid.uuid4().hex, "sequence": 0, "generation": 0,
        "total_games": 0, "total_updates": 0, "positions_since_import": 0,
        "replay_capacity": training.replay_capacity, "replay_refs": [],
        "last_selfplay_config": asdict(selfplay), "last_training_config": asdict(training),
        "last_checkpoint_interval": 1, "handled_stop_requests": [], "cycle": None,
    }
    state_dir.mkdir(parents=True, exist_ok=False)
    with _training_lock(state_dir):
        write_new_file(state_dir / "initialization.json", json.dumps({
            "run_id": state["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed, "architecture": ARCHITECTURE,
            "selfplay": asdict(selfplay), "training": asdict(training),
        }, indent=2) + "\n")
        _sync_state(state, network, optimizer, replay, sample_rng, play_rng)
        path = _new_checkpoint(state_dir, state)
        _generation_snapshot(state_dir, state, network)
    print(f"Created fresh history with seed {seed}: {path}", flush=True)


def initialize_from_generation(source_dir: Path, generation: int, state_dir: Path, *,
                               opponent_path: str, opponent_level: int = 0,
                               simulations: int | None = None, seed: int | None = None) -> None:
    """Start a separate Stockfish-opponent history from a copy of an existing generation.

    The weights and Adam state are copied exactly; the source history is only read.
    The replay buffer starts empty so the new experiment learns only from its own games.
    """
    if state_dir.exists():
        raise FileExistsError(f"Training history already exists at {state_dir}; it will not be replaced")
    path = source_dir / "generations" / f"generation-{generation:08d}.pt"
    source = _read(path)
    _validate_checkpoint(source)
    if source["generation"] != generation:
        raise ValueError("Generation snapshot number does not match its filename")
    if seed is None:
        seed = random.SystemRandom().randrange(2**63)
    config = {**asdict(SelfPlayConfig()), **source["last_selfplay_config"]}
    config.update(opponent_level=opponent_level, opponent_path=str(opponent_path))
    if simulations is not None:
        config.update(white_simulations=simulations, black_simulations=simulations)
    selfplay = SelfPlayConfig(**config)
    training = TrainingConfig(**source["last_training_config"])
    network = PolicyValueNetwork()
    network.load_state_dict(source["model"], strict=True)
    optimizer = torch.optim.Adam(network.parameters())
    optimizer.load_state_dict(source["optimizer"])
    torch.manual_seed(seed)
    state = {
        "format_version": REFERENCED_FORMAT_VERSION, "architecture": ARCHITECTURE,
        "run_id": uuid.uuid4().hex, "sequence": 0, "generation": generation,
        "total_games": source["total_games"], "total_updates": source["total_updates"],
        "positions_since_import": 0, "replay_capacity": training.replay_capacity, "replay_refs": [],
        "last_selfplay_config": asdict(selfplay), "last_training_config": asdict(training),
        "last_checkpoint_interval": source["last_checkpoint_interval"],
        "handled_stop_requests": [], "cycle": None, "opponent_results": [],
    }
    state_dir.mkdir(parents=True, exist_ok=False)
    with _training_lock(state_dir):
        write_new_file(state_dir / "initialization.json", json.dumps({
            "run_id": state["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed, "architecture": ARCHITECTURE, "experiment": "stockfish-opponent",
            "parent": {"state_dir": str(source_dir), "generation": generation,
                       "model_id": source["model_id"]},
            "selfplay": asdict(selfplay), "training": asdict(training),
        }, indent=2) + "\n")
        _sync_state(state, network, optimizer, ReplayBuffer(training.replay_capacity),
                    random.Random(seed), random.Random(seed))
        checkpoint = _new_checkpoint(state_dir, state)
        _generation_snapshot(state_dir, state, network)
    print(f"Created Stockfish-opponent history from generation {generation}: {checkpoint}", flush=True)


def initialize_from_diagnostic(source: Path, state_dir: Path) -> None:
    """Import an explicitly chosen completed diagnostic, including Adam and replay."""
    if state_dir.exists():
        raise FileExistsError(f"Training history already exists at {state_dir}; it will not be replaced")
    saved = _read(source)
    for key in ("model", "optimizer", "replay", "metadata", "updates", "torch_rng_state", "sampling_rng_state"):
        if key not in saved:
            raise ValueError(f"Diagnostic is missing {key}: {source}")
    meta = saved["metadata"]
    training = TrainingConfig(**meta["training"])
    selfplay = SelfPlayConfig(**meta["selfplay"])
    network = PolicyValueNetwork()
    network.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=training.learning_rate)
    optimizer.load_state_dict(saved["optimizer"])
    if saved["updates"] < 1 or not optimizer.state:
        raise ValueError("Diagnostic has no completed optimizer updates")
    replay = _load_replay(saved["replay"], training.replay_capacity)
    if not replay.examples:
        raise ValueError("Diagnostic has no replay examples")
    initial_path = source.parent / "initial.pt"
    initial = _read(initial_path)
    if initial.get("metadata", {}).get("run_id") != meta["run_id"]:
        raise ValueError("Diagnostic starting model does not match the selected run")
    initial_network = PolicyValueNetwork()
    initial_network.load_state_dict(initial["model"], strict=True)
    sample_rng = random.Random()
    sample_rng.setstate(saved["sampling_rng_state"])
    play_rng = random.Random(meta["seed"])
    torch.set_rng_state(saved["torch_rng_state"])
    state = {
        "format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
        "run_id": uuid.uuid4().hex, "sequence": 0, "generation": 1,
        "source_diagnostic_id": meta["run_id"],
        "source_model_id": saved.get("final_model_id"),
        "total_games": len(saved.get("game_ids", [])),
        "total_updates": saved["updates"],
        "positions_since_import": 0,
        "replay_capacity": training.replay_capacity,
        "last_selfplay_config": asdict(selfplay),
        "last_training_config": asdict(training),
        "last_checkpoint_interval": 1,
        "handled_stop_requests": [],
        "cycle": None,
    }
    state_dir.mkdir(parents=True, exist_ok=False)
    with _training_lock(state_dir):
        write_new_file(state_dir / "generations" / "generation-00000000-model.pt", initial_path.read_bytes())
        _sync_state(state, network, optimizer, replay, sample_rng, play_rng)
        # The diagnostic did not save its CUDA RNG state; do not pretend otherwise.
        state["cuda_rng_states"] = None
        path = _new_checkpoint(state_dir, state)
        _generation_snapshot(state_dir, state, network)
    print(f"Imported diagnostic {meta['run_id']} with {saved['updates']} updates: {path}", flush=True)


def _restore(state: dict, device: torch.device, state_dir: Path | None = None):
    network = PolicyValueNetwork().to(device)
    network.load_state_dict(state["model"], strict=True)
    if model_id(network) != state.get("model_id"):
        raise ValueError("Checkpoint model identifier does not match its weights")
    optimizer = torch.optim.Adam(network.parameters())
    optimizer.load_state_dict(state["optimizer"])
    if state["format_version"] == FORMAT_VERSION:
        replay = _load_replay(state["replay"], state["replay_capacity"])
    else:
        if state_dir is None:
            raise ValueError("The training directory is required to load replay references")
        replay = _load_replay_refs(state_dir, state)
    sample_rng = random.Random()
    sample_rng.setstate(state["sampling_rng_state"])
    play_rng = random.Random()
    play_rng.setstate(state["selfplay_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_states = state.get("cuda_rng_states")
    if device.type == "cuda" and cuda_states is not None:
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            print("WARNING: CUDA device count changed; CUDA RNG state cannot be restored exactly.",
                  file=sys.stderr, flush=True)
    elif device.type == "cuda":
        print("WARNING: No saved CUDA RNG state; exact reproduction is not guaranteed.",
              file=sys.stderr, flush=True)
    return network, optimizer, replay, sample_rng, play_rng


def _start_cycle(state_dir: Path, state: dict, selfplay: SelfPlayConfig,
                 training: TrainingConfig, checkpoint_interval: int, play_rng: random.Random) -> None:
    cycle_id = uuid.uuid4().hex
    seed = random.SystemRandom().randrange(2**63)
    play_rng.seed(seed)
    cycle = {
        "id": cycle_id, "phase": "selfplay", "games_done": 0, "updates_done": 0,
        "selfplay": asdict(selfplay), "training": asdict(training), "seed": seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "game_ids": [], "metrics": [],
        "checkpoint_interval": checkpoint_interval,
    }
    write_new_file(state_dir / "sessions" / f"{cycle_id}.json", json.dumps(cycle, indent=2) + "\n")
    state["cycle"] = cycle
    state["last_selfplay_config"] = asdict(selfplay)
    state["last_training_config"] = asdict(training)
    state["last_checkpoint_interval"] = checkpoint_interval


def _record_path(state_dir: Path, cycle: dict) -> Path:
    return state_dir / "game_records" / f"{cycle['id']}-{cycle['games_done']:08d}.pt"


def _archive_game(games_dir: Path, record: dict) -> None:
    path = games_dir / f"game-{record['game_id']}.pgn"
    if path.exists():
        if path.read_text(encoding="utf-8") != record["pgn"]:
            raise ValueError(f"Existing game differs from journal; refusing to overwrite: {path}")
    else:
        write_new_file(path, record["pgn"])


def _reconcile_archives(state_dir: Path, games_dir: Path) -> None:
    directory = state_dir / "game_records"
    if directory.exists():
        for path in sorted(directory.glob("*.pt")):
            _archive_game(games_dir, _read(path))


def _apply_game_record(state_dir: Path, state: dict, network: PolicyValueNetwork, replay: ReplayBuffer,
                       play_rng: random.Random, games_dir: Path, record: dict) -> None:
    cycle = state["cycle"]
    if (record["cycle_id"] != cycle["id"] or record["game_index"] != cycle["games_done"]
            or record["model_id"] != model_id(network)):
        raise ValueError("Game journal does not match the checkpoint; refusing unsafe recovery")
    claim_draws = bool(cycle["selfplay"].get("claim_draws", False))
    exploration = state.get("exploration")
    expected_exploration = bool(exploration and exploration["games_remaining"] > 0
                                and exploration["deferred_cycle"] != cycle["id"])
    if bool(record.get("exploration_mode", False)) != expected_exploration:
        raise ValueError("Game journal exploration mode does not match the checkpoint")
    game = chess.pgn.read_game(io.StringIO(record["pgn"]))
    if game is None or game.errors or game.headers.get("GameId") != record["game_id"]:
        raise ValueError("Invalid game journal PGN")
    board = game.board()
    positions = []
    moves = list(game.mainline_moves())
    if len(moves) != len(record["visits"]):
        raise ValueError("Game journal visit count does not match its PGN")
    for move, visits in zip(moves, record["visits"]):
        positions.append(PositionVisits(board.copy(stack=True),
                                        {chess.Move.from_uci(uci): count for uci, count in visits.items()}))
        if move not in board.legal_moves:
            raise ValueError("Game journal contains an illegal move")
        board.push(move)
    _archive_game(games_dir, record)
    if state["format_version"] == REFERENCED_FORMAT_VERSION:
        batch = ReplayBuffer(max(1, len(positions)))
        batch.add_game(game, positions, claim_draws)
        source = "game-" + record["game_id"]
        if re.fullmatch(r"game-[0-9a-f]{32}", source) is None:
            raise ValueError("Invalid game identifier in journal")
        shard_path = state_dir / "replay_sources" / f"{source}.pt"
        examples = _replay_data(batch)
        if shard_path.exists():
            saved = _read_replay_source(state_dir, source)
            if len(saved) != len(examples) or any(
                not torch.equal(a["position"], b["position"])
                or set(a["legal_actions"]) != set(b["legal_actions"])
                or a["visit_counts"] != b["visit_counts"] or a["value"] != b["value"]
                for a, b in zip(saved, examples)
            ):
                raise ValueError(f"Existing replay source conflicts with game journal: {shard_path}")
        else:
            write_new_file(shard_path, _pack({"source_id": source, "examples": examples}))
        replay.examples.extend(batch.examples)
        state["replay_refs"] = (state["replay_refs"] +
                                [[source, i] for i in range(len(batch.examples))])[-state["replay_capacity"]:]
    else:
        replay.add_game(game, positions, claim_draws)
    play_rng.setstate(record["selfplay_rng_state"])
    cycle["games_done"] += 1
    cycle["game_ids"].append(record["game_id"])
    state["total_games"] += 1
    state["positions_since_import"] += len(positions)
    if "OpponentLevel" in game.headers:
        network_white = game.headers["NetworkColour"] == "white"
        state.setdefault("opponent_results", []).append([
            int(game.headers["OpponentLevel"]),
            network_score(game.headers["Result"], network_white), record["game_id"],
        ])
    if expected_exploration:
        exploration["games_remaining"] -= 1


def request_stop(state_dir: Path) -> Path:
    state, _ = load_latest(state_dir)
    cycle = state["cycle"]
    if cycle is None or cycle["phase"] == "complete":
        raise ValueError("No active training cycle to stop")
    path = state_dir / "stop_requests" / f"{cycle['id']}-{uuid.uuid4().hex}.json"
    write_new_file(path, json.dumps({"cycle_id": cycle["id"], "requested_utc":
                                    datetime.now(timezone.utc).isoformat()}) + "\n")
    return path


def _stop_requested(state_dir: Path, state: dict) -> bool:
    directory = state_dir / "stop_requests"
    requests = sorted(directory.glob("*.json")) if directory.exists() else []
    handled = state.setdefault("handled_stop_requests", [])
    fresh = [path.name for path in requests if path.name not in handled]
    handled.extend(fresh)
    return bool(fresh)


def configure_exploration(state_dir: Path, games: int = 20, simulations: int = 25,
                          value_scale: float = 2.0) -> None:
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ValueError("Exploration games must be a positive integer")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 2:
        raise ValueError("Exploration requires at least two simulations")
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be positive and finite")
    with _training_lock(state_dir):
        state, _ = load_latest(state_dir, validate_replay=True)
        if state.get("exploration", {}).get("games_remaining", 0) > 0:
            raise ValueError("An exploration phase is already active or scheduled")
        _, _, replay, _, _ = _restore(state, torch.device("cpu"), state_dir)
        _migrate_replay(state_dir, state, replay)
        baseline = dict(state["last_selfplay_config"])
        baseline.update(white_simulations=simulations, black_simulations=simulations,
                        alternate_budgets=False, value_scale=value_scale)
        selfplay = SelfPlayConfig(**baseline)
        cycle = state["cycle"]
        deferred = cycle["id"] if cycle and cycle["phase"] != "complete" else None
        configuration = {
            "id": uuid.uuid4().hex,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "games_total": games, "games_remaining": games,
            "deferred_cycle": deferred,
            "noise_fraction": 0.5, "temperature": 1.25, "temperature_plies": 10000,
            "baseline_selfplay": asdict(selfplay),
        }
        write_new_file(state_dir / "configurations" / f"{configuration['id']}.json",
                       json.dumps(configuration, indent=2) + "\n")
        state["last_selfplay_config"] = asdict(selfplay)
        state["exploration"] = configuration
        path = _new_checkpoint(state_dir, state)
        print(f"Configured {games} exploratory games, {simulations} simulations per side, "
              f"rewards +{value_scale:g}/0/-{value_scale:g}: {path}", flush=True)
        if deferred:
            print("The unfinished cycle retains its old settings; the new profile starts afterward.", flush=True)


def configure_learning(state_dir: Path, replay_capacity: int = 8192,
                       checkpoint_interval: int = 8, include_untrained_cycle: bool = False) -> None:
    """Schedule a full pass, optionally including an active cycle before its first update."""
    if isinstance(replay_capacity, bool) or not isinstance(replay_capacity, int) or replay_capacity < 1:
        raise ValueError("Replay capacity must be a positive integer")
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int) or checkpoint_interval < 1:
        raise ValueError("Checkpoint interval must be a positive integer")
    with _training_lock(state_dir):
        state, _ = load_latest(state_dir, validate_replay=True)
        if replay_capacity < state["replay_capacity"]:
            raise ValueError("Reducing replay capacity is not supported")
        cycle = state["cycle"]
        if include_untrained_cycle:
            if not cycle or cycle["phase"] != "selfplay" or cycle["updates_done"] != 0:
                raise ValueError("The active cycle must still be generating games with no training updates")
            if state["format_version"] != REFERENCED_FORMAT_VERSION:
                raise ValueError("The active cycle must use referenced replay storage")
            for game_id in cycle["game_ids"]:
                _read_replay_source(state_dir, "game-" + game_id)
        training = dict(state["last_training_config"])
        training.update(replay_capacity=replay_capacity, learning_mode="fresh_pass")
        TrainingConfig(**training)
        if include_untrained_cycle:
            active_training = dict(cycle["training"])
            active_training.update(replay_capacity=replay_capacity, learning_mode="fresh_pass")
            TrainingConfig(**active_training)
        configuration = {
            "id": uuid.uuid4().hex,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training": training,
            "checkpoint_interval": checkpoint_interval,
            "applies_after_cycle": (cycle["id"] if cycle and cycle["phase"] != "complete"
                                    and not include_untrained_cycle else None),
            "includes_untrained_cycle": cycle["id"] if include_untrained_cycle else None,
        }
        write_new_file(state_dir / "configurations" / f"{configuration['id']}.json",
                       json.dumps(configuration, indent=2) + "\n")
        state["last_training_config"] = training
        state["last_checkpoint_interval"] = checkpoint_interval
        if include_untrained_cycle:
            cycle["training"] = active_training
            cycle["checkpoint_interval"] = checkpoint_interval
            state["replay_capacity"] = replay_capacity
        path = _new_checkpoint(state_dir, state)
        print(f"Scheduled fresh-game training with replay capacity {replay_capacity}: {path}", flush=True)
        if configuration["applies_after_cycle"]:
            print("The unfinished cycle keeps its saved settings.", flush=True)


def _prepare_fresh_pass(state_dir: Path, cycle: dict, rng: random.Random) -> None:
    refs = []
    for game_id in cycle["game_ids"]:
        source = "game-" + game_id
        examples = _read_replay_source(state_dir, source)
        refs.extend([[source, index] for index in range(len(examples))])
    if not refs:
        raise ValueError("A fresh training pass requires completed game positions")
    rng.shuffle(refs)
    cycle["fresh_order"] = refs
    cycle["fresh_cursor"] = 0


def _fresh_batch(state_dir: Path, state: dict, replay: ReplayBuffer,
                 rng: random.Random, cache: dict[str, list[dict]],
                 batch_size: int) -> tuple[list[ReplayExample], int]:
    cycle = state["cycle"]
    cursor = cycle["fresh_cursor"]
    selected = cycle["fresh_order"][cursor:cursor + batch_size]
    if not selected:
        raise ValueError("Fresh training pass has no remaining positions")
    examples = []
    for source, index in selected:
        if source not in cache:
            cache[source] = _read_replay_source(state_dir, source)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cache[source]):
            raise ValueError("Invalid fresh training position index")
        examples.append(ReplayExample(**cache[source][index]))
    fresh_sources = {"game-" + game_id for game_id in cycle["game_ids"]}
    older = [example for ref, example in zip(state["replay_refs"], replay.examples)
             if ref[0] not in fresh_sources]
    examples.extend(rng.sample(older, min(len(selected), len(older))))
    return examples, len(selected)


def _resume_body(
    pool_box: dict,
    workers: int,
    state_dir: Path,
    games_dir: Path,
    device: torch.device,
    *,
    overrides: dict | None = None,
    minutes: float | None = None,
    recover_previous: bool = False,
    stop_flag: list[bool] | None = None,
) -> None:
    if minutes is not None and (not math.isfinite(minutes) or minutes <= 0):
        raise ValueError("minutes must be positive and finite")
    state, path = load_latest(state_dir, recover_previous=recover_previous, validate_replay=True)
    network, optimizer, replay, sample_rng, play_rng = _restore(state, device, state_dir)
    _migrate_replay(state_dir, state, replay)
    _reconcile_archives(state_dir, games_dir)
    print(f"Resuming {path}; generation {state['generation']}, {state['total_updates']} updates, device {device}.",
          flush=True)
    if state["cycle"] is None or state["cycle"]["phase"] == "complete":
        _generation_snapshot(state_dir, state, network)
    elif overrides:
        print("Settings overrides will apply when the unfinished cycle completes; "
              "it continues with its saved settings.", flush=True)
    invocation_id = uuid.uuid4().hex
    write_new_file(state_dir / "invocations" / f"{invocation_id}.json", json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": path.name,
        "device": str(device), "minutes": minutes, "overrides": overrides or {},
        "recover_previous": recover_previous,
    }, indent=2) + "\n")
    deadline = None if minutes is None else time.monotonic() + minutes * 60
    stopped = stop_flag if stop_flag is not None else [False]
    completed_in_invocation = 0
    fresh_cache: dict[str, list[dict]] = {}

    def save() -> None:
        _sync_state(state, network, optimizer, replay, sample_rng, play_rng)
        _new_checkpoint(state_dir, state)

    def pool_has_finished_game() -> bool:
        pool = pool_box.get("pool")
        cycle = state["cycle"]
        return bool(pool and cycle and cycle["phase"] == "selfplay"
                    and pool.is_done(cycle["games_done"]))

    while True:
        requested = _stop_requested(state_dir, state)
        stopping = bool(stopped[0] or requested or (deadline is not None and time.monotonic() >= deadline))
        if stopping and not pool_has_finished_game():
            save()
            print("Stopped safely; resume without configuration overrides.", flush=True)
            return
        cycle = state["cycle"]
        if cycle is None or cycle["phase"] == "complete":
            if completed_in_invocation and (deadline is None or time.monotonic() >= deadline):
                return
            sp = {**asdict(SelfPlayConfig()), **state["last_selfplay_config"]}
            tc = dict(state["last_training_config"])
            if sp.get("opponent_level") is not None:
                advanced = next_level(sp["opponent_level"], state.get("opponent_results", []))
                if advanced != sp["opponent_level"]:
                    print(f"Opponent ladder: advancing from level {sp['opponent_level']} to {advanced}.",
                          flush=True)
                    state.setdefault("opponent_ladder_history", []).append({
                        "from": sp["opponent_level"], "to": advanced,
                        "games": state["total_games"],
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    sp["opponent_level"] = advanced
            checkpoint_interval = state["last_checkpoint_interval"]
            for name, value in (overrides or {}).items():
                if name in sp:
                    sp[name] = value
                elif name in tc:
                    tc[name] = value
                elif name == "checkpoint_interval":
                    checkpoint_interval = value
                else:
                    raise ValueError(f"Unknown configuration option: {name}")
            if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int) or checkpoint_interval < 1:
                raise ValueError("checkpoint_interval must be a positive number of updates")
            selfplay = SelfPlayConfig(**sp)
            training = TrainingConfig(**tc)
            if training.replay_capacity < state["replay_capacity"]:
                raise ValueError("Reducing replay capacity on resume is not supported")
            if training.replay_capacity > state["replay_capacity"]:
                enlarged = ReplayBuffer(training.replay_capacity)
                enlarged.examples.extend(replay.examples)
                replay = enlarged
                state["replay_capacity"] = training.replay_capacity
            for group in optimizer.param_groups:
                group["lr"] = training.learning_rate
            _start_cycle(state_dir, state, selfplay, training, checkpoint_interval, play_rng)
            save()
            cycle = state["cycle"]
            overrides = None
        selfplay = SelfPlayConfig(**cycle["selfplay"])
        training = TrainingConfig(**cycle["training"])
        if cycle["phase"] == "selfplay":
            journal_path = _record_path(state_dir, cycle)
            if journal_path.exists():
                record = _read(journal_path)
                print(f"Recovering completed game from {journal_path}", flush=True)
            else:
                positions: list[PositionVisits] = []
                exploration = state.get("exploration")

                def config_for(offset: int) -> tuple[bool, SelfPlayConfig]:
                    active = bool(exploration and exploration["games_remaining"] - offset > 0
                                  and exploration["deferred_cycle"] != cycle["id"])
                    return active, (replace(selfplay, noise_fraction=exploration["noise_fraction"],
                                            temperature=exploration["temperature"],
                                            temperature_plies=exploration["temperature_plies"])
                                    if active else selfplay)

                exploratory, game_config = config_for(0)
                white, black = budgets_for_game(game_config, cycle["games_done"])
                if workers > 1:
                    pool = pool_box.get("pool")
                    if pool is None:
                        pool = pool_box["pool"] = GamePool(network, workers)
                    last = cycle["games_done"] + (0 if stopping else workers)
                    last = min(training.games, max(last, cycle["games_done"] + 1))
                    fingerprint = model_id(network)
                    for offset, index in enumerate(range(cycle["games_done"], last)):
                        active, config = config_for(offset)
                        w, b = budgets_for_game(config, index)
                        target = state_dir / "game_records" / f"{cycle['id']}-{index:08d}.pt"
                        if not target.exists():
                            context = {
                                "cycle_id": cycle["id"], "game_index": index,
                                "model_id": fingerprint, "generation": state["generation"],
                                "exploratory": active, "selfplay_rng_state": play_rng.getstate(),
                                "journal_path": str(target.resolve()),
                            }
                            pool.submit(index, game_seed(cycle["seed"], index), config, w, b, context)
                    if cycle["games_done"] in pool.pending:
                        pool.result(cycle["games_done"])
                    record = _read(journal_path)
                else:
                    game = play_game(lambda board: predict_legal_moves(network, board), game_config,
                                     play_rng, white, black, positions, cycle["games_done"])
                    game_id = uuid.uuid4().hex
                    fingerprint = model_id(network)
                    label_game(game, game_id, cycle["id"], fingerprint, state["generation"],
                               cycle["games_done"], white, black)
                    game.headers["ExplorationMode"] = "on" if exploratory else "off"
                    game.headers["RewardScale"] = str(game_config.value_scale)
                    record = {
                        "cycle_id": cycle["id"], "game_index": cycle["games_done"],
                        "game_id": game_id, "model_id": fingerprint,
                        "pgn": str(game) + "\n\n",
                        "visits": [{move.uci(): count for move, count in position.visits.items()}
                                   for position in positions],
                        "selfplay_rng_state": play_rng.getstate(),
                        "exploration_mode": exploratory, "selfplay_config": asdict(game_config),
                    }
                    write_new_file(journal_path, _pack(record))
            _apply_game_record(state_dir, state, network, replay, play_rng, games_dir, record)
            if cycle["games_done"] == training.games:
                cycle["phase"] = "training"
                if training.learning_mode == "fresh_pass":
                    _prepare_fresh_pass(state_dir, cycle, sample_rng)
            save()
            headers = chess.pgn.read_headers(io.StringIO(record["pgn"]))
            summary = f"{headers['Result']} by {headers.get('Termination', '?').lower()} in {headers.get('PlyCount', '?')} plies"
            if "OpponentLevel" in headers:
                network_white = headers["NetworkColour"] == "white"
                score = network_score(headers["Result"], network_white)
                verdict = {1.0: "network won", 0.5: "draw", 0.0: "network lost"}[score]
                summary += (f"; network was {headers['NetworkColour']} vs Stockfish level "
                            f"{headers['OpponentLevel']}: {verdict}")
            print(f"Saved game {record['game_id']}; {cycle['games_done']}/{training.games}. Result: {summary}.",
                  flush=True)
            if record.get("exploration_mode"):
                print(f"Exploration games remaining: {state['exploration']['games_remaining']}", flush=True)
        elif cycle["phase"] == "training":
            if pool_box.get("pool") is not None:
                pool_box.pop("pool").close()
            if training.learning_mode == "fresh_pass":
                examples, fresh_count = _fresh_batch(
                    state_dir, state, replay, sample_rng, fresh_cache, training.batch_size)
            else:
                examples = replay.sample(training.batch_size, sample_rng)
                fresh_count = 0
            policy, value = train_step(network, optimizer, examples, value_scale=selfplay.value_scale)
            if fresh_count:
                cycle["fresh_cursor"] += fresh_count
            cycle["updates_done"] += 1
            state["total_updates"] += 1
            cycle["metrics"].append({"update": cycle["updates_done"], "policy_loss": policy, "value_loss": value})
            complete = (cycle["fresh_cursor"] == len(cycle["fresh_order"])
                        if training.learning_mode == "fresh_pass"
                        else cycle["updates_done"] == training.updates)
            if complete:
                cycle["phase"] = "complete"
                state["generation"] += 1
            if cycle["phase"] == "complete" or cycle["updates_done"] % cycle["checkpoint_interval"] == 0:
                save()
            progress = (f"{cycle['fresh_cursor']}/{len(cycle['fresh_order'])} new positions"
                        if training.learning_mode == "fresh_pass"
                        else f"{cycle['updates_done']}/{training.updates}")
            print(f"Update {progress}: policy {policy:.4f}, value {value:.4f}.", flush=True)
            if cycle["phase"] == "complete":
                _generation_snapshot(state_dir, state, network)
                fresh_cache.clear()
                completed_in_invocation += 1
                print(f"Generation {state['generation']} saved.", flush=True)
        else:
            raise ValueError(f"Unknown cycle phase: {cycle['phase']}")


def resume(
    state_dir: Path,
    games_dir: Path,
    device: torch.device,
    *,
    overrides: dict | None = None,
    minutes: float | None = None,
    recover_previous: bool = False,
    stop_flag: list[bool] | None = None,
    workers: int = 1,
) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    with _training_lock(state_dir):
        pool_box: dict = {}
        try:
            _resume_body(pool_box, workers, state_dir, games_dir, device, overrides=overrides,
                         minutes=minutes, recover_previous=recover_previous, stop_flag=stop_flag)
        finally:
            if pool_box.get("pool") is not None:
                pool_box["pool"].close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain the main resumable training history.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fresh = subparsers.add_parser("initialize-fresh")
    fresh.add_argument("--state-dir", type=Path, required=True)
    fresh.add_argument("--seed", type=int)
    fresh.add_argument("--simulations", type=int, default=1)
    fresh.add_argument("--reward-scale", type=float, default=1.0)
    initialize = subparsers.add_parser("import-diagnostic")
    opponent_init = subparsers.add_parser(
        "initialize-opponent",
        help="start a separate Stockfish-opponent history from a copy of a generation")
    opponent_init.add_argument("--source-state-dir", type=Path, required=True)
    opponent_init.add_argument("--generation", type=int, required=True)
    opponent_init.add_argument("--state-dir", type=Path, required=True)
    opponent_init.add_argument("--stockfish-path", required=True)
    opponent_init.add_argument("--level", type=int, default=0)
    opponent_init.add_argument("--simulations", type=int)
    opponent_init.add_argument("--seed", type=int)
    initialize.add_argument("source", type=Path)
    initialize.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    status = subparsers.add_parser("status")
    status.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    configure = subparsers.add_parser("configure-exploration")
    configure.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    configure.add_argument("--games", type=int, default=20)
    configure.add_argument("--simulations", type=int, default=25)
    configure.add_argument("--reward-scale", type=float, default=2.0)
    learning = subparsers.add_parser("configure-learning")
    learning.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    learning.add_argument("--replay-capacity", type=int, default=8192)
    learning.add_argument("--checkpoint-interval", type=int, default=8)
    learning.add_argument("--include-untrained-cycle", action="store_true")
    run = subparsers.add_parser("resume")
    run.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    run.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--minutes", type=float)
    run.add_argument("--workers", type=int, default=0,
                     help="parallel self-play processes; 0 chooses automatically")
    run.add_argument("--recover-previous", action="store_true")
    run.add_argument("--games", type=int)
    run.add_argument("--updates", type=int)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--learning-rate", type=float)
    run.add_argument("--white-simulations", type=int)
    run.add_argument("--black-simulations", type=int)
    run.add_argument("--checkpoint-interval", type=int)
    run.add_argument("--opponent-level", type=int,
                     help="manually set the Stockfish ladder level (the ladder otherwise advances itself)")
    run.add_argument("--opponent-path", type=str)
    claims = run.add_mutually_exclusive_group()
    claims.add_argument("--claim-draws", dest="claim_draws", action="store_true", default=None,
                        help="end games at threefold repetition or the 50-move rule (applies to the next cycle)")
    claims.add_argument("--no-claim-draws", dest="claim_draws", action="store_false")
    budget_mode = run.add_mutually_exclusive_group()
    budget_mode.add_argument("--alternate-budgets", dest="alternate_budgets",
                             action="store_true", default=None)
    budget_mode.add_argument("--fixed-budgets", dest="alternate_budgets", action="store_false")
    args = parser.parse_args()
    if args.command == "initialize-fresh":
        initialize_fresh(args.state_dir, args.seed, simulations=args.simulations,
                         value_scale=args.reward_scale)
    elif args.command == "initialize-opponent":
        initialize_from_generation(args.source_state_dir, args.generation, args.state_dir,
                                   opponent_path=args.stockfish_path, opponent_level=args.level,
                                   simulations=args.simulations, seed=args.seed)
    elif args.command == "import-diagnostic":
        initialize_from_diagnostic(args.source, args.state_dir)
    elif args.command == "stop":
        print(f"Stop requested: {request_stop(args.state_dir)}")
    elif args.command == "status":
        state, path = load_latest(args.state_dir)
        print(f"Checkpoint: {path}\nGeneration: {state['generation']}\n"
              f"Games: {state['total_games']}\nUpdates: {state['total_updates']}\n"
              f"Exploration: {state.get('exploration')}\nCycle: {state['cycle']}")
    elif args.command == "configure-exploration":
        configure_exploration(args.state_dir, args.games, args.simulations, args.reward_scale)
    elif args.command == "configure-learning":
        configure_learning(args.state_dir, args.replay_capacity, args.checkpoint_interval,
                           args.include_untrained_cycle)
    else:
        names = ("games", "updates", "batch_size", "learning_rate",
                 "white_simulations", "black_simulations",                  "alternate_budgets", "checkpoint_interval",
                                  "claim_draws", "opponent_level", "opponent_path")
        overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
        workers = args.workers or max(1, min(8, (os.cpu_count() or 2) // 2))
        flag = [False]
        previous_handler = signal.getsignal(signal.SIGINT)

        def stop_after_step(signum, frame):
            flag[0] = True
            print("\nStop requested; finishing the current game or training step.", flush=True)

        signal.signal(signal.SIGINT, stop_after_step)
        try:
            resume(args.state_dir, args.games_dir, _device(args.device), overrides=overrides,
                   minutes=args.minutes, recover_previous=args.recover_previous, stop_flag=flag, workers=workers)
        finally:
            signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
