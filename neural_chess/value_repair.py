"""Create a recorded, separate history with a reactivated value head."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import torch

from .network import PolicyValueNetwork
from .persistence import (
    REFERENCED_FORMAT_VERSION, _generation_snapshot, _load_replay_refs, _model_id_for_state,
    _new_checkpoint, _read, _training_lock, _validate_checkpoint,
)
from .selfplay import write_new_file


def repair_value_head(source_dir: Path, generation: int, state_dir: Path) -> dict:
    """Fork a completed snapshot, adjusting only its inactive value-convolution bias."""
    if state_dir.exists():
        raise FileExistsError(f"Training history already exists at {state_dir}")
    path = source_dir / "generations" / f"generation-{generation:08d}.pt"
    state = _read(path)
    _validate_checkpoint(state)
    if state["generation"] != generation:
        raise ValueError("Generation snapshot number does not match its filename")
    if state["cycle"] is not None and state["cycle"]["phase"] != "complete":
        raise ValueError("Repair requires a completed generation snapshot")
    if state["format_version"] != REFERENCED_FORMAT_VERSION:
        raise ValueError("Repair requires a history with referenced replay")
    replay = _load_replay_refs(source_dir, state)
    examples = list(replay.examples)
    if not examples:
        raise ValueError("Repair requires stored replay positions")
    indices = [i * len(examples) // min(len(examples), 128)
               for i in range(min(len(examples), 128))]
    inputs = torch.stack([examples[i].position for i in indices])
    network = PolicyValueNetwork()
    network.load_state_dict(state["model"], strict=True)
    network.eval()
    with torch.no_grad():
        features = network.body(network.stem(inputs))
        preactivation = network.value[0](features)
        if (preactivation > 0).any():
            raise ValueError("Value head is active on sampled positions; refusing automatic repair")
        weakest = preactivation.flatten(1).max(dim=1).values.min().item()
        old_bias = network.value[0].bias.item()
        network.value[0].bias.add_(-weakest + 0.05)
        new_bias = network.value[0].bias.item()
        active = network.value[0](features).flatten(1).max(dim=1).values
        values = network.value(features).flatten()
        if (active <= 0).any() or not torch.isfinite(values).all():
            raise ValueError("Value head repair did not activate all sampled positions")
        span = (values.max() - values.min()).item()
    network.zero_grad(set_to_none=True)
    network.value(features.detach()).sum().backward()
    gradient = network.value[0].weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max().item() == 0:
        raise ValueError("Value head repair did not restore a usable gradient")
    if len(examples) > 1 and span == 0:
        raise ValueError("Value head remains constant on sampled positions")

    parent_id = state["model_id"]
    parent_run = state["run_id"]
    state["model"]["value.0.bias"] = network.value[0].bias.detach().clone()
    state["model_id"] = _model_id_for_state(state["model"])
    state["run_id"] = uuid.uuid4().hex
    state["sequence"] = 0
    state["handled_stop_requests"] = []
    repair = {
        "type": "value_head_bias_reactivation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "source_generation": generation,
        "source_run_id": parent_run,
        "source_model_id": parent_id,
        "repaired_model_id": state["model_id"],
        "parameter": "value.0.bias",
        "old_bias": old_bias,
        "new_bias": new_bias,
        "sample_positions": len(indices),
        "target_minimum_preactivation": 0.05,
        "sample_value_span": span,
        "optimizer_preserved": True,
    }
    state["value_head_repair"] = repair
    state_dir.mkdir(parents=True, exist_ok=False)
    with _training_lock(state_dir):
        for source in sorted({ref[0] for ref in state["replay_refs"]}):
            source_path = source_dir / "replay_sources" / f"{source}.pt"
            write_new_file(state_dir / "replay_sources" / f"{source}.pt", source_path.read_bytes())
        _load_replay_refs(state_dir, state)
        write_new_file(state_dir / "initialization.json", json.dumps(repair, indent=2) + "\n")
        checkpoint = _new_checkpoint(state_dir, state)
        _generation_snapshot(state_dir, state, network)
    print(f"Created repaired history: {checkpoint}", flush=True)
    print(f"Value-head bias: {old_bias:.6f} -> {new_bias:.6f}; "
          f"sample value range: {span:.6f} across {len(indices)} positions.", flush=True)
    return repair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    repair_value_head(args.source_state_dir, args.generation, args.state_dir)


if __name__ == "__main__":
    main()
