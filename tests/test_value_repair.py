import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import chess
import torch

from neural_chess.encoding import encode_move, encode_position
from neural_chess.persistence import (
    _generation_snapshot, _new_checkpoint, _pack, _replay_data, _restore, _sync_state,
    initialize_fresh, load_latest,
)
from neural_chess.selfplay import write_new_file
from neural_chess.training import ReplayExample, train_step
from neural_chess.value_repair import repair_value_head


class ValueRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.destination = Path(self.tmp.name) / "repaired"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(self.source, seed=3)
        state, _ = load_latest(self.source)
        network, optimizer, replay, sampling, play = _restore(state, torch.device("cpu"), self.source)
        board = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
            legal = list(board.legal_moves)
            replay.examples.append(ReplayExample(
                encode_position(board), [encode_move(m) for m in legal],
                {encode_move(legal[0]): 1}, -1.0,
            ))
            board.push_uci(move)
        train_step(network, optimizer, list(replay.examples))
        with torch.no_grad():
            network.value[0].bias.fill_(-10)
        source_id = "origin-" + "a" * 32
        write_new_file(self.source / "replay_sources" / f"{source_id}.pt",
                       _pack({"source_id": source_id, "examples": _replay_data(replay)}))
        state["replay_refs"] = [[source_id, i] for i in range(len(replay.examples))]
        state["generation"] = 1
        _sync_state(state, network, optimizer, replay, sampling, play)
        _new_checkpoint(self.source, state)
        _generation_snapshot(self.source, state, network)

    def test_repairs_only_bias_and_preserves_optimizer_and_replay(self):
        source, _ = load_latest(self.source, validate_replay=True)
        with contextlib.redirect_stdout(io.StringIO()):
            details = repair_value_head(self.source, 1, self.destination)
        repaired, _ = load_latest(self.destination, validate_replay=True)
        self.assertNotEqual(source["model_id"], repaired["model_id"])
        self.assertGreater(details["new_bias"], details["old_bias"])
        self.assertGreater(details["sample_value_span"], 0)
        self.assertEqual(source["replay_refs"], repaired["replay_refs"])
        for name in source["model"]:
            if name != "value.0.bias":
                self.assertTrue(torch.equal(source["model"][name], repaired["model"][name]), name)
        self.assertEqual(source["optimizer"]["param_groups"], repaired["optimizer"]["param_groups"])
        for key, entry in source["optimizer"]["state"].items():
            for name, value in entry.items():
                self.assertTrue(torch.equal(value, repaired["optimizer"]["state"][key][name]))
        self.assertEqual(load_latest(self.source)[0]["model_id"], source["model_id"])
        network, optimizer, replay, _, _ = _restore(repaired, torch.device("cpu"), self.destination)
        old_bias = network.value[0].bias.detach().clone()
        train_step(network, optimizer, list(replay.examples))
        self.assertFalse(torch.equal(old_bias, network.value[0].bias))
        with self.assertRaises(FileExistsError):
            repair_value_head(self.source, 1, self.destination)

    def test_refuses_missing_replay_or_already_active_head(self):
        empty = Path(self.tmp.name) / "empty"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(empty, seed=4)
        with self.assertRaisesRegex(ValueError, "stored replay"):
            repair_value_head(empty, 0, self.destination)
        state, _ = load_latest(self.source)
        network, optimizer, replay, sampling, play = _restore(state, torch.device("cpu"), self.source)
        with torch.no_grad():
            network.value[0].bias.fill_(10)
        state["generation"] = 2
        _sync_state(state, network, optimizer, replay, sampling, play)
        _generation_snapshot(self.source, state, network)
        with self.assertRaisesRegex(ValueError, "already|active"):
            repair_value_head(self.source, 2, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
