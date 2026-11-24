import contextlib
from dataclasses import replace
import io
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import torch

from neural_chess.network import PolicyValueNetwork
from neural_chess import parallel, persistence
from neural_chess.persistence import (
    _restore, _training_lock, configure_exploration, configure_learning, initialize_fresh, initialize_from_diagnostic,
    load_latest, model_id, request_stop, resume,
)
from neural_chess.selfplay import SelfPlayConfig, play_game
from neural_chess.training import TrainingConfig, run_diagnostic
from neural_chess.training import train_step as real_train_step


def scripted_predictor(board):
    sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
    target = chess.Move.from_uci(sequence[len(board.move_stack)])
    return {move: float(move == target) for move in board.legal_moves}, 0.0


def scripted_play(predict, config, rng, white, black, positions, *_):
    return play_game(scripted_predictor, config, rng, white, black, positions)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_dir = self.root / "state"
        self.games_dir = self.root / "games"
        with patch("neural_chess.selfplay.predict_legal_moves",
                   side_effect=lambda network, board: scripted_predictor(board)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.diagnostic = run_diagnostic(
                    PolicyValueNetwork(), SelfPlayConfig(noise_fraction=0, temperature=0),
                    TrainingConfig(games=1, updates=2, batch_size=4, replay_capacity=10),
                    7, self.root / "runs", self.games_dir,
                )
                initialize_from_diagnostic(self.diagnostic / "final.pt", self.state_dir)

    def test_fresh_history_records_explicit_search_settings_without_exploration_phase(self):
        fresh = self.root / "configured_fresh"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(fresh, seed=123, simulations=25, value_scale=2)
            configure_learning(fresh, replay_capacity=8192)
        state, _ = load_latest(fresh, validate_replay=True)
        self.assertEqual((state["total_games"], state["total_updates"]), (0, 0))
        self.assertFalse(state["optimizer"]["state"])
        self.assertNotIn("exploration", state)
        self.assertEqual(state["last_selfplay_config"]["white_simulations"], 25)
        self.assertEqual(state["last_selfplay_config"]["black_simulations"], 25)
        self.assertEqual(state["last_selfplay_config"]["value_scale"], 2)
        self.assertEqual(state["last_selfplay_config"]["noise_fraction"], 0.25)
        self.assertEqual(state["last_training_config"]["games"], 1)
        self.assertEqual(state["last_training_config"]["learning_mode"], "fresh_pass")

    def test_fresh_pass_covers_evicted_positions_across_stop_and_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            configure_learning(self.state_dir, replay_capacity=10, checkpoint_interval=1)
        stop = [False]
        seen = []
        original_batch = persistence._fresh_batch

        def capture_batch(state_dir, state, replay, rng, cache, batch_size):
            cursor = state["cycle"]["fresh_cursor"]
            result = original_batch(state_dir, state, replay, rng, cache, batch_size)
            seen.extend(tuple(ref) for ref in state["cycle"]["fresh_order"][cursor:cursor + result[1]])
            return result

        def train_then_stop(*args, **kwargs):
            result = real_train_step(*args, **kwargs)
            stop[0] = True
            return result

        with patch("neural_chess.persistence.play_game", side_effect=scripted_play), \
             patch("neural_chess.persistence._fresh_batch", side_effect=capture_batch), \
             contextlib.redirect_stdout(io.StringIO()):
            with patch("neural_chess.persistence.train_step", side_effect=train_then_stop):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"games": 3, "batch_size": 2}, stop_flag=stop)
            paused, _ = load_latest(self.state_dir, validate_replay=True)
            self.assertEqual(paused["cycle"]["fresh_cursor"], 2)
            resume(self.state_dir, self.games_dir, torch.device("cpu"))
        state, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(state["cycle"]["phase"], "complete")
        self.assertEqual(state["cycle"]["fresh_cursor"], 12)
        self.assertEqual(state["cycle"]["updates_done"], 6)
        self.assertEqual(len(state["replay_refs"]), 10)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(set(seen)), 12)
        self.assertEqual(set(seen), {tuple(ref) for ref in state["cycle"]["fresh_order"]})

    def test_configure_learning_preserves_unfinished_cycle_and_optimizer(self):
        stop = [False]

        def finish_then_stop(*args):
            game = scripted_play(*args)
            stop[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=finish_then_stop), \
             contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"),
                   overrides={"games": 2, "updates": 1}, stop_flag=stop)
            before, _ = load_latest(self.state_dir)
            configure_learning(self.state_dir, replay_capacity=20)
            after, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(after["cycle"], before["cycle"])
        self.assertEqual(after["model_id"], before["model_id"])
        self.assertEqual(after["total_updates"], before["total_updates"])
        self.assertEqual(after["replay_capacity"], before["replay_capacity"])
        self.assertEqual(after["last_training_config"]["learning_mode"], "fresh_pass")
        self.assertEqual(after["last_training_config"]["replay_capacity"], 20)
        for key, values in before["optimizer"]["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, after["optimizer"]["state"][key][name]))

    def test_fresh_pass_mixes_older_positions(self):
        sizes = []

        def capture(*args, **kwargs):
            sizes.append(len(args[2]))
            return real_train_step(*args, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()):
            configure_learning(self.state_dir, replay_capacity=20, checkpoint_interval=1)
            with patch("neural_chess.persistence.play_game", side_effect=scripted_play), \
                 patch("neural_chess.persistence.train_step", side_effect=capture):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"games": 1, "batch_size": 2})
        self.assertEqual(sizes, [4, 4])
        state, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(state["replay_capacity"], 20)
        self.assertEqual(state["cycle"]["fresh_cursor"], 4)

    def test_claim_draws_override_reaches_self_play_and_is_recorded(self):
        seen = []

        def capture_play(predict, config, *args):
            seen.append(config.claim_draws)
            return scripted_play(predict, config, *args)

        with patch("neural_chess.persistence.play_game", side_effect=capture_play), \
             contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"),
                   overrides={"games": 1, "batch_size": 2, "claim_draws": True})
        self.assertEqual(seen, [True])
        state, _ = load_latest(self.state_dir)
        self.assertTrue(state["last_selfplay_config"]["claim_draws"])

    def test_enable_learning_for_active_untrained_cycle(self):
        stop = [False]

        def finish_then_stop(*args):
            game = scripted_play(*args)
            stop[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=finish_then_stop), \
             contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"),
                   overrides={"games": 2, "batch_size": 2}, stop_flag=stop)
            before, _ = load_latest(self.state_dir)
            configure_learning(self.state_dir, replay_capacity=20, checkpoint_interval=1,
                               include_untrained_cycle=True)
            configured, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(configured["model_id"], before["model_id"])
        self.assertEqual(configured["cycle"]["games_done"], 1)
        self.assertEqual(configured["cycle"]["training"]["learning_mode"], "fresh_pass")
        self.assertEqual(configured["replay_capacity"], 20)
        with patch("neural_chess.persistence.play_game", side_effect=scripted_play), \
             contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"))
        after, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(after["cycle"]["fresh_cursor"], 8)
        self.assertEqual(after["cycle"]["updates_done"], 4)
        self.assertEqual(after["cycle"]["phase"], "complete")

    def test_interrupted_fresh_pass_matches_uninterrupted_training(self):
        stop = [False]
        games_played = 0

        def finish_games(*args):
            nonlocal games_played
            result = scripted_play(*args)
            games_played += 1
            if games_played == 2:
                stop[0] = True
            return result

        with contextlib.redirect_stdout(io.StringIO()):
            configure_learning(self.state_dir, replay_capacity=20, checkpoint_interval=1)
            with patch("neural_chess.persistence.play_game", side_effect=finish_games):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"games": 2, "batch_size": 2}, stop_flag=stop)
        baseline = self.root / "fresh_baseline"
        shutil.copytree(self.state_dir, baseline)
        stop = [False]

        def train_then_stop(*args, **kwargs):
            result = real_train_step(*args, **kwargs)
            stop[0] = True
            return result

        with contextlib.redirect_stdout(io.StringIO()):
            with patch("neural_chess.persistence.train_step", side_effect=train_then_stop):
                resume(self.state_dir, self.games_dir, torch.device("cpu"), stop_flag=stop)
            resume(self.state_dir, self.games_dir, torch.device("cpu"))
            resume(baseline, self.root / "fresh_baseline_games", torch.device("cpu"))
        resumed, _ = load_latest(self.state_dir)
        uninterrupted, _ = load_latest(baseline)
        self.assertEqual(resumed["cycle"]["fresh_cursor"], 8)
        self.assertEqual(resumed["sampling_rng_state"], uninterrupted["sampling_rng_state"])
        for key in resumed["model"]:
            self.assertTrue(torch.equal(resumed["model"][key], uninterrupted["model"][key]), key)
        for key, values in resumed["optimizer"]["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, uninterrupted["optimizer"]["state"][key][name]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fresh_pass_trains_on_cuda(self):
        with contextlib.redirect_stdout(io.StringIO()):
            configure_learning(self.state_dir, replay_capacity=20, checkpoint_interval=1)
            before, _ = load_latest(self.state_dir)
            with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
                resume(self.state_dir, self.games_dir, torch.device("cuda"),
                       overrides={"games": 1, "batch_size": 2})
        after, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(after["cycle"]["fresh_cursor"], 4)
        self.assertEqual(after["cycle"]["updates_done"], 2)
        self.assertNotEqual(after["model_id"], before["model_id"])

    def test_parallel_journals_recover_even_when_next_game_was_not_applied(self):
        class ImmediatePool:
            def __init__(self, network, workers):
                self.pending = {}

            def submit(self, index, seed, config, white, black, context):
                if index not in self.pending:
                    parallel._play(seed, config, white, black, context)
                    self.pending[index] = True

            def result(self, index):
                self.pending.pop(index)

            def is_done(self, index):
                return index in self.pending

            def close(self):
                pass

        with patch.object(parallel, "_network", PolicyValueNetwork()), \
             patch.object(parallel, "play_game", side_effect=scripted_play), \
             patch("neural_chess.persistence.GamePool", ImmediatePool), \
             patch("neural_chess.persistence._apply_game_record", side_effect=RuntimeError("power loss")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"games": 2, "updates": 1}, workers=2)
        self.assertEqual(len(list((self.state_dir / "game_records").glob("*.pt"))), 2)
        with patch("neural_chess.persistence.play_game", side_effect=AssertionError("game was replayed")), \
             contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"), workers=1)
        state, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(state["cycle"]["games_done"], 2)
        self.assertEqual(len(state["cycle"]["game_ids"]), 2)
        self.assertEqual(state["cycle"]["phase"], "complete")

    def test_parallel_workers_play_a_cycle_and_the_history_is_valid(self):
        fresh = self.root / "parallel"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(fresh, seed=5)
            configure_exploration(fresh, games=2, simulations=2, value_scale=2)
            resume(fresh, self.games_dir, torch.device("cpu"), overrides={"games": 2, "updates": 1},
                   workers=2)
        state, _ = load_latest(fresh, validate_replay=True)
        self.assertEqual((state["generation"], state["total_games"], state["total_updates"]), (1, 2, 1))
        self.assertEqual(state["exploration"]["games_remaining"], 0)
        self.assertEqual(len(state["cycle"]["game_ids"]), 2)

    def test_fresh_history_starts_empty_and_trains_without_touching_existing_history(self):
        original, original_path = load_latest(self.state_dir)
        original_bytes = original_path.read_bytes()
        fresh = self.root / "fresh"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_fresh(fresh, seed=123)
            configure_exploration(fresh, games=2, simulations=25, value_scale=2)
        initial, _ = load_latest(fresh, validate_replay=True)
        self.assertEqual((initial["generation"], initial["total_games"], initial["total_updates"]), (0, 0, 0))
        self.assertFalse(initial["optimizer"]["state"])
        self.assertEqual(initial["replay_refs"], [])
        self.assertEqual(initial["exploration"]["games_remaining"], 2)
        self.assertNotEqual(initial["model_id"], original["model_id"])
        self.assertEqual(original_path.read_bytes(), original_bytes)
        with self.assertRaises(FileExistsError):
            initialize_fresh(fresh, seed=123)
        def finish_game(predict, config, rng, white, black, positions, *_):
            return play_game(scripted_predictor, replace(config, noise_fraction=0, temperature=0),
                             rng, 1, 1, positions)

        with patch("neural_chess.persistence.play_game", side_effect=finish_game):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(fresh, self.games_dir, torch.device("cpu"))
        trained, _ = load_latest(fresh, validate_replay=True)
        self.assertEqual((trained["generation"], trained["total_games"], trained["total_updates"]), (1, 1, 1))
        self.assertNotEqual(trained["model_id"], initial["model_id"])
        self.assertTrue(trained["optimizer"]["state"])
        self.assertEqual(trained["exploration"]["games_remaining"], 1)
        self.assertEqual(len(trained["replay_refs"]), 4)
        self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_import_preserves_model_optimizer_and_replay(self):
        source = torch.load(self.diagnostic / "final.pt", map_location="cpu", weights_only=True)
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["generation"], 1)
        self.assertEqual(state["total_updates"], 2)
        self.assertEqual(len(state["replay"]), len(source["replay"]))
        for name, value in source["model"].items():
            self.assertTrue(torch.equal(state["model"][name], value))
        source_steps = [v["step"].item() for v in source["optimizer"]["state"].values()]
        saved_steps = [v["step"].item() for v in state["optimizer"]["state"].values()]
        self.assertEqual(source_steps, saved_steps)
        self.assertEqual(len(list((self.state_dir / "generations").glob("*.pt"))), 2)
        with self.assertRaises(FileExistsError):
            initialize_from_diagnostic(self.diagnostic / "final.pt", self.state_dir)

    def test_replay_migration_preserves_data_and_detects_missing_source(self):
        old, old_path = load_latest(self.state_dir)
        old_bytes = old_path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            configure_exploration(self.state_dir, games=2, simulations=25, value_scale=2)
        new, new_path = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(new["format_version"], 2)
        self.assertNotIn("replay", new)
        self.assertEqual(old_path.read_bytes(), old_bytes)
        self.assertEqual(old["model_id"], new["model_id"])
        self.assertEqual(old["total_updates"], new["total_updates"])
        self.assertEqual(old["sampling_rng_state"], new["sampling_rng_state"])
        for key, values in old["optimizer"]["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, new["optimizer"]["state"][key][name]))
        _, _, replay, _, _ = _restore(new, torch.device("cpu"), self.state_dir)
        for original, example in zip(old["replay"], replay.examples):
            self.assertTrue(torch.equal(original["position"], example.position))
            self.assertEqual(original["visit_counts"], example.visit_counts)
            self.assertEqual(original["value"], example.value)
        self.assertLess(new_path.stat().st_size, old_path.stat().st_size)
        source = self.state_dir / "replay_sources" / (new["replay_refs"][0][0] + ".pt")
        source.unlink()
        with self.assertRaises(ValueError):
            load_latest(self.state_dir, validate_replay=True)
        with contextlib.redirect_stderr(io.StringIO()):
            recovered, recovered_path = load_latest(self.state_dir, recover_previous=True, validate_replay=True)
        self.assertEqual(recovered_path, old_path)
        self.assertEqual(recovered["model_id"], old["model_id"])

    def test_exploration_game_count_survives_stop_and_ends_on_time(self):
        with contextlib.redirect_stdout(io.StringIO()):
            configure_exploration(self.state_dir, games=2, simulations=25, value_scale=2)
        stop = [False]
        configs = []

        def capture_game(predict, config, rng, white, black, positions, *_):
            configs.append(config)
            actual = replace(config, noise_fraction=0, temperature=0)
            game = play_game(scripted_predictor, actual, rng, 1, 1, positions)
            if len(configs) == 1:
                stop[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=capture_game):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"games": 3, "updates": 1}, stop_flag=stop)
                paused, _ = load_latest(self.state_dir)
                self.assertEqual(paused["exploration"]["games_remaining"], 1)
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
        state, _ = load_latest(self.state_dir, validate_replay=True)
        self.assertEqual(state["exploration"]["games_remaining"], 0)
        self.assertEqual(state["cycle"]["games_done"], 3)
        self.assertEqual([config.noise_fraction for config in configs], [0.5, 0.5, 0])
        self.assertEqual([config.temperature_plies for config in configs], [10000, 10000, 30])
        self.assertTrue(all(config.value_scale == 2 for config in configs))
        self.assertTrue(all(config.white_simulations == config.black_simulations == 25 for config in configs))
        self.assertEqual(len(state["replay_refs"]), 10)
        self.assertEqual(len(_restore(state, torch.device("cpu"), self.state_dir)[2].examples), 10)
        modes = []
        for game_id in state["cycle"]["game_ids"]:
            with (self.games_dir / f"game-{game_id}.pgn").open(encoding="utf-8") as file:
                game = chess.pgn.read_game(file)
            modes.append(game.headers["ExplorationMode"])
            self.assertEqual(game.headers["RewardScale"], "2")
        self.assertEqual(modes, ["on", "on", "off"])

    def test_new_profile_waits_for_unfinished_cycle(self):
        flag = [False]

        def finish_then_stop(*args):
            game = scripted_play(*args)
            flag[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=finish_then_stop):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"updates": 1}, stop_flag=flag)
                configure_exploration(self.state_dir, games=2, simulations=25, value_scale=2)
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["cycle"]["phase"], "complete")
        self.assertEqual(state["exploration"]["games_remaining"], 2)
        self.assertEqual(state["cycle"]["selfplay"]["value_scale"], 1)
        self.assertEqual(state["last_selfplay_config"]["value_scale"], 2)

    def test_safe_stop_then_resume_preserves_progress(self):
        before, _ = load_latest(self.state_dir)
        stop = [False]

        def finish_then_stop(*args):
            game = scripted_play(*args)
            stop[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=finish_then_stop):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"),
                       overrides={"updates": 1}, stop_flag=stop)
        paused, _ = load_latest(self.state_dir)
        self.assertEqual(paused["cycle"]["phase"], "training")
        self.assertEqual(paused["cycle"]["games_done"], 1)
        self.assertEqual(paused["total_updates"], before["total_updates"])
        self.assertEqual(len(list(self.games_dir.glob("*.pgn"))), 2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            resume(self.state_dir, self.games_dir, torch.device("cpu"), overrides={"updates": 2},
                   stop_flag=[True])
        self.assertIn("will apply when the unfinished cycle completes", output.getvalue())
        self.assertEqual(load_latest(self.state_dir)[0]["cycle"]["training"]["updates"], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"))
        after, _ = load_latest(self.state_dir)
        self.assertEqual(after["cycle"]["phase"], "complete")
        self.assertEqual(after["generation"], 2)
        self.assertEqual(after["total_updates"], 3)
        self.assertEqual(after["total_games"], 2)
        self.assertEqual(len(after["replay_refs"]), 8)
        self.assertFalse(torch.equal(before["model"]["stem.0.weight"], after["model"]["stem.0.weight"]))
        self.assertEqual([v["step"].item() for v in after["optimizer"]["state"].values()],
                         [v["step"].item() + 1 for v in before["optimizer"]["state"].values()])
        self.assertEqual(len(list((self.state_dir / "generations").glob("*.pt"))), 3)
        self.assertGreaterEqual(len(list((self.state_dir / "checkpoints").glob("*.pt"))), 4)

    def test_stop_request_is_acknowledged_and_can_resume(self):
        flag = [False]

        def finish_then_stop(*args):
            game = scripted_play(*args)
            flag[0] = True
            return game

        with patch("neural_chess.persistence.play_game", side_effect=finish_then_stop):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"), stop_flag=flag)
        request = request_stop(self.state_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"))
        paused, _ = load_latest(self.state_dir)
        self.assertIn(request.name, paused["handled_stop_requests"])
        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["cycle"]["phase"], "complete")

    def test_game_journal_recovers_after_interruption(self):
        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with patch("neural_chess.persistence._apply_game_record", side_effect=RuntimeError("power loss")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "power loss"):
                        resume(self.state_dir, self.games_dir, torch.device("cpu"), overrides={"updates": 1})
        self.assertEqual(len(list((self.state_dir / "game_records").glob("*.pt"))), 1)
        self.assertEqual(len(list(self.games_dir.glob("*.pgn"))), 1)
        with patch("neural_chess.persistence.play_game", side_effect=AssertionError("must recover, not replay")):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["cycle"]["phase"], "complete")
        self.assertEqual(state["cycle"]["games_done"], 1)
        self.assertEqual(len(list(self.games_dir.glob("*.pgn"))), 2)
        self.assertEqual(state["total_updates"], 3)

    def test_interrupted_updates_match_uninterrupted_cpu_training(self):
        baseline_dir = self.root / "baseline"
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_from_diagnostic(self.diagnostic / "final.pt", baseline_dir)
        stop = [False]

        def one_update_then_stop(*args, **kwargs):
            losses = real_train_step(*args, **kwargs)
            stop[0] = True
            return losses

        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("neural_chess.persistence.train_step", side_effect=one_update_then_stop):
                    resume(self.state_dir, self.games_dir, torch.device("cpu"),
                           overrides={"updates": 2}, stop_flag=stop)
                paused, _ = load_latest(self.state_dir)
                self.assertEqual(paused["cycle"]["updates_done"], 1)
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
                resume(baseline_dir, self.root / "baseline_games", torch.device("cpu"),
                       overrides={"updates": 2})
        resumed, _ = load_latest(self.state_dir)
        baseline, _ = load_latest(baseline_dir)
        self.assertEqual(resumed["total_updates"], baseline["total_updates"])
        self.assertEqual(resumed["sampling_rng_state"], baseline["sampling_rng_state"])
        for key in resumed["model"]:
            self.assertTrue(torch.equal(resumed["model"][key], baseline["model"][key]), key)
        for key, values in resumed["optimizer"]["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, baseline["optimizer"]["state"][key][name]))

    def test_interruption_after_pgn_save_does_not_duplicate_game(self):
        checkpoint = persistence._new_checkpoint
        def fail_after_game(*args):
            cycle = args[1]["cycle"]
            if cycle and cycle["games_done"] == 1 and cycle["phase"] == "training":
                raise RuntimeError("interrupted before checkpoint")
            return checkpoint(*args)

        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with patch("neural_chess.persistence._new_checkpoint", side_effect=fail_after_game):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        resume(self.state_dir, self.games_dir, torch.device("cpu"), overrides={"updates": 1})
        before = {path.name: path.read_bytes() for path in self.games_dir.glob("*.pgn")}
        self.assertEqual(len(before), 2)
        with patch("neural_chess.persistence.play_game", side_effect=AssertionError("game was replayed")):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"))
        after = {path.name: path.read_bytes() for path in self.games_dir.glob("*.pgn")}
        self.assertEqual(after, before)
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["cycle"]["games_done"], 1)

    def test_stop_request_at_cycle_boundary_is_not_lost(self):
        checkpoint = persistence._new_checkpoint
        requested = []

        def request_before_completion(*args):
            if args[1]["cycle"] and args[1]["cycle"]["phase"] == "complete" and not requested:
                requested.append(request_stop(self.state_dir))
            return checkpoint(*args)

        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with patch("neural_chess.persistence._new_checkpoint", side_effect=request_before_completion):
                with contextlib.redirect_stdout(io.StringIO()):
                    resume(self.state_dir, self.games_dir, torch.device("cpu"),
                           overrides={"updates": 1}, minutes=1)
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["generation"], 2)
        self.assertEqual(state["cycle"]["phase"], "complete")
        self.assertIn(requested[0].name, state["handled_stop_requests"])

    def test_corrupt_latest_requires_explicit_recovery(self):
        state, previous = load_latest(self.state_dir)
        corrupt = previous.with_name("checkpoint-000000000002.pt")
        corrupt.write_bytes(b"interrupted")
        with self.assertRaises(ValueError):
            load_latest(self.state_dir)
        with contextlib.redirect_stderr(io.StringIO()):
            recovered, path = load_latest(self.state_dir, recover_previous=True)
        self.assertEqual(path, previous)
        self.assertEqual(recovered["model_id"], state["model_id"])
        self.assertEqual(corrupt.read_bytes(), b"interrupted")

    def test_checksum_detects_corrupt_optimizer_or_replay_payload(self):
        state, previous = load_latest(self.state_dir)
        envelope = torch.load(previous, map_location="cpu", weights_only=True)
        altered = bytearray(envelope["payload"])
        altered[len(altered) // 2] ^= 1
        envelope["payload"] = bytes(altered)
        corrupt = previous.with_name("checkpoint-000000000002.pt")
        torch.save(envelope, corrupt)
        with self.assertRaises(ValueError):
            load_latest(self.state_dir)
        with contextlib.redirect_stderr(io.StringIO()):
            recovered, path = load_latest(self.state_dir, recover_previous=True)
        self.assertEqual(path, previous)
        self.assertEqual(recovered["model_id"], state["model_id"])

    def test_completed_journal_restores_missing_pgn(self):
        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cpu"), overrides={"updates": 1})
        state, _ = load_latest(self.state_dir)
        path = self.games_dir / f"game-{state['cycle']['game_ids'][0]}.pgn"
        original = path.read_text(encoding="utf-8")
        path.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            resume(self.state_dir, self.games_dir, torch.device("cpu"), stop_flag=[True])
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_training_history_lock_rejects_concurrent_writer(self):
        with _training_lock(self.state_dir):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with _training_lock(self.state_dir):
                    pass
        with _training_lock(self.state_dir):
            pass

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_model_identifier_is_device_independent(self):
        network = PolicyValueNetwork()
        cpu_id = model_id(network)
        self.assertEqual(model_id(network.to("cuda")), cpu_id)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_resume_optimizer_on_cuda(self):
        with patch("neural_chess.persistence.play_game", side_effect=scripted_play):
            with contextlib.redirect_stdout(io.StringIO()):
                resume(self.state_dir, self.games_dir, torch.device("cuda"), overrides={"updates": 1})
        state, _ = load_latest(self.state_dir)
        self.assertEqual(state["total_updates"], 3)
        self.assertTrue(all(v["step"].item() == 3 for v in state["optimizer"]["state"].values()))
        self.assertTrue(all(t.device.type == "cpu" for t in state["model"].values()))


if __name__ == "__main__":
    unittest.main()
