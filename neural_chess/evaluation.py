"""Generation-versus-generation matches.

Evaluation only reads saved generation snapshots. It never changes weights,
optimiser state, training histories or any champion, and it never promotes a
network. Both sides search with their own network and the same simulation
budget; there is no external engine or handcrafted evaluation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import chess
import chess.pgn
import torch

from .mcts import Predictor, search, terminal_outcome, visit_policy
from .network import PolicyValueNetwork, predict_legal_moves
from .parallel import game_seed
from .persistence import _read, _validate_checkpoint, model_id
from .selfplay import write_new_file

DEFAULT_EVALUATIONS = Path("evaluations")
_networks: tuple[PolicyValueNetwork, PolicyValueNetwork] | None = None


def load_generation(state_dir: Path, generation: int) -> dict:
    """Read one generation snapshot, checking its integrity envelope and model hash."""
    path = state_dir / "generations" / f"generation-{generation:08d}.pt"
    if not path.exists():
        raise ValueError(f"Generation {generation} has no snapshot in {state_dir}")
    state = _read(path)
    _validate_checkpoint(state)
    network = _network_from(state)
    if state["generation"] != generation or model_id(network) != state["model_id"]:
        raise ValueError(f"Snapshot does not match its recorded generation or model hash: {path}")
    return state


def _network_from(state: dict) -> PolicyValueNetwork:
    network = PolicyValueNetwork()
    network.load_state_dict(state["model"])
    network.eval()
    return network


def _predictor(network: PolicyValueNetwork) -> Predictor:
    return lambda board: predict_legal_moves(network, board)


def play_match_game(
    white_predict: Predictor,
    black_predict: Predictor,
    simulations: int,
    seed: int,
    *,
    value_scale: float = 1.0,
    exploration: float = 1.5,
    claim_draws: bool = False,
    temperature_plies: int = 8,
) -> chess.pgn.Game:
    """Play one game. Each side searches with its own predictor.

    Dirichlet noise is off. The first `temperature_plies` moves are sampled from
    visit counts so repeated games differ; every later move is the most-visited move.
    """
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    while terminal_outcome(board, claim_draws) is None:
        predict = white_predict if board.turn == chess.WHITE else black_predict
        root = search(board, predict, simulations, exploration=exploration, noise_fraction=0,
                      value_scale=value_scale, rng=rng, claim_draws=claim_draws)
        temperature = 1.0 if len(board.move_stack) < temperature_plies else 0
        policy = visit_policy(root, temperature)
        if temperature == 0:
            move = max(policy, key=policy.get)
        else:
            move = rng.choices(list(policy), weights=list(policy.values()), k=1)[0]
        node = node.add_variation(move)
        board.push(move)
    outcome = terminal_outcome(board, claim_draws)
    game.headers["Result"] = outcome.result()
    game.headers["Termination"] = outcome.termination.name
    game.headers["PlyCount"] = str(len(board.move_stack))
    return game


def candidate_score(result: str, candidate_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == candidate_is_white else 0.0


def elo_from_score(score: float) -> float:
    """Elo difference implied by an expected score; infinite at 0 or 1."""
    if score <= 0:
        return -math.inf
    if score >= 1:
        return math.inf
    return -400 * math.log10(1 / score - 1) + 0.0


def summarise(scores: list[float], colours: list[bool], terminations: list[str],
              seed: int = 0, resamples: int = 5000) -> dict:
    """Score, bootstrap interval and a cautious verdict. `scores` are the candidate's."""
    if not scores or not (len(scores) == len(colours) == len(terminations)):
        raise ValueError("Need matching, non-empty result lists")
    games = len(scores)
    mean = sum(scores) / games
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(scores, k=games)) / games for _ in range(resamples))
    low, high = means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))]
    if low > 0.5:
        verdict = "candidate scored higher; interval excludes equal strength"
    elif high < 0.5:
        verdict = "candidate scored lower; interval excludes equal strength"
    else:
        verdict = "inconclusive; interval includes equal strength"
    as_white = [s for s, white in zip(scores, colours) if white]
    as_black = [s for s, white in zip(scores, colours) if not white]
    terminations_count: dict[str, int] = {}
    for name in terminations:
        terminations_count[name] = terminations_count.get(name, 0) + 1
    return {
        "games": games,
        "candidate_wins": sum(s == 1.0 for s in scores),
        "draws": sum(s == 0.5 for s in scores),
        "candidate_losses": sum(s == 0.0 for s in scores),
        "score": mean,
        "score_interval_95": [low, high],
        "elo_difference": elo_from_score(mean),
        "elo_interval_95": [elo_from_score(low), elo_from_score(high)],
        "score_as_white": sum(as_white) / len(as_white) if as_white else None,
        "score_as_black": sum(as_black) / len(as_black) if as_black else None,
        "terminations": terminations_count,
        "verdict": verdict,
        "note": "A verdict is evidence from these games only; it is not a promotion decision.",
    }


def _init_worker(candidate: dict, baseline: dict) -> None:
    global _networks
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _networks = (_network_from({"model": candidate}), _network_from({"model": baseline}))


def _worker_game(seed: int, candidate_is_white: bool, settings: dict) -> str:
    assert _networks is not None
    candidate, baseline = (_predictor(network) for network in _networks)
    white, black = (candidate, baseline) if candidate_is_white else (baseline, candidate)
    return str(play_match_game(white, black, seed=seed, **settings))


def _cpu_copy(model: dict) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.items()}


def run_match(
    state_dir: Path,
    candidate_generation: int,
    baseline_generation: int,
    *,
    games: int = 20,
    simulations: int = 25,
    seed: int = 0,
    claim_draws: bool = False,
    temperature_plies: int = 8,
    workers: int = 1,
    value_scale: float | None = None,
    evaluations_dir: Path = DEFAULT_EVALUATIONS,
) -> Path:
    """Play a colour-balanced match and write PGNs, configuration and results."""
    if isinstance(games, bool) or not isinstance(games, int) or games < 2 or games % 2:
        raise ValueError("games must be an even number of at least 2 so colours stay balanced")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if candidate_generation == baseline_generation:
        raise ValueError("Candidate and baseline must be different generations")
    candidate = load_generation(state_dir, candidate_generation)
    baseline = load_generation(state_dir, baseline_generation)
    if value_scale is None:
        value_scale = float(candidate.get("last_selfplay_config", {}).get("value_scale", 1.0))
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be positive and finite")
    settings = {"simulations": simulations, "value_scale": value_scale,
                "claim_draws": claim_draws, "temperature_plies": temperature_plies}
    evaluation_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    directory = evaluations_dir / evaluation_id
    config = {
        "evaluation_id": evaluation_id, "state_dir": str(state_dir),
        "candidate_generation": candidate_generation, "candidate_model_id": candidate["model_id"],
        "baseline_generation": baseline_generation, "baseline_model_id": baseline["model_id"],
        "games": games, "seed": seed, "workers": workers, "device": "cpu",
        "torch_version": str(torch.__version__), **settings,
        "dirichlet_noise": 0.0, "exploration": 1.5,
        "colours": "candidate plays white in even-numbered games, black in odd-numbered games",
    }
    write_new_file(directory / "config.json", json.dumps(config, indent=2) + "\n")

    candidate_net, baseline_net = _network_from(candidate), _network_from(baseline)
    jobs = [(game_seed(seed, index), index % 2 == 0) for index in range(games)]
    if workers == 1:
        candidate_predict, baseline_predict = _predictor(candidate_net), _predictor(baseline_net)
        texts = []
        for index, (game_seed_value, candidate_is_white) in enumerate(jobs):
            white, black = ((candidate_predict, baseline_predict) if candidate_is_white
                            else (baseline_predict, candidate_predict))
            texts.append(str(play_match_game(white, black, seed=game_seed_value, **settings)))
            print(f"Match game {index + 1}/{games} finished.", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(_cpu_copy(candidate["model"]),
                                           _cpu_copy(baseline["model"]))) as pool:
            futures = [pool.submit(_worker_game, s, white, settings) for s, white in jobs]
            texts = []
            for index, future in enumerate(futures):
                texts.append(future.result())
                print(f"Match game {index + 1}/{games} finished.", flush=True)

    scores, colours, terminations = [], [], []
    for index, (text, (_, candidate_is_white)) in enumerate(zip(texts, jobs)):
        game = chess.pgn.read_game(io.StringIO(text))
        game.headers.update({
            "Event": "Neural Chess Evolution generation match", "Site": "Local",
            "Round": str(index + 1), "EvaluationId": evaluation_id,
            "White": f"Generation {candidate_generation if candidate_is_white else baseline_generation}",
            "Black": f"Generation {baseline_generation if candidate_is_white else candidate_generation}",
            "Simulations": str(simulations), "ClaimDraws": str(claim_draws),
        })
        write_new_file(directory / "games" / f"game-{index + 1:04d}.pgn", str(game) + "\n\n")
        scores.append(candidate_score(game.headers["Result"], candidate_is_white))
        colours.append(candidate_is_white)
        terminations.append(game.headers["Termination"])
    result = summarise(scores, colours, terminations, seed=seed)
    result.update({"evaluation_id": evaluation_id, "candidate_generation": candidate_generation,
                   "baseline_generation": baseline_generation})
    write_new_file(directory / "result.json", json.dumps(result, indent=2) + "\n")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a generation-versus-generation match.")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True, help="newer generation number")
    parser.add_argument("--baseline", type=int, required=True, help="older generation number")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--claim-draws", action="store_true")
    parser.add_argument("--temperature-plies", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--evaluations-dir", type=Path, default=DEFAULT_EVALUATIONS)
    args = parser.parse_args()
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    started = time.monotonic()
    directory = run_match(args.state_dir, args.candidate, args.baseline, games=args.games,
                          simulations=args.simulations, seed=args.seed, claim_draws=args.claim_draws,
                          temperature_plies=args.temperature_plies, workers=workers,
                          value_scale=args.reward_scale, evaluations_dir=args.evaluations_dir)
    result = json.loads((directory / "result.json").read_text())
    print(f"\nGeneration {args.candidate} (candidate) vs generation {args.baseline} (baseline), "
          f"{result['games']} games in {time.monotonic() - started:.0f}s")
    print(f"Candidate: +{result['candidate_wins']} ={result['draws']} -{result['candidate_losses']}  "
          f"score {result['score']:.3f}, 95% interval "
          f"[{result['score_interval_95'][0]:.3f}, {result['score_interval_95'][1]:.3f}]")
    print(f"Elo difference {result['elo_difference']:+.0f} (interval "
          f"{result['elo_interval_95'][0]:+.0f} to {result['elo_interval_95'][1]:+.0f})")
    print(f"As white {result['score_as_white']}, as black {result['score_as_black']}; "
          f"endings {result['terminations']}")
    print(f"{result['verdict']}\n{result['note']}\nSaved to {directory}")


if __name__ == "__main__":
    main()
