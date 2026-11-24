"""Play several self-play games at once in worker processes.

Weights never change while a cycle is playing games, so every worker uses a CPU copy
of the same network. Parallel games use seeds derived from the cycle and game index.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import random
import signal
import uuid

import torch

from .network import PolicyValueNetwork, predict_legal_moves
from .selfplay import PositionVisits, SelfPlayConfig, label_game, play_game, write_new_file

_network: PolicyValueNetwork | None = None


def _init_worker(state_dict: dict) -> None:
    global _network
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process coordinates stopping
    torch.set_num_threads(1)
    _network = PolicyValueNetwork()
    _network.load_state_dict(state_dict)
    _network.eval()


def game_seed(cycle_seed: int, game_index: int) -> int:
    return random.Random(f"{cycle_seed}:{game_index}").getrandbits(63)


def _play(seed: int, config: SelfPlayConfig, white: int, black: int, context: dict) -> None:
    assert _network is not None
    positions: list[PositionVisits] = []
    game = play_game(lambda board: predict_legal_moves(_network, board), config,
                     random.Random(seed), white, black, positions, context["game_index"])
    game_id = uuid.uuid4().hex
    label_game(game, game_id, context["cycle_id"], context["model_id"], context["generation"],
               context["game_index"], white, black)
    game.headers["ExplorationMode"] = "on" if context["exploratory"] else "off"
    game.headers["RewardScale"] = str(config.value_scale)
    record = {
        "cycle_id": context["cycle_id"], "game_index": context["game_index"],
        "game_id": game_id, "model_id": context["model_id"], "pgn": str(game) + "\n\n",
        "visits": [{move.uci(): count for move, count in position.visits.items()}
                   for position in positions],
        "selfplay_rng_state": context["selfplay_rng_state"],
        "exploration_mode": context["exploratory"], "selfplay_config": asdict(config),
    }
    from .persistence import _pack
    write_new_file(Path(context["journal_path"]), _pack(record))


class GamePool:
    def __init__(self, network: PolicyValueNetwork, workers: int) -> None:
        state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
        self.executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                            initargs=(state,))
        self.workers = workers
        self.pending: dict[int, object] = {}

    def submit(self, index: int, seed: int, config: SelfPlayConfig, white: int, black: int,
               context: dict) -> None:
        if index not in self.pending:
            self.pending[index] = self.executor.submit(_play, seed, config, white, black, context)

    def is_done(self, index: int) -> bool:
        future = self.pending.get(index)
        return future is not None and future.done()

    def result(self, index: int) -> None:
        return self.pending.pop(index).result()

    def close(self) -> None:
        for future in self.pending.values():
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
