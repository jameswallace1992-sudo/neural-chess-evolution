"""Randomly initialized policy-value network; no chess evaluation rules."""

import chess
import torch
from torch import nn

from .encoding import ACTION_SIZE, POSITION_CHANNELS, encode_move, encode_position, legal_action_mask


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class PolicyValueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(POSITION_CHANNELS, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(64) for _ in range(4)))
        # Channel order matches (destination, promotion); spatial position is origin.
        self.policy = nn.Conv2d(64, ACTION_SIZE // 64, 1)
        self.value = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(self.stem(inputs))
        return self.policy(features).flatten(start_dim=1), self.value(features)


@torch.inference_mode()
def predict_legal_moves(
    network: PolicyValueNetwork, board: chess.Board
) -> tuple[dict[chess.Move, float], float]:
    """Return legal move probabilities and value from the side-to-move perspective."""
    if board.is_game_over():
        raise ValueError("Cannot predict moves for a finished game")
    mask = legal_action_mask(board)
    device = next(network.parameters()).device
    logits, value = network(encode_position(board).unsqueeze(0).to(device))
    probabilities = torch.softmax(logits[0].masked_fill(~mask.to(device), -torch.inf), dim=0)
    legal_moves = list(board.legal_moves)
    legal_probabilities = probabilities[[encode_move(move) for move in legal_moves]].cpu().tolist()
    moves = dict(zip(legal_moves, legal_probabilities))
    return moves, value[0, 0].item()
