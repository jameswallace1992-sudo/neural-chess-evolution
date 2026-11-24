"""Run a small, untrained network demonstration."""

import chess
import torch

from .encoding import ACTION_SIZE, POSITION_CHANNELS
from .network import PolicyValueNetwork, predict_legal_moves


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = PolicyValueNetwork().to(device)
    board = chess.Board()
    probabilities, value = predict_legal_moves(network, board)
    print(f"Device: {device}")
    print(f"Parameters: {sum(parameter.numel() for parameter in network.parameters()):,}")
    print(f"Input: ({POSITION_CHANNELS}, 8, 8); policy: ({ACTION_SIZE},); value: (1,)")
    print(f"Legal moves: {len(probabilities)}; probability sum: {sum(probabilities.values()):.6f}")
    print(f"Untrained value, side to move: {value:.4f}")
    for move, probability in sorted(probabilities.items(), key=lambda item: -item[1])[:5]:
        print(f"  {move.uci()}: {probability:.4f}")


if __name__ == "__main__":
    main()
