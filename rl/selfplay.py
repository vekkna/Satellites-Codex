from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from agents.alpha_mcts import AlphaMCTS
from engine import SatellitesGame
from rl.encode import FeatureEncoder


@dataclass
class TrainingExample:
    obs: np.ndarray
    policy: np.ndarray
    value: float


def run_selfplay_game(
    mcts: AlphaMCTS,
    encoder: FeatureEncoder,
    *,
    max_steps: int = 512,
    temperature_turn_cutoff: int = 20,
) -> List[TrainingExample]:
    game = SatellitesGame(headless=True)
    history: List[tuple[np.ndarray, np.ndarray, int]] = []
    step = 0

    while game.state != "GAME_OVER" and step < max_steps:
        temp = 1.0 if step < temperature_turn_cutoff else 0.2
        obs = encoder.encode(game)
        action, info = mcts.select_action(game, temperature=temp)
        pi = info["policy"].astype(np.float32)
        history.append((obs, pi, int(game.turn)))
        ok = game.apply_action(action)
        if not ok:
            break
        step += 1

    examples: List[TrainingExample] = []
    winner = game.winner
    for obs, pi, player in history:
        if winner is None or winner == -1:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        examples.append(TrainingExample(obs=obs, policy=pi, value=z))
    return examples

