from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from typing import Dict, List

import numpy as np
import torch

from agents.alpha_mcts import AlphaMCTS
from engine import SatellitesGame
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder
from rl.model import SatellitesPolicyValueNet


@dataclass
class TrainingExample:
    obs: np.ndarray
    policy: np.ndarray
    value: float


def _owner_unit_breakdown(game: SatellitesGame, owner: int) -> tuple[int, int]:
    game._ensure_cache()
    tank_count = 0
    bot_count = 0
    for cid in game.owner_tank_cells[owner]:
        tank_count += int(game.unit_count[cid])
    for cid in game.owner_bot_cells[owner]:
        bot_count += int(game.unit_count[cid])
    return tank_count, bot_count


def adjudicate_winner(game: SatellitesGame) -> int:
    """Training-time tie-breaker for truncated games: score, then tanks, then bots."""
    if game.winner is not None:
        return int(game.winner)
    s0 = int(game.scores[0])
    s1 = int(game.scores[1])
    if s0 != s1:
        return 0 if s0 > s1 else 1
    t0, b0 = _owner_unit_breakdown(game, 0)
    t1, b1 = _owner_unit_breakdown(game, 1)
    if t0 != t1:
        return 0 if t0 > t1 else 1
    if b0 != b1:
        return 0 if b0 > b1 else 1
    return -1


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
        legal_now = game.legal_actions()
        if not legal_now:
            break
        temp = 1.0 if step < temperature_turn_cutoff else 0.2
        obs = encoder.encode(game)
        try:
            action, info = mcts.select_action(game, temperature=temp)
        except ValueError as exc:
            if "No legal actions from root state." in str(exc):
                break
            raise
        pi = info["policy"].astype(np.float32)
        history.append((obs, pi, int(game.turn)))
        ok = game.apply_action(action)
        if not ok:
            break
        step += 1

    examples: List[TrainingExample] = []
    winner = adjudicate_winner(game)
    for obs, pi, player in history:
        if winner is None or winner == -1:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        examples.append(TrainingExample(obs=obs, policy=pi, value=z))
    return examples


def _run_selfplay_worker_chunk(
    *,
    games: int,
    seed_base: int,
    input_dim: int,
    action_dim: int,
    model_state_dict: Dict[str, torch.Tensor],
    simulations: int,
    device: str,
    max_steps: int,
    temperature_turn_cutoff: int,
    worker_torch_threads: int,
    heuristic_value_weight: float,
    heuristic_action_weight: float,
    heuristic_score_weight: float,
    heuristic_tank_weight: float,
    heuristic_bot_weight: float,
) -> List[TrainingExample]:
    torch.set_num_threads(max(1, int(worker_torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    action_space = GlobalActionSpace()
    encoder = FeatureEncoder()
    model = SatellitesPolicyValueNet(input_dim, action_dim)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    examples: List[TrainingExample] = []
    for game_idx in range(games):
        seed = int(seed_base + game_idx)
        np.random.seed(seed)
        mcts = AlphaMCTS(
            model,
            action_space,
            encoder,
            simulations=simulations,
            device=device,
            seed=seed,
            heuristic_value_weight=heuristic_value_weight,
            heuristic_action_weight=heuristic_action_weight,
            heuristic_score_weight=heuristic_score_weight,
            heuristic_tank_weight=heuristic_tank_weight,
            heuristic_bot_weight=heuristic_bot_weight,
        )
        examples.extend(
            run_selfplay_game(
                mcts,
                encoder,
                max_steps=max_steps,
                temperature_turn_cutoff=temperature_turn_cutoff,
            )
        )
    return examples


def run_selfplay_games_parallel(
    *,
    games: int,
    input_dim: int,
    action_dim: int,
    model_state_dict: Dict[str, torch.Tensor],
    simulations: int,
    device: str = "cpu",
    max_workers: int | None = None,
    seed_base: int = 1,
    max_steps: int = 512,
    temperature_turn_cutoff: int = 20,
    worker_torch_threads: int = 1,
    heuristic_value_weight: float = 0.0,
    heuristic_action_weight: float = 0.0,
    heuristic_score_weight: float = 1.0,
    heuristic_tank_weight: float = 0.18,
    heuristic_bot_weight: float = 0.12,
) -> List[TrainingExample]:
    if games <= 0:
        return []

    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(int(max_workers), games))

    if max_workers == 1:
        return _run_selfplay_worker_chunk(
            games=games,
            seed_base=seed_base,
            input_dim=input_dim,
            action_dim=action_dim,
            model_state_dict=model_state_dict,
            simulations=simulations,
            device=device,
            max_steps=max_steps,
            temperature_turn_cutoff=temperature_turn_cutoff,
            worker_torch_threads=worker_torch_threads,
            heuristic_value_weight=heuristic_value_weight,
            heuristic_action_weight=heuristic_action_weight,
            heuristic_score_weight=heuristic_score_weight,
            heuristic_tank_weight=heuristic_tank_weight,
            heuristic_bot_weight=heuristic_bot_weight,
        )

    chunk_base = games // max_workers
    remainder = games % max_workers
    chunks = [chunk_base + (1 if i < remainder else 0) for i in range(max_workers)]

    examples: List[TrainingExample] = []
    next_seed = int(seed_base)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for chunk in chunks:
            if chunk <= 0:
                continue
            futures.append(
                ex.submit(
                    _run_selfplay_worker_chunk,
                    games=chunk,
                    seed_base=next_seed,
                    input_dim=input_dim,
                    action_dim=action_dim,
                    model_state_dict=model_state_dict,
                    simulations=simulations,
                    device=device,
                    max_steps=max_steps,
                    temperature_turn_cutoff=temperature_turn_cutoff,
                    worker_torch_threads=worker_torch_threads,
                    heuristic_value_weight=heuristic_value_weight,
                    heuristic_action_weight=heuristic_action_weight,
                    heuristic_score_weight=heuristic_score_weight,
                    heuristic_tank_weight=heuristic_tank_weight,
                    heuristic_bot_weight=heuristic_bot_weight,
                )
            )
            next_seed += chunk

        for fut in as_completed(futures):
            examples.extend(fut.result())

    return examples
