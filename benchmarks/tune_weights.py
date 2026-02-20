import argparse
import pathlib
import random
import sys
from typing import Dict, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcts import MCTS, SatellitesAdapter
from engine import SatellitesGame


TUNABLE_KEYS = [
    "score_diff",
    "near_win",
    "move_bot_capture",
    "move_bot_capture_stack_scale",
    "move_bot_safe_approach",
    "move_tank_adj_enemy_bot",
    "move_tank_vs_tank_win",
    "add_tank_near_artefact",
    "sat_add_tank_bonus",
    "eval_safe_bot_near_artefact",
]


def play_game(p0: MCTS, p1: MCTS, think_s: float) -> int:
    game = SatellitesGame(headless=True)
    while game.state != "GAME_OVER":
        agent = p0 if game.turn == 0 else p1
        action, _ = agent.select_action_for_time(game, think_s, min_iterations=6)
        ok = game.apply_action(action)
        if not ok:
            legal = game.legal_actions()
            if not legal:
                break
            game.apply_action(legal[0])
    return game.winner if game.winner is not None else -1


def head_to_head(
    challenger_weights: Dict[str, float],
    baseline_weights: Dict[str, float],
    games: int,
    think_s: float,
    seed: int,
) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    c_adapter = SatellitesAdapter(weights=challenger_weights)
    b_adapter = SatellitesAdapter(weights=baseline_weights)
    c_mcts = MCTS(c_adapter, iterations=100, rollout_depth=14, seed=rng.randrange(10**9))
    b_mcts = MCTS(b_adapter, iterations=100, rollout_depth=14, seed=rng.randrange(10**9))

    c_wins = 0
    b_wins = 0
    draws = 0
    for g in range(games):
        if g % 2 == 0:
            winner = play_game(c_mcts, b_mcts, think_s)
            if winner == 0:
                c_wins += 1
            elif winner == 1:
                b_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(b_mcts, c_mcts, think_s)
            if winner == 0:
                b_wins += 1
            elif winner == 1:
                c_wins += 1
            else:
                draws += 1
    return c_wins, b_wins, draws


def mutate(base: Dict[str, float], rng: random.Random, sigma_frac: float) -> Dict[str, float]:
    out = base.copy()
    k = rng.choice(TUNABLE_KEYS)
    cur = out[k]
    span = max(1.0, abs(cur))
    delta = rng.gauss(0.0, sigma_frac * span)
    out[k] = max(0.0, cur + delta)
    return out


def tune(rounds: int, games: int, think_s: float, seed: int, sigma_frac: float):
    rng = random.Random(seed)
    best = SatellitesAdapter().get_weights()
    best_score = 0.5
    print(f"start_score={best_score:.3f}")

    for r in range(1, rounds + 1):
        cand = mutate(best, rng, sigma_frac=sigma_frac)
        c_wins, b_wins, draws = head_to_head(cand, best, games, think_s, rng.randrange(10**9))
        score = (c_wins + 0.5 * draws) / games if games > 0 else 0.0
        improved = score > best_score
        print(
            f"round={r} score={score:.3f} "
            f"(c={c_wins}, b={b_wins}, d={draws}) "
            f"{'ACCEPT' if improved else 'REJECT'}"
        )
        if improved:
            best = cand
            best_score = score

    print("best_score=", f"{best_score:.3f}")
    print("best_weights_subset=")
    for k in TUNABLE_KEYS:
        print(f"  {k}={best[k]:.4f}")


def main():
    p = argparse.ArgumentParser(description="Local random-search tuner for Satellites MCTS weights.")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--think-s", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma-frac", type=float, default=0.20)
    args = p.parse_args()
    tune(args.rounds, args.games, args.think_s, args.seed, args.sigma_frac)


if __name__ == "__main__":
    main()
