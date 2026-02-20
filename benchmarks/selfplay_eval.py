import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcts import MCTS, SatellitesAdapter
from engine import SatellitesGame


def play_game(
    p0_agent: MCTS,
    p1_agent: MCTS,
    *,
    think_s: float,
) -> int:
    game = SatellitesGame(headless=True)
    while game.state != "GAME_OVER":
        agent = p0_agent if game.turn == 0 else p1_agent
        action, _ = agent.select_action_for_time(game, think_s, min_iterations=10)
        ok = game.apply_action(action)
        if not ok:
            # Fallback: if a sampled action becomes invalid due to edge cases, pick first legal.
            legal = game.legal_actions()
            if not legal:
                break
            game.apply_action(legal[0])
    return game.winner if game.winner is not None else -1


def run_eval(games: int, think_s: float, seed: int):
    strong_adapter = SatellitesAdapter()
    base_adapter = SatellitesAdapter(weights={"move_bot_capture": 4.0, "move_bot_capture_stack_scale": 0.2})

    strong_a = MCTS(strong_adapter, iterations=120, rollout_depth=16, seed=seed)
    base_a = MCTS(base_adapter, iterations=120, rollout_depth=16, seed=seed + 1)

    strong_wins = 0
    base_wins = 0
    draws = 0

    for g in range(games):
        # Alternate colors to reduce first-player bias.
        if g % 2 == 0:
            winner = play_game(strong_a, base_a, think_s=think_s)
            if winner == 0:
                strong_wins += 1
            elif winner == 1:
                base_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(base_a, strong_a, think_s=think_s)
            if winner == 0:
                base_wins += 1
            elif winner == 1:
                strong_wins += 1
            else:
                draws += 1

    print(f"games={games}")
    print(f"think_s={think_s:.3f}")
    print(f"strong_wins={strong_wins}")
    print(f"base_wins={base_wins}")
    print(f"draws={draws}")
    if games > 0:
        print(f"strong_win_rate={strong_wins / games:.3f}")


def main():
    p = argparse.ArgumentParser(description="Self-play evaluator for Satellites MCTS configs.")
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--think-s", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run_eval(args.games, args.think_s, args.seed)


if __name__ == "__main__":
    main()
