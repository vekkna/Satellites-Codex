import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcts import MCTS, SatellitesAdapter
from engine import SatellitesGame


def play_game(p0: MCTS, p1: MCTS, think_s: float) -> int:
    game = SatellitesGame(headless=True)
    while game.state != "GAME_OVER":
        agent = p0 if game.turn == 0 else p1
        action, _ = agent.select_action_for_time(game, think_s, min_iterations=10)
        ok = game.apply_action(action)
        if not ok:
            legal = game.legal_actions()
            if not legal:
                break
            game.apply_action(legal[0])
    return game.winner if game.winner is not None else -1


def run(games: int, think_s: float, seed: int):
    rng = random.Random(seed)
    adapter = SatellitesAdapter()

    tt_on = MCTS(
        adapter,
        iterations=120,
        rollout_depth=16,
        seed=rng.randrange(10**9),
        use_transposition=True,
    )
    tt_off = MCTS(
        adapter,
        iterations=120,
        rollout_depth=16,
        seed=rng.randrange(10**9),
        use_transposition=False,
    )

    on_wins = 0
    off_wins = 0
    draws = 0

    for g in range(games):
        if g % 2 == 0:
            winner = play_game(tt_on, tt_off, think_s)
            if winner == 0:
                on_wins += 1
            elif winner == 1:
                off_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(tt_off, tt_on, think_s)
            if winner == 0:
                off_wins += 1
            elif winner == 1:
                on_wins += 1
            else:
                draws += 1

    print(f"games={games}")
    print(f"think_s={think_s:.3f}")
    print(f"tt_on_wins={on_wins}")
    print(f"tt_off_wins={off_wins}")
    print(f"draws={draws}")
    if games > 0:
        print(f"tt_on_win_rate={on_wins / games:.3f}")


def main():
    p = argparse.ArgumentParser(description="Compare MCTS with TT on/off.")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--think-s", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run(args.games, args.think_s, args.seed)


if __name__ == "__main__":
    main()
