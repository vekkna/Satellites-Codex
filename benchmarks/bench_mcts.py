import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcts import MCTS, SatellitesAdapter
from engine import SatellitesGame


def run_benchmark(decisions: int, iterations: int, rollout_depth: int, seed: int):
    adapter = SatellitesAdapter()
    game = SatellitesGame(headless=True)
    mcts = MCTS(adapter, iterations=iterations, rollout_depth=rollout_depth, seed=seed)

    start = time.perf_counter()
    chosen = 0
    for _ in range(decisions):
        if game.state == "GAME_OVER":
            break
        action, _ = mcts.select_action(game)
        game.apply_action(action)
        chosen += 1
    elapsed = time.perf_counter() - start

    decisions_per_sec = chosen / elapsed if elapsed > 0 else 0.0
    rollouts_per_sec = (chosen * iterations) / elapsed if elapsed > 0 else 0.0
    avg_ms = (elapsed / chosen * 1000.0) if chosen > 0 else 0.0

    print(f"decisions={chosen}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"avg_decision_ms={avg_ms:.3f}")
    print(f"decisions_per_sec={decisions_per_sec:.3f}")
    print(f"rollouts_per_sec={rollouts_per_sec:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCTS on Satellites engine.")
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(args.decisions, args.iterations, args.rollout_depth, args.seed)


if __name__ == "__main__":
    main()
