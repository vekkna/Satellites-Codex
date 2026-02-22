from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import time
from typing import Deque, List

import numpy as np
import torch
import torch.nn.functional as F

from agents.alpha_mcts import AlphaMCTS
from engine import SatellitesGame
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder
from rl.model import SatellitesPolicyValueNet
from rl.selfplay import TrainingExample, adjudicate_winner, run_selfplay_game, run_selfplay_games_parallel


@dataclass
class TrainConfig:
    # Tuned baseline for a 14-core/20-thread laptop CPU.
    selfplay_games_per_round: int = 24
    rounds: int = 50
    simulations: int = 96
    selfplay_workers: int = 12
    selfplay_worker_torch_threads: int = 1
    selfplay_device: str = "cpu"
    max_steps_per_game: int = 512
    temperature_turn_cutoff: int = 20
    alpha_heuristic_value_weight: float = 0.30
    alpha_heuristic_action_weight: float = 0.60
    alpha_heuristic_score_weight: float = 1.0
    alpha_heuristic_tank_weight: float = 0.18
    alpha_heuristic_bot_weight: float = 0.12
    arena_games_per_round: int = 16
    arena_simulations: int = 128
    arena_promote_threshold: float = 0.52
    batch_size: int = 128
    train_steps_per_round: int = 96
    replay_size: int = 50000
    lr: float = 1e-3
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_rounds: int = 1
    resume_checkpoint: str | None = None


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buf: Deque[TrainingExample] = deque(maxlen=maxlen)

    def extend(self, items: List[TrainingExample]) -> None:
        self.buf.extend(items)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, n: int) -> List[TrainingExample]:
        n = min(n, len(self.buf))
        return random.sample(list(self.buf), n)


@dataclass
class ArenaResult:
    challenger_wins: int
    incumbent_wins: int
    draws: int

    @property
    def games(self) -> int:
        return self.challenger_wins + self.incumbent_wins + self.draws

    @property
    def challenger_win_rate(self) -> float:
        if self.games <= 0:
            return 0.0
        return float(self.challenger_wins) / float(self.games)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.action_space = GlobalActionSpace()
        self.encoder = FeatureEncoder()
        self.model = SatellitesPolicyValueNet(self.encoder.feature_dim, self.action_space.size).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.replay_size)
        self.best_model_state = self._model_state_cpu()
        self.rounds_trained = 0
        self.promotions = 0
        self.last_arena = ArenaResult(challenger_wins=0, incumbent_wins=0, draws=0)
        if self.config.resume_checkpoint:
            self._load_checkpoint(self.config.resume_checkpoint)

    def _model_state_cpu(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _build_model_from_state(self, state: dict, device: str) -> SatellitesPolicyValueNet:
        m = SatellitesPolicyValueNet(self.encoder.feature_dim, self.action_space.size).to(device)
        m.load_state_dict(state)
        m.eval()
        return m

    def _collect_selfplay_examples(self, round_idx: int) -> List[TrainingExample]:
        games = int(self.config.selfplay_games_per_round)
        if games <= 0:
            return []

        workers = int(self.config.selfplay_workers)
        if workers <= 0:
            workers = min(games, os.cpu_count() or 1)

        self.model.eval()
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if workers <= 1:
            mcts = AlphaMCTS(
                self.model,
                self.action_space,
                self.encoder,
                simulations=self.config.simulations,
                device=self.config.selfplay_device,
                heuristic_value_weight=self.config.alpha_heuristic_value_weight,
                heuristic_action_weight=self.config.alpha_heuristic_action_weight,
                heuristic_score_weight=self.config.alpha_heuristic_score_weight,
                heuristic_tank_weight=self.config.alpha_heuristic_tank_weight,
                heuristic_bot_weight=self.config.alpha_heuristic_bot_weight,
            )
            examples: List[TrainingExample] = []
            for _ in range(games):
                examples.extend(
                    run_selfplay_game(
                        mcts,
                        self.encoder,
                        max_steps=self.config.max_steps_per_game,
                        temperature_turn_cutoff=self.config.temperature_turn_cutoff,
                    )
                )
            return examples

        return run_selfplay_games_parallel(
            games=games,
            input_dim=self.encoder.feature_dim,
            action_dim=self.action_space.size,
            model_state_dict=cpu_state,
            simulations=self.config.simulations,
            device=self.config.selfplay_device,
            max_workers=workers,
            seed_base=100000 * (round_idx + 1),
            max_steps=self.config.max_steps_per_game,
            temperature_turn_cutoff=self.config.temperature_turn_cutoff,
            worker_torch_threads=self.config.selfplay_worker_torch_threads,
            heuristic_value_weight=self.config.alpha_heuristic_value_weight,
            heuristic_action_weight=self.config.alpha_heuristic_action_weight,
            heuristic_score_weight=self.config.alpha_heuristic_score_weight,
            heuristic_tank_weight=self.config.alpha_heuristic_tank_weight,
            heuristic_bot_weight=self.config.alpha_heuristic_bot_weight,
        )

    def _arena_match(self, challenger_state: dict, incumbent_state: dict, round_idx: int) -> ArenaResult:
        games = int(self.config.arena_games_per_round)
        if games <= 0:
            return ArenaResult(challenger_wins=0, incumbent_wins=0, draws=0)

        challenger_model = self._build_model_from_state(challenger_state, self.config.selfplay_device)
        incumbent_model = self._build_model_from_state(incumbent_state, self.config.selfplay_device)
        challenger_mcts = AlphaMCTS(
            challenger_model,
            self.action_space,
            self.encoder,
            simulations=self.config.arena_simulations,
            device=self.config.selfplay_device,
            seed=1100000 + round_idx,
            heuristic_value_weight=0.0,
            heuristic_action_weight=0.0,
        )
        incumbent_mcts = AlphaMCTS(
            incumbent_model,
            self.action_space,
            self.encoder,
            simulations=self.config.arena_simulations,
            device=self.config.selfplay_device,
            seed=2200000 + round_idx,
            heuristic_value_weight=0.0,
            heuristic_action_weight=0.0,
        )

        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(games):
            seed = 9000000 + (round_idx * 1000) + game_idx
            random.seed(seed)
            np.random.seed(seed)
            game = SatellitesGame(headless=True)
            challenger_is_p0 = (game_idx % 2 == 0)
            step = 0
            while game.state != "GAME_OVER" and step < self.config.max_steps_per_game:
                legal = game.legal_actions()
                if not legal:
                    break
                if game.turn == 0:
                    agent = challenger_mcts if challenger_is_p0 else incumbent_mcts
                else:
                    agent = incumbent_mcts if challenger_is_p0 else challenger_mcts
                try:
                    action, _ = agent.select_action(game, temperature=0.0, add_root_noise=False)
                except ValueError as exc:
                    if "No legal actions from root state." in str(exc):
                        action = legal[0]
                    else:
                        raise
                ok = game.apply_action(action)
                if not ok:
                    game.apply_action(legal[0])
                step += 1

            winner = adjudicate_winner(game)
            if winner == -1:
                draws += 1
            else:
                challenger_won = (winner == 0 and challenger_is_p0) or (winner == 1 and not challenger_is_p0)
                if challenger_won:
                    wins += 1
                else:
                    losses += 1
        return ArenaResult(challenger_wins=wins, incumbent_wins=losses, draws=draws)

    def _save_checkpoint(
        self,
        *,
        round_number: int,
        promoted: bool,
        train_stats: dict,
        selfplay_examples: int,
    ) -> None:
        if self.config.checkpoint_every_rounds <= 0:
            return
        if round_number % self.config.checkpoint_every_rounds != 0:
            return

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "round_number": round_number,
            "model_state": self._model_state_cpu(),
            "best_model_state": {k: v.clone() for k, v in self.best_model_state.items()},
            "optimizer_state": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "replay": {
                "size": len(self.buffer),
                "capacity": self.buffer.buf.maxlen,
            },
            "training": {
                "promoted_this_round": bool(promoted),
                "promotions_total": int(self.promotions),
                "selfplay_examples_this_round": int(selfplay_examples),
                "loss": float(train_stats.get("loss", 0.0)),
                "policy_loss": float(train_stats.get("policy_loss", 0.0)),
                "value_loss": float(train_stats.get("value_loss", 0.0)),
                "arena": {
                    "wins": int(self.last_arena.challenger_wins),
                    "losses": int(self.last_arena.incumbent_wins),
                    "draws": int(self.last_arena.draws),
                    "win_rate": float(self.last_arena.challenger_win_rate),
                },
            },
        }
        round_path = ckpt_dir / f"round_{round_number:04d}.pt"
        latest_path = ckpt_dir / "latest.pt"
        torch.save(payload, round_path)
        torch.save(payload, latest_path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        model_state = ckpt.get("best_model_state") or ckpt.get("model_state")
        if model_state is None:
            raise ValueError(f"Checkpoint at {path} missing model state.")

        self.model.load_state_dict(model_state)
        self.best_model_state = {k: v.detach().cpu().clone() for k, v in model_state.items()}
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "round_number" in ckpt:
            self.rounds_trained = int(ckpt["round_number"])
        training = ckpt.get("training") or {}
        self.promotions = int(training.get("promotions_total", self.promotions))
        replay = ckpt.get("replay") or {}
        print(
            f"Resumed from {path}: round={self.rounds_trained}, "
            f"saved_replay_size={int(replay.get('size', 0))}, promotions={self.promotions}"
        )

    def _train_step(self, batch: List[TrainingExample]) -> dict:
        x = torch.from_numpy(np.stack([b.obs for b in batch])).float().to(self.config.device)
        target_pi = torch.from_numpy(np.stack([b.policy for b in batch])).float().to(self.config.device)
        target_v = torch.from_numpy(np.array([b.value for b in batch], dtype=np.float32)).to(self.config.device)

        logits, value = self.model(x)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_v)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def run(self) -> None:
        for round_idx in range(self.rounds_trained, self.config.rounds):
            print(f"\nRound {round_idx + 1}/{self.config.rounds}")
            t0 = time.perf_counter()
            examples = self._collect_selfplay_examples(round_idx)
            self.buffer.extend(examples)
            t_selfplay = time.perf_counter() - t0

            if len(self.buffer) == 0:
                continue
            self.model.train()
            pre_step_model_state = self._model_state_cpu()
            pre_step_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            steps = max(1, int(self.config.train_steps_per_round))
            accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            for _ in range(steps):
                batch = self.buffer.sample(self.config.batch_size)
                s = self._train_step(batch)
                accum["loss"] += s["loss"]
                accum["policy_loss"] += s["policy_loss"]
                accum["value_loss"] += s["value_loss"]
            stats = {k: v / float(steps) for k, v in accum.items()}
            candidate_state = self._model_state_cpu()
            self.last_arena = self._arena_match(candidate_state, self.best_model_state, round_idx)
            promoted = self.last_arena.challenger_win_rate >= self.config.arena_promote_threshold
            if promoted:
                self.best_model_state = {k: v.clone() for k, v in candidate_state.items()}
                self.promotions += 1
            else:
                self.model.load_state_dict(pre_step_model_state)
                self.optimizer.load_state_dict(pre_step_optimizer_state)

            examples_count = len(examples)
            ex_per_sec = float(examples_count) / max(1e-6, t_selfplay)
            print(
                f"Self-play: {self.config.selfplay_games_per_round} games, "
                f"{examples_count} positions, {t_selfplay:.1f}s ({ex_per_sec:.1f} pos/s)"
            )
            print(
                f"Learning: loss={stats['loss']:.4f} "
                f"(policy={stats['policy_loss']:.4f}, value={stats['value_loss']:.4f}, steps={steps})"
            )
            print(
                f"Arena: new vs best over {self.last_arena.games} games -> "
                f"W/L/D {self.last_arena.challenger_wins}/{self.last_arena.incumbent_wins}/{self.last_arena.draws}, "
                f"win rate {self.last_arena.challenger_win_rate * 100.0:.1f}% "
                f"(target {self.config.arena_promote_threshold * 100.0:.1f}%)"
            )
            print(
                "Result: promoted new model." if promoted else
                "Result: kept previous best model (new one did not beat it)."
            )
            print(f"Progress: best model promotions so far = {self.promotions}")

            self.rounds_trained = round_idx + 1
            self._save_checkpoint(
                round_number=self.rounds_trained,
                promoted=promoted,
                train_stats=stats,
                selfplay_examples=examples_count,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero-style Satellites agent.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--rounds", type=int, default=None, help="Total training rounds.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to write checkpoints.")
    parser.add_argument("--train-steps", type=int, default=None, help="SGD updates per round.")
    parser.add_argument("--arena-games", type=int, default=None, help="Arena games per round.")
    parser.add_argument("--promote-threshold", type=float, default=None, help="Arena win-rate threshold to promote.")
    parser.add_argument("--heuristic-value-w", type=float, default=None, help="AlphaMCTS heuristic value blend [0..1].")
    parser.add_argument("--heuristic-action-w", type=float, default=None, help="AlphaMCTS heuristic action logit weight.")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.resume:
        cfg.resume_checkpoint = args.resume
    if args.rounds is not None:
        cfg.rounds = int(args.rounds)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.train_steps is not None:
        cfg.train_steps_per_round = int(args.train_steps)
    if args.arena_games is not None:
        cfg.arena_games_per_round = int(args.arena_games)
    if args.promote_threshold is not None:
        cfg.arena_promote_threshold = float(args.promote_threshold)
    if args.heuristic_value_w is not None:
        cfg.alpha_heuristic_value_weight = float(args.heuristic_value_w)
    if args.heuristic_action_w is not None:
        cfg.alpha_heuristic_action_weight = float(args.heuristic_action_w)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

