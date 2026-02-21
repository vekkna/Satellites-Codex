from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import random
from typing import Deque, List

import numpy as np
import torch
import torch.nn.functional as F

from agents.alpha_mcts import AlphaMCTS
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder
from rl.model import SatellitesPolicyValueNet
from rl.selfplay import TrainingExample, run_selfplay_game


@dataclass
class TrainConfig:
    selfplay_games_per_round: int = 4
    selfplay_workers: int = 1
    rounds: int = 10
    simulations: int = 64
    batch_size: int = 64
    replay_size: int = 20000
    lr: float = 1e-3
    device: str = "cpu"


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


def _run_selfplay_worker(
    model_state: dict,
    simulations: int,
    games: int,
    seed: int,
) -> List[TrainingExample]:
    torch.set_num_threads(1)
    action_space = GlobalActionSpace()
    encoder = FeatureEncoder()
    model = SatellitesPolicyValueNet(encoder.feature_dim, action_space.size).to("cpu")
    model.load_state_dict(model_state)
    model.eval()
    mcts = AlphaMCTS(
        model,
        action_space,
        encoder,
        simulations=simulations,
        device="cpu",
        seed=seed,
    )
    out: List[TrainingExample] = []
    for _ in range(games):
        out.extend(run_selfplay_game(mcts, encoder))
    return out


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.action_space = GlobalActionSpace()
        self.encoder = FeatureEncoder()
        self.model = SatellitesPolicyValueNet(self.encoder.feature_dim, self.action_space.size).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.replay_size)

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
        for round_idx in range(self.config.rounds):
            mcts = AlphaMCTS(
                self.model,
                self.action_space,
                self.encoder,
                simulations=self.config.simulations,
                device=self.config.device,
            )
            if self.config.selfplay_workers <= 1:
                for _ in range(self.config.selfplay_games_per_round):
                    examples = run_selfplay_game(mcts, self.encoder)
                    self.buffer.extend(examples)
            else:
                workers = min(self.config.selfplay_workers, self.config.selfplay_games_per_round)
                base = self.config.selfplay_games_per_round // workers
                rem = self.config.selfplay_games_per_round % workers
                games_per_worker = [base + (1 if i < rem else 0) for i in range(workers)]
                model_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = []
                    for i, games in enumerate(games_per_worker):
                        if games <= 0:
                            continue
                        futures.append(
                            ex.submit(
                                _run_selfplay_worker,
                                model_state,
                                self.config.simulations,
                                games,
                                1000 * round_idx + i + 1,
                            )
                        )
                    for fut in as_completed(futures):
                        self.buffer.extend(fut.result())

            if len(self.buffer) == 0:
                continue
            batch = self.buffer.sample(self.config.batch_size)
            stats = self._train_step(batch)
            print(
                f"round={round_idx + 1} buffer={len(self.buffer)} "
                f"loss={stats['loss']:.4f} p={stats['policy_loss']:.4f} v={stats['value_loss']:.4f}"
            )


def main() -> None:
    trainer = Trainer(TrainConfig())
    trainer.run()


if __name__ == "__main__":
    main()
