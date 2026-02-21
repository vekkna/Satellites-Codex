from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Tuple

import numpy as np
import torch

from engine import SatellitesGame
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder
from rl.model import SatellitesPolicyValueNet


@dataclass
class AlphaNode:
    player_to_move: int
    priors: Dict[int, float] = field(default_factory=dict)
    visit_count: Dict[int, int] = field(default_factory=dict)
    value_sum: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "AlphaNode"] = field(default_factory=dict)
    expanded: bool = False
    visits: int = 0

    def q(self, action_idx: int) -> float:
        n = self.visit_count.get(action_idx, 0)
        if n <= 0:
            return 0.0
        return self.value_sum.get(action_idx, 0.0) / float(n)

    def u(self, action_idx: int, c_puct: float) -> float:
        n = self.visit_count.get(action_idx, 0)
        p = self.priors.get(action_idx, 0.0)
        return c_puct * p * math.sqrt(max(1.0, float(self.visits))) / (1.0 + float(n))

    def best_action(self, c_puct: float) -> int:
        assert self.priors, "best_action called on empty node"
        return max(self.priors.keys(), key=lambda a: self.q(a) + self.u(a, c_puct))


class AlphaMCTS:
    """Minimal AlphaZero-style MCTS scaffolding with model priors/value."""

    def __init__(
        self,
        model: SatellitesPolicyValueNet,
        action_space: GlobalActionSpace,
        encoder: FeatureEncoder,
        *,
        simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.model = model
        self.action_space = action_space
        self.encoder = encoder
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = torch.device(device)
        self.rng = random.Random(seed)

    @torch.no_grad()
    def _policy_value(self, game: SatellitesGame) -> Tuple[np.ndarray, float]:
        x = torch.from_numpy(self.encoder.encode(game)).float().unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        return logits.squeeze(0).detach().cpu().numpy(), float(value.item())

    def _expand(self, node: AlphaNode, game: SatellitesGame, add_noise: bool = False) -> float:
        legal = self.action_space.legal_action_indices(game)
        if not legal:
            node.expanded = True
            node.priors = {}
            return 0.0

        logits, value = self._policy_value(game)
        legal_logits = logits[legal]
        legal_logits = legal_logits - np.max(legal_logits)
        probs = np.exp(legal_logits)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            probs = np.ones_like(probs) / float(len(probs))
        else:
            probs = probs / denom

        if add_noise and len(legal) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            probs = (1.0 - self.dirichlet_eps) * probs + self.dirichlet_eps * noise

        node.priors = {a: float(p) for a, p in zip(legal, probs)}
        node.visit_count = {a: 0 for a in legal}
        node.value_sum = {a: 0.0 for a in legal}
        node.expanded = True
        return value

    def _terminal_value_for_current_player(self, game: SatellitesGame) -> float:
        if game.winner is None or game.winner == -1:
            return 0.0
        return 1.0 if game.winner == game.turn else -1.0

    def search(self, root_game: SatellitesGame) -> Tuple[AlphaNode, np.ndarray]:
        root = AlphaNode(player_to_move=int(root_game.turn))
        self._expand(root, root_game, add_noise=True)

        for _ in range(self.simulations):
            game = root_game.clone()
            node = root
            path: list[tuple[AlphaNode, int]] = []

            while node.expanded and node.priors and game.state != "GAME_OVER":
                action_idx = node.best_action(self.c_puct)
                action = self.action_space.from_index(action_idx)
                ok = game.apply_action(action)
                if not ok:
                    node.priors[action_idx] = 0.0
                    continue
                path.append((node, action_idx))
                child = node.children.get(action_idx)
                if child is None:
                    child = AlphaNode(player_to_move=int(game.turn))
                    node.children[action_idx] = child
                    node = child
                    break
                node = child

            if game.state == "GAME_OVER":
                value = self._terminal_value_for_current_player(game)
            else:
                value = self._expand(node, game, add_noise=False)

            cur = value
            for parent, aidx in reversed(path):
                parent.visits += 1
                parent.visit_count[aidx] = parent.visit_count.get(aidx, 0) + 1
                parent.value_sum[aidx] = parent.value_sum.get(aidx, 0.0) + cur
                cur = -cur

        pi = self.action_space.visit_policy(root.visit_count, temperature=1.0)
        return root, pi

    def select_action(self, root_game: SatellitesGame, temperature: float = 1.0):
        root, _ = self.search(root_game)
        pi = self.action_space.visit_policy(root.visit_count, temperature=temperature)
        if pi.sum() <= 0:
            legal = root_game.legal_actions()
            if not legal:
                raise ValueError("No legal actions from root state.")
            action = legal[self.rng.randrange(len(legal))]
            return action, {"policy": pi, "root_visits": 0}
        action_idx = int(np.random.choice(np.arange(self.action_space.size), p=pi))
        action = self.action_space.from_index(action_idx)
        return action, {"policy": pi, "root_visits": int(sum(root.visit_count.values()))}

