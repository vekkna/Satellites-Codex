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
        heuristic_value_weight: float = 0.0,
        heuristic_action_weight: float = 0.0,
        heuristic_score_weight: float = 1.0,
        heuristic_tank_weight: float = 0.18,
        heuristic_bot_weight: float = 0.12,
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
        self.heuristic_value_weight = max(0.0, min(1.0, float(heuristic_value_weight)))
        self.heuristic_action_weight = max(0.0, float(heuristic_action_weight))
        self.heuristic_score_weight = float(heuristic_score_weight)
        self.heuristic_tank_weight = float(heuristic_tank_weight)
        self.heuristic_bot_weight = float(heuristic_bot_weight)
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
        if self.heuristic_action_weight > 0.0:
            bonuses = np.array(
                [self._heuristic_action_bonus(game, self.action_space.from_index(a)) for a in legal],
                dtype=np.float32,
            )
            logits = logits.copy()
            logits[legal] = logits[legal] + self.heuristic_action_weight * bonuses
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
        if self.heuristic_value_weight > 0.0:
            h = self._heuristic_state_value(game, int(game.turn))
            value = (1.0 - self.heuristic_value_weight) * value + self.heuristic_value_weight * h
        return value

    def _owner_counts(self, game: SatellitesGame, owner: int) -> tuple[int, int]:
        game._ensure_cache()
        tank_count = 0
        bot_count = 0
        for cid in game.owner_tank_cells[owner]:
            tank_count += int(game.unit_count[cid])
        for cid in game.owner_bot_cells[owner]:
            bot_count += int(game.unit_count[cid])
        return tank_count, bot_count

    def _heuristic_state_value(self, game: SatellitesGame, player: int) -> float:
        enemy = 1 - player
        score_diff = float(game.scores[player] - game.scores[enemy])
        my_tanks, my_bots = self._owner_counts(game, player)
        en_tanks, en_bots = self._owner_counts(game, enemy)
        tank_diff = float(my_tanks - en_tanks)
        bot_diff = float(my_bots - en_bots)
        raw = (
            self.heuristic_score_weight * score_diff
            + self.heuristic_tank_weight * tank_diff
            + self.heuristic_bot_weight * bot_diff
        )
        return float(math.tanh(raw / 3.0))

    def _heuristic_action_bonus(self, game: SatellitesGame, action) -> float:
        kind = action[0]
        if kind == "move":
            src, dst, amount = action[1], action[2], int(action[3])
            u = game.grid.get(src)
            if not u:
                return 0.0
            if u["type"] == "bot":
                bonus = 0.0
                if dst in game.artefacts:
                    bonus += 3.0 + 0.15 * float(amount)
                old_d = min((game.get_hex_distance(src, a) for a in game.artefacts), default=99)
                new_d = min((game.get_hex_distance(dst, a) for a in game.artefacts), default=99)
                bonus += 0.25 * float(old_d - new_d)
                return bonus
            if u["type"] == "tank":
                return 0.05 * float(amount)
        return 0.0

    def _terminal_value_for_current_player(self, game: SatellitesGame) -> float:
        if game.winner is None or game.winner == -1:
            return 0.0
        return 1.0 if game.winner == game.turn else -1.0

    def search(self, root_game: SatellitesGame, *, add_root_noise: bool = True) -> Tuple[AlphaNode, np.ndarray]:
        root = AlphaNode(player_to_move=int(root_game.turn))
        self._expand(root, root_game, add_noise=add_root_noise)

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

    def select_action(
        self,
        root_game: SatellitesGame,
        temperature: float = 1.0,
        *,
        add_root_noise: bool = True,
    ):
        root, _ = self.search(root_game, add_root_noise=add_root_noise)
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

