from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from engine import SatellitesGame

Action = Any


class GlobalActionSpace:
    """Fixed action indexing for policy networks."""

    def __init__(self, game_template: SatellitesGame | None = None, max_move_amount: int = 20):
        self.game_template = game_template or SatellitesGame(headless=True)
        self.max_move_amount = max_move_amount
        self.index_to_action: List[Action] = []
        self.action_to_index: Dict[Action, int] = {}
        self._build()

    @property
    def size(self) -> int:
        return len(self.index_to_action)

    def _add(self, action: Action) -> None:
        idx = len(self.index_to_action)
        self.index_to_action.append(action)
        self.action_to_index[action] = idx

    def _build(self) -> None:
        # Satellite choice.
        for i in range(6):
            self._add(("select_satellite", i))
        # Direction choice.
        self._add(("set_direction", False))
        self._add(("set_direction", True))
        # Adds for every board cell.
        for coord in self.game_template.cell_id_to_coord:
            self._add(("add", coord[0], coord[1]))
        # Move actions for directed adjacent pairs with amount 1..max_move_amount.
        for src in self.game_template.cell_id_to_coord:
            src_id = self.game_template.coord_to_cell_id[src]
            for dst in self.game_template.neighbors_by_cell_id[src_id]:
                for amount in range(1, self.max_move_amount + 1):
                    self._add(("move", src, dst, amount))

    def to_index(self, action: Action) -> int:
        return self.action_to_index[action]

    def from_index(self, index: int) -> Action:
        return self.index_to_action[index]

    def legal_action_indices(self, game: SatellitesGame) -> List[int]:
        out: List[int] = []
        for action in game.legal_actions():
            idx = self.action_to_index.get(action)
            if idx is not None:
                out.append(idx)
        return out

    def legal_action_mask(self, game: SatellitesGame) -> np.ndarray:
        mask = np.zeros(self.size, dtype=np.bool_)
        for idx in self.legal_action_indices(game):
            mask[idx] = True
        return mask

    def visit_policy(self, visit_counts: Dict[int, int], temperature: float = 1.0) -> np.ndarray:
        pi = np.zeros(self.size, dtype=np.float32)
        if not visit_counts:
            return pi
        t = max(1e-6, float(temperature))
        keys = list(visit_counts.keys())
        values = np.array([float(visit_counts[k]) for k in keys], dtype=np.float32)
        if t < 1e-3:
            best = keys[int(np.argmax(values))]
            pi[best] = 1.0
            return pi
        values = np.power(values, 1.0 / t)
        total = float(values.sum())
        if total <= 0.0:
            return pi
        for key, v in zip(keys, values):
            pi[key] = float(v / total)
        return pi

