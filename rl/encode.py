from __future__ import annotations

from typing import Dict

import numpy as np

from engine import SatellitesGame, _sat_native


class FeatureEncoder:
    """Flat numeric encoder for policy/value training."""

    STATE_INDEX: Dict[str, int] = {
        "CHOOSE_SATELLITE": 0,
        "CHOOSE_DIRECTION": 1,
        "PERFORM_ACTIONS": 2,
        "GAME_OVER": 3,
    }
    SAT_TYPES = ("move_tank", "move_bot", "add_tank", "add_bot")
    SAT_CODE = {name: idx for idx, name in enumerate(SAT_TYPES)}

    def __init__(self, game_template: SatellitesGame | None = None):
        self.game_template = game_template or SatellitesGame(headless=True)
        self.num_cells = self.game_template.num_cells
        # p0_bot, p0_tank, p1_bot, p1_tank, artefact, p0_start, p1_start
        self.cell_feature_size = 7
        # side_to_move(2), scores(2), state(4), active_sat(7), counters(3), satellites(6*5)
        self.global_feature_size = 2 + 2 + 4 + 7 + 3 + 30
        self.feature_dim = self.num_cells * self.cell_feature_size + self.global_feature_size

    def encode(self, game: SatellitesGame) -> np.ndarray:
        if hasattr(game, "encode_features"):
            return np.asarray(game.encode_features(), dtype=np.float32)

        game._ensure_cache()
        if _sat_native is not None and hasattr(_sat_native, "encode_features"):
            active_idx = -1 if game.active_satellite_idx is None else int(game.active_satellite_idx)
            sat_type_codes = [self.SAT_CODE.get(sat["type"], 0) for sat in game.satellites]
            sat_charges = [int(sat["charges"]) for sat in game.satellites]
            vec = _sat_native.encode_features(
                game.unit_owner,
                game.unit_kind,
                game.unit_count,
                game.is_artefact_cell,
                game.is_p0_start_cell,
                game.is_p1_start_cell,
                int(game.turn),
                int(game.scores[0]),
                int(game.scores[1]),
                self._state_code(game.state),
                active_idx,
                int(game.actions_remaining),
                int(game.picked_up_charges),
                int(game.turn_count),
                int(max(1, game.MAX_TURNS)),
                sat_type_codes,
                sat_charges,
            )
            return np.asarray(vec, dtype=np.float32)

        feat = np.zeros(self.feature_dim, dtype=np.float32)
        p = 0

        for cid in range(self.num_cells):
            owner = game.unit_owner[cid]
            kind = game.unit_kind[cid]
            cnt = game.unit_count[cid] / 20.0
            if owner == 0 and kind == 1:
                feat[p + 0] = cnt
            elif owner == 0 and kind == 2:
                feat[p + 1] = cnt
            elif owner == 1 and kind == 1:
                feat[p + 2] = cnt
            elif owner == 1 and kind == 2:
                feat[p + 3] = cnt
            feat[p + 4] = 1.0 if game.is_artefact_cell[cid] else 0.0
            feat[p + 5] = 1.0 if game.is_p0_start_cell[cid] else 0.0
            feat[p + 6] = 1.0 if game.is_p1_start_cell[cid] else 0.0
            p += self.cell_feature_size

        # Side to move one-hot.
        feat[p + int(game.turn)] = 1.0
        p += 2

        # Scores.
        feat[p + 0] = float(game.scores[0]) / 9.0
        feat[p + 1] = float(game.scores[1]) / 9.0
        p += 2

        # Phase.
        phase_idx = self.STATE_INDEX.get(game.state, 0)
        feat[p + phase_idx] = 1.0
        p += 4

        # Active satellite (0..5), 6 means none.
        aidx = 6 if game.active_satellite_idx is None else int(game.active_satellite_idx)
        feat[p + aidx] = 1.0
        p += 7

        # Counters.
        feat[p + 0] = float(game.actions_remaining) / 3.0
        feat[p + 1] = float(game.picked_up_charges) / 3.0
        feat[p + 2] = float(game.turn_count) / float(max(1, game.MAX_TURNS))
        p += 3

        # Satellites: per slot one-hot type + charge.
        for sat in game.satellites:
            t = sat["type"]
            for i, sat_type in enumerate(self.SAT_TYPES):
                feat[p + i] = 1.0 if t == sat_type else 0.0
            feat[p + 4] = float(sat["charges"]) / 3.0
            p += 5

        return feat

    def _state_code(self, state: str) -> int:
        if state == "GAME_OVER":
            return 0
        if state == "CHOOSE_SATELLITE":
            return 1
        if state == "CHOOSE_DIRECTION":
            return 2
        if state == "PERFORM_ACTIONS":
            return 3
        return 255
