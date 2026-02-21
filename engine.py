import random
from pathlib import Path
import os
import sys

_NATIVE_ACTION_INDEX = os.environ.get("SAT_USE_NATIVE_ACTION_INDEX", "0") == "1"

try:
    import satellites_native as _sat_native
except Exception:
    _sat_native = None
    _native_root = Path(__file__).resolve().parent / "native" / "satellites_native" / "target"
    for _build in ("release", "debug"):
        _native_dir = _native_root / _build
        if not _native_dir.exists():
            continue
        for _p in (_native_dir, _native_dir / "deps"):
            _ps = str(_p)
            if _ps not in sys.path:
                sys.path.insert(0, _ps)
        try:
            import satellites_native as _sat_native
            break
        except Exception:
            _sat_native = None

# ==========================================
# PART 1: GAME LOGIC (Headless Engine)
# ==========================================

class SatellitesGame:
    def __init__(self, headless=False):
        self.headless = headless
        
        # Board Setup
        # Rows 0-8. Widths: 8, 9, 10, 11, 12, 11, 10, 9, 8
        self.row_widths = [8, 9, 10, 11, 12, 11, 10, 9, 8]
        (
            self.cell_id_to_coord,
            self.coord_to_cell_id,
            self.neighbors_by_cell_id,
        ) = self._build_topology()
        self.neighbor_ids_by_cell_id = tuple(
            tuple(self.coord_to_cell_id[coord] for coord in neighbors)
            for neighbors in self.neighbors_by_cell_id
        )
        self.edge_ordinal_by_cells = {}
        self.edge_by_ordinal = []
        edge_ordinal = 0
        for sid, nids in enumerate(self.neighbor_ids_by_cell_id):
            for eid in nids:
                self.edge_ordinal_by_cells[(sid, eid)] = edge_ordinal
                self.edge_by_ordinal.append((sid, eid))
                edge_ordinal += 1
        self.num_directed_edges = edge_ordinal
        self.num_cells = len(self.cell_id_to_coord)
        self.distance_by_cell_id = self._build_distance_matrix()
        self._grid = {}
        self.grid = {} # Key: (row, col), Value: {'owner': 0/1, 'type': 'tank'/'bot', 'count': int}
        self._cache_dirty = True
        self.unit_owner = [-1] * self.num_cells
        self.unit_kind = [0] * self.num_cells  # 0 empty, 1 bot, 2 tank
        self.unit_count = [0] * self.num_cells
        self.owner_total_units = [0, 0]
        self.owner_bot_cells = [set(), set()]
        self.owner_tank_cells = [set(), set()]
        self.is_artefact_cell = [False] * self.num_cells
        self.is_p0_start_cell = [False] * self.num_cells
        self.is_p1_start_cell = [False] * self.num_cells
        
        # Artefacts
        self.artefacts = [(2,1), (2,8), (4,4), (4,7), (6,1), (6,8)]
        for coord in self.artefacts:
            self.is_artefact_cell[self.coord_to_cell_id[coord]] = True
        for coord in ((0,3), (0,4)):
            self.is_p0_start_cell[self.coord_to_cell_id[coord]] = True
        for coord in ((8,3), (8,4)):
            self.is_p1_start_cell[self.coord_to_cell_id[coord]] = True
        
        # Players: 0 (Red), 1 (Blue)
        # Starting units
        self.add_unit(0, 3, 0, 'bot', 2)
        self.add_unit(0, 4, 0, 'tank', 2)
        
        self.add_unit(8, 3, 1, 'bot', 2)
        self.add_unit(8, 4, 1, 'tank', 2)
        self._ensure_cache()

        # Satellites
        self.satellites = [
            {'type': 'move_tank', 'charges': 2, 'name': 'Move Tank'},
            {'type': 'move_tank', 'charges': 2, 'name': 'Move Tank'},
            {'type': 'move_bot',  'charges': 2, 'name': 'Move Bot'},
            {'type': 'move_bot',  'charges': 2, 'name': 'Move Bot'},
            {'type': 'add_tank',  'charges': 0, 'name': 'Add Tank'},
            {'type': 'add_bot',   'charges': 0, 'name': 'Add Bot'},
        ]
        random.shuffle(self.satellites)
        
        self.scores = [0, 0]
        self.turn = 0 # Player 0 starts
        
        # Turn State Machine
        self.state = "CHOOSE_SATELLITE" 
        
        self.active_satellite_idx = None
        self.actions_remaining = 0
        self.picked_up_charges = 0
        self.action_type = None
        
        self.selected_hex = None 
        
        # Move Quantity Logic
        self.pending_move_dest = None 
        self.pending_move_max = 0
        self.move_amount_selection = 1

        self.info_message = ""
        if not self.headless:
             self.info_message = "Player 1's Turn: Choose a Satellite"
        self.winner = None
        
        # FIX: Turn Limit
        self.turn_count = 1
        self.MAX_TURNS = 100

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value
        self._cache_dirty = True

    def _ensure_cache(self):
        if not self._cache_dirty:
            return
        self.unit_owner = [-1] * self.num_cells
        self.unit_kind = [0] * self.num_cells
        self.unit_count = [0] * self.num_cells
        self.owner_total_units = [0, 0]
        self.owner_bot_cells = [set(), set()]
        self.owner_tank_cells = [set(), set()]
        for (r, c), u in self.grid.items():
            cid = self.coord_to_cell_id[(r, c)]
            owner = u['owner']
            kind = 2 if u['type'] == 'tank' else 1
            count = u['count']
            self.unit_owner[cid] = owner
            self.unit_kind[cid] = kind
            self.unit_count[cid] = count
            self.owner_total_units[owner] += count
            if kind == 2:
                self.owner_tank_cells[owner].add(cid)
            else:
                self.owner_bot_cells[owner].add(cid)
        self._cache_dirty = False

    def add_unit(self, r, c, owner, u_type, count):
        if (r, c) not in self.grid:
            self.grid[(r, c)] = {'owner': owner, 'type': u_type, 'count': 0}
        self.grid[(r, c)]['count'] += count
        self._cache_dirty = True

    def clone(self):
        """Fast, engine-aware clone used by search code."""
        new = self.__class__.__new__(self.__class__)

        # Immutable/static topology can be shared.
        new.headless = self.headless
        new.row_widths = self.row_widths
        new.cell_id_to_coord = self.cell_id_to_coord
        new.coord_to_cell_id = self.coord_to_cell_id
        new.neighbors_by_cell_id = self.neighbors_by_cell_id
        new.neighbor_ids_by_cell_id = self.neighbor_ids_by_cell_id
        new.edge_ordinal_by_cells = self.edge_ordinal_by_cells
        new.edge_by_ordinal = self.edge_by_ordinal
        new.num_directed_edges = self.num_directed_edges
        new.num_cells = self.num_cells
        new.distance_by_cell_id = self.distance_by_cell_id

        # Mutable game state.
        new._grid = {k: v.copy() for k, v in self._grid.items()}
        new._cache_dirty = self._cache_dirty
        new.unit_owner = self.unit_owner.copy()
        new.unit_kind = self.unit_kind.copy()
        new.unit_count = self.unit_count.copy()
        new.owner_total_units = self.owner_total_units.copy()
        new.owner_bot_cells = [self.owner_bot_cells[0].copy(), self.owner_bot_cells[1].copy()]
        new.owner_tank_cells = [self.owner_tank_cells[0].copy(), self.owner_tank_cells[1].copy()]
        new.is_artefact_cell = self.is_artefact_cell.copy()
        new.is_p0_start_cell = self.is_p0_start_cell.copy()
        new.is_p1_start_cell = self.is_p1_start_cell.copy()

        new.artefacts = self.artefacts.copy()
        new.satellites = [sat.copy() for sat in self.satellites]
        new.scores = self.scores.copy()
        new.turn = self.turn
        new.state = self.state
        new.active_satellite_idx = self.active_satellite_idx
        new.actions_remaining = self.actions_remaining
        new.picked_up_charges = self.picked_up_charges
        new.action_type = self.action_type
        new.selected_hex = self.selected_hex
        new.pending_move_dest = self.pending_move_dest
        new.pending_move_max = self.pending_move_max
        new.move_amount_selection = self.move_amount_selection
        new.info_message = self.info_message
        new.winner = self.winner
        new.turn_count = self.turn_count
        new.MAX_TURNS = self.MAX_TURNS
        if hasattr(self, 'distribution_direction'):
            new.distribution_direction = self.distribution_direction
        return new

    def _capture_undo_token_for_action(self, action):
        kind = action[0]
        changed_cells = {}
        if kind == 'add':
            coords = [(action[1], action[2])]
        elif kind == 'move':
            coords = [action[1], action[2]]
        else:
            coords = []

        for coord in coords:
            if coord in changed_cells:
                continue
            cell = self._grid.get(coord)
            changed_cells[coord] = None if cell is None else cell.copy()

        return {
            "_grid_cells": changed_cells,
            "artefacts": self.artefacts.copy(),
            "is_artefact_cell": self.is_artefact_cell.copy(),
            "satellites": [sat.copy() for sat in self.satellites],
            "scores": self.scores.copy(),
            "turn": self.turn,
            "state": self.state,
            "active_satellite_idx": self.active_satellite_idx,
            "actions_remaining": self.actions_remaining,
            "picked_up_charges": self.picked_up_charges,
            "action_type": self.action_type,
            "selected_hex": self.selected_hex,
            "pending_move_dest": self.pending_move_dest,
            "pending_move_max": self.pending_move_max,
            "move_amount_selection": self.move_amount_selection,
            "info_message": self.info_message,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "MAX_TURNS": self.MAX_TURNS,
            "distribution_direction": getattr(self, "distribution_direction", None),
        }

    def undo_action(self, token):
        for coord, cell in token["_grid_cells"].items():
            if cell is None:
                self._grid.pop(coord, None)
            else:
                self._grid[coord] = cell
        self.artefacts = token["artefacts"]
        self.is_artefact_cell = token["is_artefact_cell"]
        self.satellites = token["satellites"]
        self._cache_dirty = True
        self.scores = token["scores"]
        self.turn = token["turn"]
        self.state = token["state"]
        self.active_satellite_idx = token["active_satellite_idx"]
        self.actions_remaining = token["actions_remaining"]
        self.picked_up_charges = token["picked_up_charges"]
        self.action_type = token["action_type"]
        self.selected_hex = token["selected_hex"]
        self.pending_move_dest = token["pending_move_dest"]
        self.pending_move_max = token["pending_move_max"]
        self.move_amount_selection = token["move_amount_selection"]
        self.info_message = token["info_message"]
        self.winner = token["winner"]
        self.turn_count = token["turn_count"]
        self.MAX_TURNS = token["MAX_TURNS"]
        if token["distribution_direction"] is not None:
            self.distribution_direction = token["distribution_direction"]
        elif hasattr(self, "distribution_direction"):
            delattr(self, "distribution_direction")

    def apply_action_with_undo(self, action):
        """Apply an action and return (success, token, aux).

        Supported action formats:
        - ('select_satellite', idx)
        - ('set_direction', clockwise_bool)
        - ('add', r, c)
        - ('move', (r0, c0), (r1, c1), amount)
        """
        kind = action[0]
        if kind == 'select_satellite':
            token = self._capture_undo_token_for_action(action)
            old_state = self.state
            old_active = self.active_satellite_idx
            self.select_satellite(action[1])
            success = (self.state != old_state) or (self.active_satellite_idx != old_active)
            return success, token, None
        if kind == 'set_direction':
            token = self._capture_undo_token_for_action(action)
            old_state = self.state
            self.set_distribution_direction(action[1])
            success = self.state != old_state or old_state == "CHOOSE_DIRECTION"
            return success, token, None
        if kind == 'add':
            token = self._capture_undo_token_for_action(action)
            success = self.execute_add(action[1], action[2])
            return success, token, None
        if kind == 'move':
            token = self._capture_undo_token_for_action(action)
            success, kills, score_gain = self.execute_move(action[1], action[2], action[3])
            return success, token, (kills, score_gain)
        raise ValueError(f"Unsupported action kind: {kind}")

    def apply_action(self, action):
        """Apply an action without keeping undo token."""
        kind = action[0]
        if kind == 'select_satellite':
            old_state = self.state
            old_active = self.active_satellite_idx
            self.select_satellite(action[1])
            return (self.state != old_state) or (self.active_satellite_idx != old_active)
        if kind == 'set_direction':
            old_state = self.state
            self.set_distribution_direction(action[1])
            return self.state != old_state or old_state == "CHOOSE_DIRECTION"
        if kind == 'add':
            return self.execute_add(action[1], action[2])
        if kind == 'move':
            success, _, _ = self.execute_move(action[1], action[2], action[3])
            return success
        raise ValueError(f"Unsupported action kind: {kind}")

    def _is_legal_add(self, r, c):
        if self.state != "PERFORM_ACTIONS" or "add" not in (self.action_type or ""):
            return False
        self._ensure_cache()
        if self.get_player_unit_count(self.turn) >= 20:
            return False

        unit_type = 'tank' if 'tank' in self.action_type else 'bot'
        cid = self.coord_to_cell_id.get((r, c))
        if cid is None:
            return False
        occ_owner = self.unit_owner[cid]
        occ_kind = self.unit_kind[cid]
        current = self.grid.get((r, c))

        if unit_type == 'tank':
            is_own_tank_stack = (
                occ_owner == self.turn and
                occ_kind == 2
            )
            if occ_owner != -1 and not is_own_tank_stack:
                return False
            opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
            if (r, c) in opp_starts:
                return False
            if (r, c) in self.artefacts:
                return False
            return True

        is_own_stack = (current and current['owner'] == self.turn and current['type'] == unit_type)
        valid_starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
        is_start_zone = ((r, c) in valid_starts) and (not current or is_own_stack)
        return bool(is_own_stack or is_start_zone)

    def _is_legal_move(self, start, end, amount):
        if self.state != "PERFORM_ACTIONS" or "move" not in (self.action_type or ""):
            return False
        self._ensure_cache()
        sid = self.coord_to_cell_id.get(start)
        eid = self.coord_to_cell_id.get(end)
        if sid is None or eid is None:
            return False
        if self.unit_owner[sid] != self.turn:
            return False
        if amount < 1 or amount > self.unit_count[sid]:
            return False

        req_type = 'tank' if 'tank' in self.action_type else 'bot'
        src_kind = 'tank' if self.unit_kind[sid] == 2 else ('bot' if self.unit_kind[sid] == 1 else None)
        if src_kind != req_type:
            return False

        if end not in self.get_hex_neighbors(start[0], start[1]):
            return False

        opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
        if end in opp_starts:
            return False

        move_type = src_kind
        if move_type == 'tank' and end in self.artefacts:
            return False

        if self.unit_owner[eid] == -1:
            return True
        if self.unit_owner[eid] == self.turn:
            target_type = 'tank' if self.unit_kind[eid] == 2 else 'bot'
            return target_type == move_type
        if move_type == 'bot':
            return False
        if move_type == 'tank' and self.unit_kind[eid] == 2 and self.unit_count[eid] >= amount:
            return False
        return True

    def legal_actions(self):
        if self.state == "GAME_OVER":
            return []

        if self.state == "CHOOSE_SATELLITE":
            actions = []
            for i, sat in enumerate(self.satellites):
                if sat['charges'] > 0:
                    actions.append(('select_satellite', i))
            return actions

        if self.state == "CHOOSE_DIRECTION":
            return [('set_direction', False), ('set_direction', True)]

        if self.state != "PERFORM_ACTIONS":
            return []

        actions = []
        if "add" in (self.action_type or ""):
            self._ensure_cache()
            if self.owner_total_units[self.turn] >= 20:
                return []
            unit_type = 'tank' if 'tank' in self.action_type else 'bot'
            if unit_type == 'tank':
                opp_start_mask = self.is_p1_start_cell if self.turn == 0 else self.is_p0_start_cell
                for cid, (r, c) in enumerate(self.cell_id_to_coord):
                    occ_owner = self.unit_owner[cid]
                    occ_kind = self.unit_kind[cid]
                    # Tanks: empty non-opponent-start non-artefact, or own tank stack.
                    if occ_owner == -1:
                        if (not opp_start_mask[cid]) and (not self.is_artefact_cell[cid]):
                            actions.append(('add', r, c))
                    elif occ_owner == self.turn and occ_kind == 2:
                        actions.append(('add', r, c))
            else:
                start_a, start_b = ((0, 3), (0, 4)) if self.turn == 0 else ((8, 3), (8, 4))
                own_bot_cids = self.owner_bot_cells[self.turn]
                for cid in own_bot_cids:
                    r, c = self.cell_id_to_coord[cid]
                    actions.append(('add', r, c))
                for pos in (start_a, start_b):
                    cid = self.coord_to_cell_id[pos]
                    if self.unit_owner[cid] == -1:
                        actions.append(('add', pos[0], pos[1]))
            return actions

        if "move" in (self.action_type or ""):
            self._ensure_cache()
            req_type = 'tank' if 'tank' in self.action_type else 'bot'
            compact = self._generate_compact_moves(req_type)
            for sid, eid, amount in compact:
                actions.append(
                    (
                        'move',
                        self.cell_id_to_coord[sid],
                        self.cell_id_to_coord[eid],
                        int(amount),
                    )
                )
            return actions

        return actions

    def legal_action_indices(self, max_move_amount=20):
        if max_move_amount < 1 or self.state == "GAME_OVER":
            return []
        if self.state == "PERFORM_ACTIONS":
            self._ensure_cache()

        if _NATIVE_ACTION_INDEX and _sat_native is not None and hasattr(_sat_native, "generate_legal_action_indices"):
            return list(
                _sat_native.generate_legal_action_indices(
                    self._state_code(),
                    self._action_type_code(),
                    int(self.turn),
                    [int(sat.get("charges", 0)) for sat in self.satellites],
                    self.unit_owner,
                    self.unit_kind,
                    self.unit_count,
                    self.is_artefact_cell,
                    self.is_p0_start_cell,
                    self.is_p1_start_cell,
                    int(min(255, max_move_amount)),
                )
            )

        if self.state == "CHOOSE_SATELLITE":
            out = []
            for i, sat in enumerate(self.satellites):
                if sat['charges'] > 0:
                    out.append(i)
            return out

        if self.state == "CHOOSE_DIRECTION":
            return [6, 7]

        if self.state != "PERFORM_ACTIONS":
            return []

        out = []
        add_base = 8
        move_base = add_base + self.num_cells

        if "add" in (self.action_type or ""):
            self._ensure_cache()
            if self.owner_total_units[self.turn] >= 20:
                return []
            unit_type = 'tank' if 'tank' in self.action_type else 'bot'
            if unit_type == 'tank':
                opp_start_mask = self.is_p1_start_cell if self.turn == 0 else self.is_p0_start_cell
                for cid in range(self.num_cells):
                    occ_owner = self.unit_owner[cid]
                    occ_kind = self.unit_kind[cid]
                    if occ_owner == -1:
                        if (not opp_start_mask[cid]) and (not self.is_artefact_cell[cid]):
                            out.append(add_base + cid)
                    elif occ_owner == self.turn and occ_kind == 2:
                        out.append(add_base + cid)
            else:
                start_a, start_b = ((0, 3), (0, 4)) if self.turn == 0 else ((8, 3), (8, 4))
                for cid in self.owner_bot_cells[self.turn]:
                    out.append(add_base + cid)
                for pos in (start_a, start_b):
                    cid = self.coord_to_cell_id[pos]
                    if self.unit_owner[cid] == -1:
                        out.append(add_base + cid)
            return out

        if "move" in (self.action_type or ""):
            self._ensure_cache()
            req_type = 'tank' if 'tank' in self.action_type else 'bot'
            compact = self._generate_compact_moves(req_type)
            for sid, eid, amount in compact:
                amount_i = int(amount)
                if amount_i > max_move_amount:
                    continue
                edge_ordinal = self.edge_ordinal_by_cells[(sid, eid)]
                out.append(move_base + edge_ordinal * max_move_amount + (amount_i - 1))
            return out

        return out

    def apply_action_index(self, action_index, max_move_amount=20):
        if max_move_amount < 1:
            return False
        idx = int(action_index)
        add_base = 8
        move_base = add_base + self.num_cells
        move_span = self.num_directed_edges * max_move_amount

        if 0 <= idx < 6:
            old_state = self.state
            old_active = self.active_satellite_idx
            self.select_satellite(idx)
            return (self.state != old_state) or (self.active_satellite_idx != old_active)
        if idx == 6:
            old_state = self.state
            self.set_distribution_direction(False)
            return self.state != old_state or old_state == "CHOOSE_DIRECTION"
        if idx == 7:
            old_state = self.state
            self.set_distribution_direction(True)
            return self.state != old_state or old_state == "CHOOSE_DIRECTION"
        if add_base <= idx < move_base:
            cid = idx - add_base
            if cid < 0 or cid >= self.num_cells:
                return False
            r, c = self.cell_id_to_coord[cid]
            return self.execute_add(r, c)
        if move_base <= idx < (move_base + move_span):
            rel = idx - move_base
            edge_ordinal = rel // max_move_amount
            amount = (rel % max_move_amount) + 1
            if edge_ordinal < 0 or edge_ordinal >= len(self.edge_by_ordinal):
                return False
            sid, eid = self.edge_by_ordinal[edge_ordinal]
            start = self.cell_id_to_coord[sid]
            end = self.cell_id_to_coord[eid]
            success, _, _ = self.execute_move(start, end, amount)
            return success
        return False

    def _generate_compact_moves(self, req_type):
        req_kind = 2 if req_type == 'tank' else 1
        if _sat_native is not None and hasattr(_sat_native, "generate_move_actions"):
            return _sat_native.generate_move_actions(
                self.unit_owner,
                self.unit_kind,
                self.unit_count,
                int(self.turn),
                req_kind,
                self.is_artefact_cell,
            )

        compact = []
        source_cells = self.owner_tank_cells[self.turn] if req_type == 'tank' else self.owner_bot_cells[self.turn]
        opp_start_a, opp_start_b = ((8, 3), (8, 4)) if self.turn == 0 else ((0, 3), (0, 4))
        opp_start_set = {self.coord_to_cell_id[opp_start_a], self.coord_to_cell_id[opp_start_b]}
        for sid in source_cells:
            src_count = self.unit_count[sid]
            if src_count <= 0:
                continue
            for eid in self.neighbor_ids_by_cell_id[sid]:
                if eid in opp_start_set:
                    continue
                target_owner = self.unit_owner[eid]
                target_kind = self.unit_kind[eid]  # 0 empty, 1 bot, 2 tank
                target_count = self.unit_count[eid]

                if req_type == 'tank' and self.is_artefact_cell[eid]:
                    continue

                if target_owner == -1:
                    for amount in range(1, src_count + 1):
                        compact.append((sid, eid, amount))
                    continue

                if target_owner == self.turn:
                    if (req_type == 'tank' and target_kind == 2) or (req_type == 'bot' and target_kind == 1):
                        for amount in range(1, src_count + 1):
                            compact.append((sid, eid, amount))
                    continue

                if req_type == 'bot':
                    continue
                if target_kind == 1:
                    for amount in range(1, src_count + 1):
                        compact.append((sid, eid, amount))
                    continue
                if target_kind == 2 and src_count > target_count:
                    for amount in range(target_count + 1, src_count + 1):
                        compact.append((sid, eid, amount))
        return compact

    def _state_code(self):
        if self.state == "GAME_OVER":
            return 0
        if self.state == "CHOOSE_SATELLITE":
            return 1
        if self.state == "CHOOSE_DIRECTION":
            return 2
        if self.state == "PERFORM_ACTIONS":
            return 3
        return 255

    def _action_type_code(self):
        at = self.action_type or ""
        if at == "add_tank":
            return 1
        if at == "add_bot":
            return 2
        if at == "move_tank":
            return 3
        if at == "move_bot":
            return 4
        return 0

    def to_native_state(self):
        self._ensure_cache()
        if _sat_native is None or not hasattr(_sat_native, "NativeSatGame"):
            return None
        sat_code = {
            "move_tank": 0,
            "move_bot": 1,
            "add_tank": 2,
            "add_bot": 3,
        }
        sat_type_codes = [sat_code.get(s["type"], 0) for s in self.satellites]
        sat_charges = [int(s.get("charges", 0)) for s in self.satellites]
        active_idx = -1 if self.active_satellite_idx is None else int(self.active_satellite_idx)
        winner = -1 if self.winner is None else int(self.winner)
        return _sat_native.NativeSatGame(
            self.unit_owner,
            self.unit_kind,
            self.unit_count,
            self.is_artefact_cell,
            self.is_p0_start_cell,
            self.is_p1_start_cell,
            sat_type_codes,
            sat_charges,
            int(self.turn),
            int(self.scores[0]),
            int(self.scores[1]),
            self._state_code(),
            active_idx,
            int(self.actions_remaining),
            int(self.picked_up_charges),
            self._action_type_code(),
            int(self.turn_count),
            int(max(1, self.MAX_TURNS)),
            winner,
        )

    def get_player_unit_count(self, owner):
        self._ensure_cache()
        return self.owner_total_units[owner]

    def _build_topology(self):
        cell_id_to_coord = []
        coord_to_cell_id = {}
        for r in range(9):
            for c in range(self.row_widths[r]):
                cid = len(cell_id_to_coord)
                coord = (r, c)
                cell_id_to_coord.append(coord)
                coord_to_cell_id[coord] = cid

        neighbors_by_cell_id = []
        for r, c in cell_id_to_coord:
            directions = [
                (r, c - 1),
                (r, c + 1),
            ]
            if r > 0:
                if r <= 4:
                    directions.append((r - 1, c - 1))
                    directions.append((r - 1, c))
                else:
                    directions.append((r - 1, c))
                    directions.append((r - 1, c + 1))
            if r < 8:
                if r < 4:
                    directions.append((r + 1, c))
                    directions.append((r + 1, c + 1))
                else:
                    directions.append((r + 1, c - 1))
                    directions.append((r + 1, c))

            valid = []
            for nr, nc in directions:
                if 0 <= nr < 9 and 0 <= nc < self.row_widths[nr]:
                    valid.append((nr, nc))
            neighbors_by_cell_id.append(tuple(valid))

        return tuple(cell_id_to_coord), coord_to_cell_id, tuple(neighbors_by_cell_id)

    def _build_distance_matrix(self):
        distance = [[-1] * self.num_cells for _ in range(self.num_cells)]
        for src in range(self.num_cells):
            queue = [src]
            distance[src][src] = 0
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                base_d = distance[src][cur]
                for nr, nc in self.neighbors_by_cell_id[cur]:
                    nxt = self.coord_to_cell_id[(nr, nc)]
                    if distance[src][nxt] != -1:
                        continue
                    distance[src][nxt] = base_d + 1
                    queue.append(nxt)
        return tuple(tuple(row) for row in distance)

    def get_hex_neighbors(self, r, c):
        cell_id = self.coord_to_cell_id.get((r, c))
        if cell_id is None:
            return []
        return list(self.neighbors_by_cell_id[cell_id])

    def get_hex_distance(self, a, b):
        a_id = self.coord_to_cell_id.get(a)
        b_id = self.coord_to_cell_id.get(b)
        if a_id is None or b_id is None:
            return -1
        return self.distance_by_cell_id[a_id][b_id]

    def check_actions_still_possible(self):
        """Checks if any valid moves remain for the current action type. If not, auto-end turn."""
        if not self.action_type:
            self.end_turn()
            return
        self._ensure_cache()

        req_type = 'tank' if 'tank' in self.action_type else 'bot'
        
        can_act = False
        
        # 1. ADD VALID?
        if "add" in self.action_type:
            # Check Cap
            if self.get_player_unit_count(self.turn) >= 20:
                can_act = False
            else:
                # Check Placement Locations
                if req_type == 'tank':
                    # Tanks can drop on own tank stacks, or empty non-opponent-start hexes.
                    opp_start_mask = self.is_p1_start_cell if self.turn == 0 else self.is_p0_start_cell

                    # Scan grid for any valid empty spot
                    for cid in range(self.num_cells):
                        if self.unit_owner[cid] == -1 and not opp_start_mask[cid] and not self.is_artefact_cell[cid]:
                            can_act = True
                            break
                    # Or any existing own tank stack
                    if not can_act:
                        can_act = len(self.owner_tank_cells[self.turn]) > 0
                else:
                    # OLD RULE (Bots): Start Zones or Own Stacks
                    # 1. Existing Own Stacks?
                    if req_type == 'bot':
                        can_act = len(self.owner_bot_cells[self.turn]) > 0
                    # 2. Empty Start Zones?
                    if not can_act:
                        starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
                        for pos in starts:
                            if pos not in self.grid:
                                can_act = True
                                break
        
        # 2. MOVE VALID?
        elif "move" in self.action_type:
            # Check if user has ANY units of this type that can move
            opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
            for pos, unit in self.grid.items():
                if unit['owner'] == self.turn and unit['type'] == req_type:
                    # Check neighbors for THIS unit
                    neighbors = self.get_hex_neighbors(pos[0], pos[1])
                    for nr, nc in neighbors:
                        # NEW RULE: No entry to opponent starting hexes
                        if (nr, nc) in opp_starts: continue

                        target_cell = self.grid.get((nr,nc))
                        
                        # Apply same rules as in action_mask
                        # 1. Tank -> Artefact = No
                        if req_type == 'tank' and (nr, nc) in self.artefacts: continue
                        
                        # 2. Bot -> Enemy = No
                        if req_type == 'bot' and target_cell and target_cell['owner'] != self.turn: continue
                        
                        # 3. Diff Type Merge = No
                        if target_cell and target_cell['owner'] == self.turn and target_cell['type'] != req_type: continue
                        
                        # 4. Tank Attack Size Rule
                        if req_type == 'tank' and target_cell and target_cell['owner'] != self.turn:
                            if target_cell['type'] == 'tank' and target_cell['count'] >= unit['count']:
                                continue
                        
                        # If we reach here, at least one move is possible
                        can_act = True
                        break
                if can_act: break
        
        if not can_act:
            self.end_turn()
            if not self.headless:
                self.info_message = f"Skipped: No valid {self.action_type} actions."

    def execute_add(self, r, c):
        # 1. SECURITY CHECK: State
        if self.state != "PERFORM_ACTIONS": return False
        if "add" not in self.action_type: return False

        unit_type = 'tank' if 'tank' in self.action_type else 'bot'
        current = self.grid.get((r,c))
        
        # 2. SECURITY CHECK: Unit Cap
        current_count = self.get_player_unit_count(self.turn)
        if current_count >= 20:
             self.info_message = "Unit Cap Reached (20 Max)!"
             return False

        # 3. SECURITY CHECK: Valid Placement Location
        
        if unit_type == 'tank':
            # === NEW TANK RULE ===
            
            # 1. Must be empty, or an own tank stack.
            is_own_tank_stack = (
                current is not None and
                current['owner'] == self.turn and
                current['type'] == 'tank'
            )
            if current is not None and not is_own_tank_stack:
                self.info_message = "Tanks: Drop on empty hex or own tank stack."
                return False

            # 2. Must not be opponent start zone
            opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
            if (r,c) in opp_starts:
                self.info_message = "Cannot place in opponent start zone."
                return False

            # 3. Must not be an artefact hex
            if (r,c) in self.artefacts:
                self.info_message = "Cannot place tank on an artefact."
                return False
                
            # === EXECUTION (Single Unit) ===
            if is_own_tank_stack:
                current['count'] += 1
            else:
                self.grid[(r,c)] = {'owner': self.turn, 'type': 'tank', 'count': 1}
            self._cache_dirty = True
            self.actions_remaining -= 1
            self.info_message = f"Added tank. Actions: {self.actions_remaining}"
            
            if self.actions_remaining <= 0:
                self.end_turn()
            else:
                self.check_actions_still_possible()
            return True

        else:
            # === OLD BOT RULE (Unchanged) ===
            # 1. Own Stacks (Merging)
            is_own_stack = (current and current['owner'] == self.turn and current['type'] == unit_type)
            
            # 2. Starting Zones (if empty or own)
            valid_starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
            is_start_zone = ((r,c) in valid_starts) and (not current or is_own_stack)

            if not (is_own_stack or is_start_zone):
                self.info_message = "Bots: Drop on Start Zone or Own Stack"
                return False

            # --- EXECUTION ---
            if current:
                current['count'] += 1
                self._cache_dirty = True
                self.actions_remaining -= 1
                self.info_message = f"Added {unit_type}. Actions: {self.actions_remaining}"
            else:
                self.grid[(r,c)] = {'owner': self.turn, 'type': unit_type, 'count': 1}
                self._cache_dirty = True
                self.actions_remaining -= 1
                self.info_message = f"Added {unit_type}. Actions: {self.actions_remaining}"
                
            if self.actions_remaining <= 0:
                self.end_turn()
            else:
                self.check_actions_still_possible()
            
            return True

    def handle_click(self, r, c):
        if self.state != "PERFORM_ACTIONS": return
        
        # 1. HANDLE ADDING UNITS
        if "add" in self.action_type:
            # We simply try to add at the clicked location.
            # The core engine will reject it if it's not a start zone or valid stack.
            success = self.execute_add(r, c)
            if not success and self.headless:
                print(f"Add rejected by core: {self.info_message}")

        # 2. HANDLE MOVING UNITS
        elif "move" in self.action_type:
            if self.selected_hex:
                # User has already selected a source and is clicking a destination
                
                # Deselect if clicking the same hex
                if (r,c) == self.selected_hex:
                    self.selected_hex = None 
                    return

                # --- THE "DUMB" UI CHANGE ---
                # Previously, we checked for neighbors, artefacts, and unit types here.
                # NOW, we remove all those checks. We just ask the engine:
                # "Can I move from Selected to Here?"
                
                success, _, _ = self.execute_move(self.selected_hex, (r,c), self.move_amount_selection)
                if not success:
                    # If the move failed (illegal), we usually keep the selection
                    # so the user can try a valid move instead.
                    pass 
                else:
                    # If successful, the engine has already updated the grid.
                    pass

            else:
                # User is selecting the Source hex
                if (r,c) in self.grid and self.grid[(r,c)]['owner'] == self.turn:
                    req_type = 'tank' if 'tank' in self.action_type else 'bot'
                    
                    # We still check if the unit type matches the satellite card
                    # (This is a basic UI usability filter, not really a rule check)
                    if self.grid[(r,c)]['type'] == req_type:
                        self.selected_hex = (r,c)
                        self.move_amount_selection = self.grid[(r,c)]['count']
                        self.info_message = f"Selected {req_type}. Click destination."
                    else:
                        self.info_message = f"Wrong unit type! Satellite needs {req_type}."

    def check_win(self):
        # 1. Score >= 9
        if self.scores[self.turn] >= 9:
            self.winner = self.turn
            self.state = "GAME_OVER"
            return True
        # 2. All Artefacts Captured
        if len(self.artefacts) == 0:
            if self.scores[0] > self.scores[1]: self.winner = 0
            elif self.scores[1] > self.scores[0]: self.winner = 1
            else: self.winner = self.turn # Tie-breaker
            self.state = "GAME_OVER"
            return True
        return False

   # In SatellitesGame class

    # In SatellitesGame class

    def execute_move(self, start, end, amount):
        # 1. SECURITY CHECKS
        # Failures must now return a tuple: (False, 0, 0)
        # 0 kills, 0 points
        if start not in self.grid: return False, 0, 0
        cell = self.grid[start]
        if cell['owner'] != self.turn: return False, 0, 0
        
        neighbors = self.get_hex_neighbors(start[0], start[1])
        if end not in neighbors:
            self.info_message = "Invalid Move: Not adjacent"
            return False, 0, 0

        opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
        if end in opp_starts:
            self.info_message = "Cannot move onto opponent starting hex!"
            return False, 0, 0

        move_type = cell['type']
        if move_type == 'tank' and end in self.artefacts:
            self.info_message = "Tanks cannot capture artefacts!"
            return False, 0, 0

        # --- EXECUTION ---
        cell['count'] -= amount
        self._cache_dirty = True
        if cell['count'] < 0: return False, 0, 0
        
        if cell['count'] == 0:
            del self.grid[start]
            
        target = self.grid.get(end)
        did_move_in = True 
        
        # Track our rewards
        units_destroyed = 0 
        score_gain = 0
        
        if target:
            if target['owner'] == self.turn:
                # Merge
                if target['type'] == move_type:
                    target['count'] += amount
                else: 
                     # Blocked
                     self.info_message = "Blocked: Different unit type"
                     if start not in self.grid: self.grid[start] = cell
                     self.grid[start]['count'] += amount 
                     return False, 0, 0
            else:
                # Combat / Attack
                if move_type == 'bot':
                    self.info_message = "Bots cannot attack!"
                    if start not in self.grid: self.grid[start] = cell
                    self.grid[start]['count'] += amount
                    return False, 0, 0

                if move_type == 'tank' and target['type'] == 'tank' and target['count'] >= amount:
                     self.info_message = "Can only attack smaller tank stack!"
                     if start not in self.grid: self.grid[start] = cell
                     self.grid[start]['count'] += amount
                     return False, 0, 0

                # Successful Kill
                units_destroyed = target['count'] 
                del self.grid[end]
                
                if move_type == 'tank':
                    did_move_in = False
                    if start not in self.grid: self.grid[start] = cell
                    self.grid[start]['count'] += amount
                    self.info_message = "Attack Successful! Tank holds position."
                else:
                    self.grid[end] = {'owner': self.turn, 'type': move_type, 'count': amount}
        else:
            # Move to empty
            self.grid[end] = {'owner': self.turn, 'type': move_type, 'count': amount}
        
        # --- ARTEFACT LOGIC ---
        if did_move_in and end in self.artefacts:
            self.artefacts.remove(end)
            self.is_artefact_cell[self.coord_to_cell_id[end]] = False
            # Rule: 1 point per bot in the stack
            score_gain = amount  
            self.scores[self.turn] += score_gain 
            self.info_message = f"Captured Artefact! +{score_gain} pts"

        if self.check_win():
             return True, units_destroyed, score_gain

        self.actions_remaining -= 1
        
        # Message Logic
        if did_move_in and "Captured" not in self.info_message:
            self.info_message = f"Moved {amount} units. Actions: {self.actions_remaining}"
        elif not did_move_in:
            self.info_message += f" ({self.actions_remaining} left)"
        
        if self.actions_remaining <= 0:
            self.end_turn()
            self.selected_hex = None
        else:
            if did_move_in:
                self.selected_hex = end
            else:
                self.selected_hex = start
            
        return True, units_destroyed, score_gain

    def select_satellite(self, idx):
        if self.state != "CHOOSE_SATELLITE": return
        sat = self.satellites[idx]
        if sat['charges'] > 0:
            self.active_satellite_idx = idx
            self.action_type = sat['type']
            self.picked_up_charges = sat['charges']
            
            # Remove charges immediately
            sat['charges'] = 0
            
            # NEW SATE: Choose Direction
            self.state = "CHOOSE_DIRECTION"
            self.info_message = "Choose Distribution Direction"
        else:
            self.info_message = "Satellite expects charges!"

    def set_distribution_direction(self, clockwise):
        self.distribution_direction = 1 if clockwise else -1
        
        # Distribute Immediately
        self.perform_distribution()
        
        # FIX: Check if action is possible
        req_type = 'tank' if 'tank' in self.satellites[self.active_satellite_idx]['type'] else 'bot'
        action_main = 'move' if 'move' in self.satellites[self.active_satellite_idx]['type'] else 'add'
        
        can_act = True
        self._ensure_cache()
        if action_main == 'move':
            # Check if user has ANY units of this type
            if req_type == 'tank':
                has_units = len(self.owner_tank_cells[self.turn]) > 0
            else:
                has_units = len(self.owner_bot_cells[self.turn]) > 0
            if not has_units:
                can_act = False
                self.info_message = f"No {req_type}s to move! Turn Ending."
        
        elif action_main == 'add':
            # Check if any valid placement exists
            can_add = False
            
            # 1. Check Cap
            if self.get_player_unit_count(self.turn) >= 20: 
                can_add = False # At cap, no adds allowed
            else:
                if req_type == 'tank':
                    # Tanks can drop on own tank stacks, or empty non-opponent-start hexes.
                    opp_start_mask = self.is_p1_start_cell if self.turn == 0 else self.is_p0_start_cell
                    for cid in range(self.num_cells):
                        if self.unit_owner[cid] == -1 and not opp_start_mask[cid] and not self.is_artefact_cell[cid]:
                            can_add = True
                            break
                    if not can_add:
                        can_add = len(self.owner_tank_cells[self.turn]) > 0
                else:
                    # OLD RULE (Bots)
                    # 2. Existing Own Stacks?
                    if req_type == 'bot':
                        can_add = len(self.owner_bot_cells[self.turn]) > 0
                    
                    # 3. Empty Start Zones?
                    if not can_add:
                        starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
                        for pos in starts:
                            if pos not in self.grid: # Empty start zone
                                can_add = True
                                break
            
            if not can_add:
                can_act = False
                self.info_message = f"No valid placement for {req_type}! Turn Ending."
        
        if can_act:
            self.actions_remaining = self.picked_up_charges
            self.state = "PERFORM_ACTIONS"
            self.info_message = f"Action: {self.satellites[self.active_satellite_idx]['name']} ({self.actions_remaining} remaining)"
        else:
            self.end_turn()
            self.info_message = f"Skipped Reason: No {req_type}s."

    def perform_distribution(self):
        # Use stored direction
        to_distribute = self.picked_up_charges 
        idx = self.active_satellite_idx
        
        while to_distribute > 0:
            idx = (idx + self.distribution_direction) % 6
            self.satellites[idx]['charges'] += 1
            to_distribute -= 1

    def end_turn(self):
        # FIX: Check Turn Limit
        if self.turn_count >= self.MAX_TURNS:
            self.state = "GAME_OVER"
            self.info_message = "Max Turn Limit Reached."
            if self.scores[0] > self.scores[1]: self.winner = 0
            elif self.scores[1] > self.scores[0]: self.winner = 1
            else: self.winner = -1 # Draw
            return

        self.turn = 1 - self.turn
        if self.turn == 0: 
            self.turn_count += 1
            # Log turn if needed or other round-based logic
        
        self.state = "CHOOSE_SATELLITE"
        self.selected_hex = None
        self.active_satellite_idx = None
        
        p_name = "Player 1 (Blue)" if self.turn == 1 else "Player 0 (Red)"
        if not self.headless and "Skipped" not in self.info_message:
            self.info_message = f"{p_name}'s Turn. Choose Satellite."


# ==========================================
# PART 2: PYGAME UI
# ==========================================

