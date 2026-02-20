from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Protocol, Tuple


Action = Any
Player = int


class MCTSAdapter(Protocol):
    """Game adapter required by MCTS.

    This keeps MCTS decoupled from engine internals.
    """

    def clone(self, state: Any) -> Any:
        ...

    def legal_actions(self, state: Any) -> List[Action]:
        ...

    def apply_action(self, state: Any, action: Action) -> Any:
        ...

    def apply_action_with_undo(self, state: Any, action: Action) -> Tuple[bool, Any]:
        ...

    def undo_action(self, state: Any, token: Any) -> None:
        ...

    def is_terminal(self, state: Any) -> bool:
        ...

    def current_player(self, state: Any) -> Player:
        ...

    def outcome_for_player(self, state: Any, player: Player) -> float:
        """Return terminal value from player's perspective in [-1.0, 1.0]."""

    def evaluate(self, state: Any, player: Player) -> float:
        ...

    def action_prior(self, state: Any, action: Action, player: Player) -> float:
        ...


@dataclass
class Node:
    parent: Optional["Node"] = None
    action_from_parent: Optional[Action] = None
    player_to_move: Player = 0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried_actions: Optional[List[Action]] = None

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class MCTS:
    def __init__(
        self,
        adapter: MCTSAdapter,
        *,
        iterations: int = 400,
        c_puct: float = 1.41,
        rollout_depth: int = 60,
        seed: Optional[int] = None,
    ) -> None:
        self.adapter = adapter
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def select_action(self, root_state: Any) -> Tuple[Action, Dict[str, float]]:
        work_state = self.adapter.clone(root_state)
        root = Node(
            player_to_move=self.adapter.current_player(work_state),
        )
        root.untried_actions = self.adapter.legal_actions(work_state)
        if not root.untried_actions:
            raise ValueError("No legal actions from root state.")

        for _ in range(self.iterations):
            node, path_tokens = self._select_and_expand(root, work_state)
            value = self._simulate_from_clone(work_state, node.player_to_move)
            self._backpropagate(node, value)
            self._undo_tokens(work_state, path_tokens)

        best_action, best_child = max(
            root.children.items(),
            key=lambda kv: kv[1].visits,
        )
        stats = {
            "root_visits": float(root.visits),
            "best_action_visits": float(best_child.visits),
            "best_action_value": best_child.value,
        }
        return best_action, stats

    def _select_and_expand(self, node: Node, state: Any) -> Tuple[Node, List[Any]]:
        path_tokens: List[Any] = []
        while not self.adapter.is_terminal(state):
            if node.untried_actions is None:
                node.untried_actions = self.adapter.legal_actions(state)
            if node.untried_actions:
                child, token = self._expand(node, state)
                path_tokens.append(token)
                return child, path_tokens
            action, child = self._best_ucb_edge(node)
            success, token = self.adapter.apply_action_with_undo(state, action)
            if not success:
                # Defensive: if engine rejects, prune edge and retry.
                del node.children[action]
                continue
            path_tokens.append(token)
            node = child
        return node, path_tokens

    def _expand(self, node: Node, state: Any) -> Tuple[Node, Any]:
        idx = self.rng.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(idx)
        success, token = self.adapter.apply_action_with_undo(state, action)
        if not success:
            # Defensive fallback: treat as dead-end leaf.
            child = Node(
                parent=node,
                action_from_parent=action,
                player_to_move=node.player_to_move,
            )
            child.untried_actions = []
            node.children[action] = child
            return child, None
        child = Node(
            parent=node,
            action_from_parent=action,
            player_to_move=self.adapter.current_player(state),
        )
        child.untried_actions = self.adapter.legal_actions(state)
        node.children[action] = child
        return child, token

    def _best_ucb_edge(self, node: Node) -> Tuple[Action, Node]:
        assert node.children, "UCB selection requires existing children."
        log_n = math.log(max(1, node.visits))

        def ucb(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.value
            explore = self.c_puct * math.sqrt(log_n / child.visits)
            return exploit + explore

        action, child = max(node.children.items(), key=lambda kv: ucb(kv[1]))
        return action, child

    def _simulate_from_clone(self, state: Any, rollout_player: Player) -> float:
        sim_state = self.adapter.clone(state)
        depth = 0
        while depth < self.rollout_depth and not self.adapter.is_terminal(sim_state):
            actions = self.adapter.legal_actions(sim_state)
            if not actions:
                break
            action = self._sample_action(sim_state, actions, rollout_player)
            sim_state = self.adapter.apply_action(sim_state, action)
            depth += 1

        if not self.adapter.is_terminal(sim_state):
            if hasattr(self.adapter, "evaluate"):
                return float(self.adapter.evaluate(sim_state, rollout_player))
            return 0.0
        return self.adapter.outcome_for_player(sim_state, rollout_player)

    def _sample_action(self, state: Any, actions: List[Action], player: Player) -> Action:
        if not hasattr(self.adapter, "action_prior"):
            return self.rng.choice(actions)
        priors = []
        for a in actions:
            p = float(self.adapter.action_prior(state, a, player))
            priors.append(max(0.01, p + 1.0))
        total = sum(priors)
        r = self.rng.random() * total
        acc = 0.0
        for a, w in zip(actions, priors):
            acc += w
            if acc >= r:
                return a
        return actions[-1]

    def _undo_tokens(self, state: Any, tokens: List[Any]) -> None:
        for token in reversed(tokens):
            if token is not None:
                self.adapter.undo_action(state, token)

    def _backpropagate(self, node: Node, value: float) -> None:
        cur = node
        cur_value = value
        while cur is not None:
            cur.visits += 1
            cur.value_sum += cur_value
            cur_value = -cur_value
            cur = cur.parent


class SatellitesAdapter:
    """Adapter for engine.SatellitesGame."""

    DEFAULT_WEIGHTS = {
        "score_diff": 100.0,
        "near_win": 60.0,
        "cap_next": 40.0,
        "race_early": 20.0,
        "race_mid": 35.0,
        "race_late": 50.0,
        "bot_tank_threat": 25.0,
        "tank_dominance": 20.0,
        "tanks_near_artefact": 15.0,
        "sat_charge": 2.0,
        "sat_move_bot_bonus": 1.5,
        "sat_move_tank_bonus": 1.2,
        "sat_add_tank_bonus": 1.0,
        "sat_add_bot_bonus": 0.7,
        "add_tank_near_artefact": 1.2,
        "add_tank_near_enemy_bot": 1.0,
        "add_bot_dist_gain": 2.0,
        "add_bot_tank_threat_penalty": 1.5,
        "move_bot_capture": 4.0,
        "move_bot_dist_delta": 0.5,
        "move_bot_tank_threat_penalty": 2.5,
        "move_tank_adj_enemy_bot": 1.6,
        "move_tank_vs_tank_win": 1.0,
        "move_tank_vs_tank_lose": 0.8,
        "move_tank_near_artefact": 0.8,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = float(v)

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()

    def set_weight(self, name: str, value: float) -> None:
        if name not in self.weights:
            raise KeyError(f"Unknown weight: {name}")
        self.weights[name] = float(value)

    def clone(self, state: Any) -> Any:
        if hasattr(state, "clone"):
            return state.clone()
        import copy
        return copy.deepcopy(state)

    def legal_actions(self, state: Any) -> List[Action]:
        return state.legal_actions()

    def apply_action(self, state: Any, action: Action) -> Any:
        ok = state.apply_action(action)
        if not ok:
            raise ValueError(f"Illegal action: {action}")
        return state

    def apply_action_with_undo(self, state: Any, action: Action) -> Tuple[bool, Any]:
        ok, token, _ = state.apply_action_with_undo(action)
        return ok, token

    def undo_action(self, state: Any, token: Any) -> None:
        state.undo_action(token)

    def is_terminal(self, state: Any) -> bool:
        return getattr(state, "state", None) == "GAME_OVER"

    def current_player(self, state: Any) -> Player:
        return int(state.turn)

    def outcome_for_player(self, state: Any, player: Player) -> float:
        winner = getattr(state, "winner", None)
        if winner == -1 or winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0

    def _unit_cells(self, state: Any, owner: int, utype: str):
        out = []
        for (r, c), u in state.grid.items():
            if u["owner"] == owner and u["type"] == utype:
                out.append(((r, c), u["count"]))
        return out

    def _min_bot_dist(self, state: Any, owner: int, artefact):
        bots = self._unit_cells(state, owner, "bot")
        if not bots:
            return 99
        return min(state.get_hex_distance(pos, artefact) for pos, _ in bots)

    def _is_adj_enemy_tank(self, state: Any, owner: int, pos):
        enemy = 1 - owner
        for nr, nc in state.get_hex_neighbors(pos[0], pos[1]):
            u = state.grid.get((nr, nc))
            if u and u["owner"] == enemy and u["type"] == "tank":
                return True
        return False

    def evaluate(self, state: Any, player: Player) -> float:
        enemy = 1 - player
        my_points = state.scores[player]
        opp_points = state.scores[enemy]
        score_diff = my_points - opp_points
        my_need = max(0, 9 - my_points)
        opp_need = max(0, 9 - opp_points)

        val = 0.0
        w = self.weights
        val += w["score_diff"] * score_diff
        val += w["near_win"] * (opp_need - my_need)

        artefacts = list(state.artefacts)
        cap_next_my = 0
        cap_next_opp = 0
        race_sum = 0.0
        for a in artefacts:
            d_my = self._min_bot_dist(state, player, a)
            d_opp = self._min_bot_dist(state, enemy, a)
            if d_my == 1:
                cap_next_my += 1
            if d_opp == 1:
                cap_next_opp += 1
            race_sum += max(-3, min(3, d_opp - d_my))
        val += w["cap_next"] * (cap_next_my - cap_next_opp)
        if len(artefacts) >= 5:
            race_w = w["race_early"]
        elif len(artefacts) >= 3:
            race_w = w["race_mid"]
        else:
            race_w = w["race_late"]
        val += race_w * race_sum

        my_bots = self._unit_cells(state, player, "bot")
        opp_bots = self._unit_cells(state, enemy, "bot")
        my_threat = 0.0
        opp_threat = 0.0
        for pos, ct in my_bots:
            if self._is_adj_enemy_tank(state, player, pos):
                my_threat += 1.0 + 0.25 * max(0, ct - 1)
        for pos, ct in opp_bots:
            if self._is_adj_enemy_tank(state, enemy, pos):
                opp_threat += 1.0 + 0.25 * max(0, ct - 1)
        val += w["bot_tank_threat"] * (opp_threat - my_threat)

        # Tank local dominance near enemy tanks.
        dom = 0.0
        my_tanks = self._unit_cells(state, player, "tank")
        for pos, ct in my_tanks:
            for nr, nc in state.get_hex_neighbors(pos[0], pos[1]):
                u = state.grid.get((nr, nc))
                if u and u["owner"] == enemy and u["type"] == "tank":
                    dom += 1.0 if ct >= u["count"] else -1.0
        val += w["tank_dominance"] * dom

        # Tank proximity to artefacts.
        my_near = 0
        opp_near = 0
        for pos, _ in my_tanks:
            if any(state.get_hex_distance(pos, a) <= 2 for a in artefacts):
                my_near += 1
        for pos, _ in self._unit_cells(state, enemy, "tank"):
            if any(state.get_hex_distance(pos, a) <= 2 for a in artefacts):
                opp_near += 1
        val += w["tanks_near_artefact"] * (my_near - opp_near)

        # Squash to [-1, 1] for stable MCTS backups.
        return max(-1.0, min(1.0, val / 300.0))

    def action_prior(self, state: Any, action: Action, player: Player) -> float:
        enemy = 1 - player
        w = self.weights
        kind = action[0]
        if kind == "select_satellite":
            idx = action[1]
            sat = state.satellites[idx]
            p = w["sat_charge"] * sat["charges"]
            st = sat["type"]
            if st == "move_bot":
                p += w["sat_move_bot_bonus"]
            elif st == "move_tank":
                p += w["sat_move_tank_bonus"]
            elif st == "add_tank":
                p += w["sat_add_tank_bonus"]
            else:
                p += w["sat_add_bot_bonus"]
            return p
        if kind == "set_direction":
            return 0.0
        if kind == "add":
            r, c = action[1], action[2]
            pos = (r, c)
            if "tank" in (state.action_type or ""):
                p = 0.5
                if any(state.get_hex_distance(pos, a) <= 2 for a in state.artefacts):
                    p += w["add_tank_near_artefact"]
                for nr, nc in state.get_hex_neighbors(r, c):
                    u = state.grid.get((nr, nc))
                    if u and u["owner"] == enemy and u["type"] == "bot":
                        p += w["add_tank_near_enemy_bot"]
                return p
            p = 0.5
            if state.artefacts:
                d = min(state.get_hex_distance(pos, a) for a in state.artefacts)
                p += max(0.0, w["add_bot_dist_gain"] - 0.3 * d)
            if self._is_adj_enemy_tank(state, player, pos):
                p -= w["add_bot_tank_threat_penalty"]
            return p
        if kind == "move":
            start, end, amount = action[1], action[2], action[3]
            u = state.grid.get(start)
            if not u:
                return 0.0
            if u["type"] == "bot":
                p = 0.5
                if end in state.artefacts:
                    p += w["move_bot_capture"]
                if state.artefacts:
                    d0 = min(state.get_hex_distance(start, a) for a in state.artefacts)
                    d1 = min(state.get_hex_distance(end, a) for a in state.artefacts)
                    p += w["move_bot_dist_delta"] * (d0 - d1)
                if self._is_adj_enemy_tank(state, player, end):
                    p -= w["move_bot_tank_threat_penalty"]
                return p + 0.1 * amount
            p = 0.5
            for nr, nc in state.get_hex_neighbors(end[0], end[1]):
                tgt = state.grid.get((nr, nc))
                if tgt and tgt["owner"] == enemy and tgt["type"] == "bot":
                    p += w["move_tank_adj_enemy_bot"]
                if tgt and tgt["owner"] == enemy and tgt["type"] == "tank":
                    p += w["move_tank_vs_tank_win"] if amount >= tgt["count"] else -w["move_tank_vs_tank_lose"]
            if state.artefacts and any(state.get_hex_distance(end, a) <= 2 for a in state.artefacts):
                p += w["move_tank_near_artefact"]
            return p
        return 0.0
