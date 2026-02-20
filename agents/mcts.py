from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
import time
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

    def tactical_priority(self, state: Any, action: Action, player: Player) -> int:
        ...

    def state_key(self, state: Any) -> Any:
        ...


@dataclass
class Node:
    parent: Optional["Node"] = None
    action_from_parent: Optional[Action] = None
    player_to_move: Player = 0
    state_key: Any = None
    visits: int = 0
    value_sum: float = 0.0
    child_actions: List[Action] = field(default_factory=list)
    child_nodes: List["Node"] = field(default_factory=list)
    child_priors: List[float] = field(default_factory=list)
    action_to_child_idx: Dict[Action, int] = field(default_factory=dict)
    untried_actions: Optional[List[Action]] = None
    untried_priors: Optional[List[float]] = None

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
        rollout_greedy_prob: float = 0.85,
        widening_base: int = 8,
        widening_step: int = 5,
        widening_every: int = 25,
        use_transposition: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.adapter = adapter
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.rollout_greedy_prob = rollout_greedy_prob
        self.widening_base = widening_base
        self.widening_step = widening_step
        self.widening_every = widening_every
        self.use_transposition = use_transposition
        self.tt_stats: Dict[Any, List[float]] = {}
        self.tt_legal: Dict[Any, Tuple[Action, ...]] = {}
        self.tt_ordered: Dict[Any, Tuple[Tuple[Action, ...], Tuple[float, ...]]] = {}
        self.rng = random.Random(seed)

    def select_action(self, root_state: Any) -> Tuple[Action, Dict[str, float]]:
        return self._select_action_internal(root_state, max_iterations=self.iterations)

    def select_action_for_time(
        self,
        root_state: Any,
        max_time_s: float,
        min_iterations: int = 1,
    ) -> Tuple[Action, Dict[str, float]]:
        deadline = time.perf_counter() + max(0.01, max_time_s)
        return self._select_action_internal(
            root_state,
            max_iterations=None,
            deadline=deadline,
            min_iterations=max(1, min_iterations),
        )

    def _select_action_internal(
        self,
        root_state: Any,
        *,
        max_iterations: Optional[int],
        deadline: Optional[float] = None,
        min_iterations: int = 1,
    ) -> Tuple[Action, Dict[str, float]]:
        work_state = self.adapter.clone(root_state)
        root_key = self.adapter.state_key(work_state)
        root = Node(
            player_to_move=self.adapter.current_player(work_state),
            state_key=root_key,
        )
        if self.use_transposition and root_key in self.tt_stats:
            s = self.tt_stats[root_key]
            root.visits = int(s[0])
            root.value_sum = float(s[1])
        root.untried_actions, root.untried_priors = self._ordered_actions(work_state, root.player_to_move)
        if not root.untried_actions:
            raise ValueError("No legal actions from root state.")

        iters_done = 0
        while True:
            if max_iterations is not None and iters_done >= max_iterations:
                break
            if deadline is not None and iters_done >= min_iterations and time.perf_counter() >= deadline:
                break
            node, path_tokens = self._select_and_expand(root, work_state)
            value = self._simulate_from_clone(work_state, node.player_to_move)
            self._backpropagate(node, value)
            self._undo_tokens(work_state, path_tokens)
            iters_done += 1

        if not root.child_nodes:
            raise ValueError("No child nodes expanded from root.")
        best_idx = max(range(len(root.child_nodes)), key=lambda i: root.child_nodes[i].visits)
        best_action = root.child_actions[best_idx]
        best_child = root.child_nodes[best_idx]
        stats = {
            "root_visits": float(root.visits),
            "best_action_visits": float(best_child.visits),
            "best_action_value": best_child.value,
            "iterations": float(iters_done),
        }
        return best_action, stats

    def _select_and_expand(self, node: Node, state: Any) -> Tuple[Node, List[Any]]:
        path_tokens: List[Any] = []
        while not self.adapter.is_terminal(state):
            if node.untried_actions is None:
                node.untried_actions, node.untried_priors = self._ordered_actions(state, node.player_to_move)
            total_actions = len(node.child_nodes) + len(node.untried_actions)
            allowed = self.widening_base + (node.visits // self.widening_every) * self.widening_step
            if node.untried_actions and len(node.child_nodes) < min(total_actions, allowed):
                child, token = self._expand(node, state)
                path_tokens.append(token)
                return child, path_tokens
            edge_idx, action, child = self._best_ucb_edge(node)
            success, token = self.adapter.apply_action_with_undo(state, action)
            if not success:
                # Defensive: if engine rejects, prune edge and retry.
                self._remove_child_at(node, edge_idx)
                continue
            path_tokens.append(token)
            node = child
        return node, path_tokens

    def _expand(self, node: Node, state: Any) -> Tuple[Node, Any]:
        action = node.untried_actions.pop(0)
        prior = node.untried_priors.pop(0) if node.untried_priors else 0.0
        success, token = self.adapter.apply_action_with_undo(state, action)
        if not success:
            # Defensive fallback: treat as dead-end leaf.
            child = Node(
                parent=node,
                action_from_parent=action,
                player_to_move=node.player_to_move,
            )
            child.untried_actions = []
            child.untried_priors = []
            self._add_child(node, action, prior, child)
            return child, None
        child = Node(
            parent=node,
            action_from_parent=action,
            player_to_move=self.adapter.current_player(state),
            state_key=self.adapter.state_key(state),
        )
        if self.use_transposition and child.state_key in self.tt_stats:
            s = self.tt_stats[child.state_key]
            child.visits = int(s[0])
            child.value_sum = float(s[1])
        child.untried_actions, child.untried_priors = self._ordered_actions(state, child.player_to_move)
        self._add_child(node, action, prior, child)
        return child, token

    def _add_child(self, node: Node, action: Action, prior: float, child: Node) -> None:
        idx = len(node.child_nodes)
        node.action_to_child_idx[action] = idx
        node.child_actions.append(action)
        node.child_nodes.append(child)
        node.child_priors.append(prior)

    def _remove_child_at(self, node: Node, idx: int) -> None:
        action = node.child_actions[idx]
        del node.action_to_child_idx[action]
        last = len(node.child_nodes) - 1
        if idx != last:
            node.child_actions[idx] = node.child_actions[last]
            node.child_nodes[idx] = node.child_nodes[last]
            node.child_priors[idx] = node.child_priors[last]
            moved_action = node.child_actions[idx]
            node.action_to_child_idx[moved_action] = idx
        node.child_actions.pop()
        node.child_nodes.pop()
        node.child_priors.pop()

    def _legal_actions_for_state(self, state: Any) -> List[Action]:
        if not self.use_transposition:
            return self.adapter.legal_actions(state)
        key = self.adapter.state_key(state)
        cached = self.tt_legal.get(key)
        if cached is not None:
            return list(cached)
        actions = self.adapter.legal_actions(state)
        self.tt_legal[key] = tuple(actions)
        return actions

    def _ordered_actions(self, state: Any, player: Player) -> Tuple[List[Action], List[float]]:
        if self.use_transposition:
            k = (self.adapter.state_key(state), player)
            cached = self.tt_ordered.get(k)
            if cached is not None:
                return list(cached[0]), list(cached[1])
        actions = self._legal_actions_for_state(state)
        if not actions:
            return [], []
        if not hasattr(self.adapter, "action_prior"):
            priors = [0.0] * len(actions)
            return actions, priors
        scored = []
        tactical_available = hasattr(self.adapter, "tactical_priority")
        for a in actions:
            prior = float(self.adapter.action_prior(state, a, player))
            if tactical_available:
                tact = int(self.adapter.tactical_priority(state, a, player))
                scored.append((tact, prior, a))
            else:
                scored.append((0, prior, a))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        ordered_actions = [x[2] for x in scored]
        ordered_priors = [x[1] for x in scored]
        if self.use_transposition:
            self.tt_ordered[(self.adapter.state_key(state), player)] = (tuple(ordered_actions), tuple(ordered_priors))
        return ordered_actions, ordered_priors

    def _best_ucb_edge(self, node: Node) -> Tuple[int, Action, Node]:
        assert node.child_nodes, "UCB selection requires existing children."
        log_n = math.log(max(1, node.visits))

        def ucb(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.value
            explore = self.c_puct * math.sqrt(log_n / child.visits)
            return exploit + explore

        best_idx = max(range(len(node.child_nodes)), key=lambda i: ucb(node.child_nodes[i]))
        return best_idx, node.child_actions[best_idx], node.child_nodes[best_idx]

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
        priors = [float(self.adapter.action_prior(state, a, player)) for a in actions]
        if hasattr(self.adapter, "tactical_priority"):
            tactical = [int(self.adapter.tactical_priority(state, a, player)) for a in actions]
            best_t = max(tactical) if tactical else 0
            if best_t >= 80:
                idx = max(range(len(actions)), key=lambda i: (tactical[i], priors[i]))
                return actions[idx]

        # Fast greedy-biased policy over top-k actions.
        top_k = min(6, len(actions))
        top_idx = sorted(range(len(actions)), key=lambda i: priors[i], reverse=True)[:top_k]
        if self.rng.random() < self.rollout_greedy_prob:
            return actions[top_idx[0]]
        # Weighted sample only among top-k to reduce noise and per-step overhead.
        top_actions = [actions[i] for i in top_idx]
        top_weights = [max(0.01, priors[i] + 1.0) for i in top_idx]
        total = sum(top_weights)
        r = self.rng.random() * total
        acc = 0.0
        for a, w in zip(top_actions, top_weights):
            acc += w
            if acc >= r:
                return a
        return top_actions[-1]

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
            if self.use_transposition and cur.state_key is not None:
                s = self.tt_stats.get(cur.state_key)
                if s is None:
                    self.tt_stats[cur.state_key] = [1.0, cur_value]
                else:
                    s[0] += 1.0
                    s[1] += cur_value
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
        "move_bot_capture": 10.0,
        "move_bot_capture_stack_scale": 0.8,
        "move_bot_dist_delta": 0.5,
        "move_bot_tank_threat_penalty": 2.5,
        "move_bot_safe_approach": 2.0,
        "move_tank_adj_enemy_bot": 1.6,
        "move_tank_vs_tank_win": 1.0,
        "move_tank_vs_tank_lose": 0.8,
        "move_tank_near_artefact": 0.8,
        "eval_safe_bot_near_artefact": 35.0,
        "add_bot_stack_near_artefact": 2.2,
        "gifted_power_turn": 45.0,
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

    def save_weights(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.weights, f, indent=2, sort_keys=True)

    def load_weights(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k in self.weights:
                self.weights[k] = float(v)

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
        # Training baseline: no handcrafted leaf heuristic.
        return 0.0

    def action_prior(self, state: Any, action: Action, player: Player) -> float:
        return 0.0

    def tactical_priority(self, state: Any, action: Action, player: Player) -> int:
        return 0

    def state_key(self, state: Any) -> Any:
        # Fast, order-independent checksum for occupied cells.
        grid_checksum = 0
        for (r, c), u in state.grid.items():
            grid_checksum ^= hash((r, c, u["owner"], u["type"], u["count"]))

        sats_key = tuple((sat["type"], sat["charges"]) for sat in state.satellites)
        stamp = (
            state.turn,
            state.state,
            state.active_satellite_idx,
            state.actions_remaining,
            state.picked_up_charges,
            state.action_type,
            state.selected_hex,
            state.pending_move_dest,
            state.pending_move_max,
            state.move_amount_selection,
            getattr(state, "distribution_direction", None),
            tuple(state.scores),
            tuple(sorted(state.artefacts)),
            sats_key,
            state.winner,
            state.turn_count,
            len(state.grid),
            grid_checksum,
        )
        cache_stamp = getattr(state, "_mcts_key_stamp", None)
        if cache_stamp == stamp:
            return state._mcts_key_cache
        state._mcts_key_stamp = stamp
        state._mcts_key_cache = stamp
        return stamp
