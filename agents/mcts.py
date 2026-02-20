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

    def is_terminal(self, state: Any) -> bool:
        ...

    def current_player(self, state: Any) -> Player:
        ...

    def outcome_for_player(self, state: Any, player: Player) -> float:
        """Return terminal value from player's perspective in [-1.0, 1.0]."""


@dataclass
class Node:
    state: Any
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
        root = Node(
            state=self.adapter.clone(root_state),
            player_to_move=self.adapter.current_player(root_state),
        )
        root.untried_actions = self.adapter.legal_actions(root.state)
        if not root.untried_actions:
            raise ValueError("No legal actions from root state.")

        for _ in range(self.iterations):
            node = self._select(root)
            value = self._simulate(node)
            self._backpropagate(node, value)

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

    def _select(self, node: Node) -> Node:
        while not self.adapter.is_terminal(node.state):
            if node.untried_actions is None:
                node.untried_actions = self.adapter.legal_actions(node.state)
            if node.untried_actions:
                return self._expand(node)
            node = self._best_ucb_child(node)
        return node

    def _expand(self, node: Node) -> Node:
        idx = self.rng.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(idx)
        next_state = self.adapter.apply_action(self.adapter.clone(node.state), action)
        child = Node(
            state=next_state,
            parent=node,
            action_from_parent=action,
            player_to_move=self.adapter.current_player(next_state),
        )
        child.untried_actions = self.adapter.legal_actions(next_state)
        node.children[action] = child
        return child

    def _best_ucb_child(self, node: Node) -> Node:
        assert node.children, "UCB selection requires existing children."
        log_n = math.log(max(1, node.visits))

        def ucb(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.value
            explore = self.c_puct * math.sqrt(log_n / child.visits)
            return exploit + explore

        return max(node.children.values(), key=ucb)

    def _simulate(self, node: Node) -> float:
        state = self.adapter.clone(node.state)
        rollout_player = node.player_to_move

        depth = 0
        while depth < self.rollout_depth and not self.adapter.is_terminal(state):
            actions = self.adapter.legal_actions(state)
            if not actions:
                break
            action = self.rng.choice(actions)
            state = self.adapter.apply_action(state, action)
            depth += 1

        if not self.adapter.is_terminal(state):
            return 0.0
        return self.adapter.outcome_for_player(state, rollout_player)

    def _backpropagate(self, node: Node, value: float) -> None:
        cur = node
        cur_value = value
        while cur is not None:
            cur.visits += 1
            cur.value_sum += cur_value
            cur_value = -cur_value
            cur = cur.parent


class SatellitesAdapter:
    """Minimal adapter scaffold for engine.SatellitesGame.

    Implement game-specific action encoding once legal action generation and
    pure transition API are formalized in engine.py.
    """

    def clone(self, state: Any) -> Any:
        import copy

        return copy.deepcopy(state)

    def legal_actions(self, state: Any) -> List[Action]:
        raise NotImplementedError("TODO: expose legal action generation in engine.")

    def apply_action(self, state: Any, action: Action) -> Any:
        raise NotImplementedError("TODO: expose a pure step/apply API in engine.")

    def is_terminal(self, state: Any) -> bool:
        return getattr(state, "state", None) == "GAME_OVER"

    def current_player(self, state: Any) -> Player:
        return int(state.turn)

    def outcome_for_player(self, state: Any, player: Player) -> float:
        winner = getattr(state, "winner", None)
        if winner == -1 or winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0

