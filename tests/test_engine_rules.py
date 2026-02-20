from engine import SatellitesGame
from agents.mcts import MCTS, SatellitesAdapter


def _prep_add_tank(game: SatellitesGame, turn: int = 0) -> None:
    game.turn = turn
    game.state = "PERFORM_ACTIONS"
    game.action_type = "add_tank"
    game.actions_remaining = 1


def test_win_threshold_is_nine_points() -> None:
    game = SatellitesGame(headless=True)
    game.turn = 0
    game.scores[0] = 9

    assert game.check_win() is True
    assert game.state == "GAME_OVER"
    assert game.winner == 0


def test_tank_attack_does_not_move_into_target_hex() -> None:
    game = SatellitesGame(headless=True)
    game.grid = {
        (4, 4): {"owner": 0, "type": "tank", "count": 2},
        (4, 5): {"owner": 1, "type": "bot", "count": 1},
    }
    game.turn = 0
    game.actions_remaining = 1

    success, kills, _ = game.execute_move((4, 4), (4, 5), 2)

    assert success is True
    assert kills == 1
    assert (4, 5) not in game.grid
    assert game.grid[(4, 4)]["owner"] == 0
    assert game.grid[(4, 4)]["type"] == "tank"
    assert game.grid[(4, 4)]["count"] == 2


def test_tank_drop_allows_empty_non_opponent_start_hex() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {}

    ok = game.execute_add(4, 5)

    assert ok is True
    assert game.grid[(4, 5)] == {"owner": 0, "type": "tank", "count": 1}


def test_tank_drop_allows_own_tank_stack() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {(4, 5): {"owner": 0, "type": "tank", "count": 2}}

    ok = game.execute_add(4, 5)

    assert ok is True
    assert game.grid[(4, 5)]["count"] == 3


def test_tank_drop_rejects_opponent_start_hex() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {}

    ok = game.execute_add(8, 3)

    assert ok is False
    assert (8, 3) not in game.grid
    assert "opponent start zone" in game.info_message


def test_tank_drop_rejects_artefact_hex() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {}

    ok = game.execute_add(4, 4)

    assert ok is False
    assert (4, 4) not in game.grid
    assert "artefact" in game.info_message.lower()


def test_precomputed_topology_is_consistent() -> None:
    game = SatellitesGame(headless=True)

    assert game.num_cells == sum(game.row_widths)
    assert len(game.cell_id_to_coord) == game.num_cells
    assert len(game.coord_to_cell_id) == game.num_cells
    assert len(game.neighbors_by_cell_id) == game.num_cells

    for cid, coord in enumerate(game.cell_id_to_coord):
        assert game.coord_to_cell_id[coord] == cid
        assert game.get_hex_neighbors(coord[0], coord[1]) == list(game.neighbors_by_cell_id[cid])


def test_precomputed_distances_are_symmetric_and_zero_diagonal() -> None:
    game = SatellitesGame(headless=True)

    for cid in range(game.num_cells):
        assert game.distance_by_cell_id[cid][cid] == 0
    for a in range(game.num_cells):
        for b in range(game.num_cells):
            assert game.distance_by_cell_id[a][b] == game.distance_by_cell_id[b][a]

    # Basic sanity for helper
    assert game.get_hex_distance((0, 3), (0, 3)) == 0
    assert game.get_hex_distance((0, 3), (8, 4)) > 0


def test_cache_rebuilds_after_external_grid_assignment() -> None:
    game = SatellitesGame(headless=True)
    game.grid = {
        (4, 5): {"owner": 0, "type": "tank", "count": 3},
        (4, 6): {"owner": 1, "type": "bot", "count": 2},
    }

    assert game.get_player_unit_count(0) == 3
    assert game.get_player_unit_count(1) == 2


def test_clone_is_independent() -> None:
    game = SatellitesGame(headless=True)
    cloned = game.clone()

    cloned.grid[(0, 3)]["count"] += 5
    cloned.scores[0] = 7
    cloned.artefacts.pop()
    cloned.satellites[0]["charges"] += 1

    assert game.grid[(0, 3)]["count"] != cloned.grid[(0, 3)]["count"]
    assert game.scores[0] != cloned.scores[0]
    assert len(game.artefacts) != len(cloned.artefacts)
    assert game.satellites[0]["charges"] != cloned.satellites[0]["charges"]


def test_apply_add_with_undo_roundtrip() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    before = game.clone()

    success, token, _ = game.apply_action_with_undo(("add", 4, 5))
    assert success is True

    game.undo_action(token)

    assert game.grid == before.grid
    assert game.artefacts == before.artefacts
    assert game.scores == before.scores
    assert game.turn == before.turn
    assert game.state == before.state
    assert game.actions_remaining == before.actions_remaining


def test_apply_move_with_undo_roundtrip() -> None:
    game = SatellitesGame(headless=True)
    game.grid = {
        (4, 4): {"owner": 0, "type": "tank", "count": 2},
        (4, 5): {"owner": 1, "type": "bot", "count": 1},
    }
    game.turn = 0
    game.actions_remaining = 1
    before = game.clone()

    success, token, aux = game.apply_action_with_undo(("move", (4, 4), (4, 5), 2))
    assert success is True
    assert aux == (1, 0)

    game.undo_action(token)

    assert game.grid == before.grid
    assert game.artefacts == before.artefacts
    assert game.scores == before.scores
    assert game.turn == before.turn
    assert game.state == before.state
    assert game.actions_remaining == before.actions_remaining


def test_legal_actions_choose_satellite_and_direction() -> None:
    game = SatellitesGame(headless=True)
    sat_actions = game.legal_actions()
    assert len(sat_actions) == 4
    assert all(a[0] == "select_satellite" for a in sat_actions)

    ok = game.apply_action(sat_actions[0])
    assert ok is True
    assert game.state == "CHOOSE_DIRECTION"

    dir_actions = game.legal_actions()
    assert dir_actions == [("set_direction", False), ("set_direction", True)]


def test_legal_actions_perform_add_only_returns_valid_moves() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    actions = game.legal_actions()

    assert actions
    assert all(a[0] == "add" for a in actions)
    assert ("add", 8, 3) not in actions  # opponent start
    assert ("add", 4, 4) not in actions  # artefact


def test_satellites_adapter_smoke() -> None:
    game = SatellitesGame(headless=True)
    adapter = SatellitesAdapter()
    actions = adapter.legal_actions(game)
    next_state = adapter.apply_action(game.clone(), actions[0])

    assert isinstance(actions, list)
    assert next_state is not None


def test_mcts_smoke_selects_legal_action() -> None:
    game = SatellitesGame(headless=True)
    adapter = SatellitesAdapter()
    mcts = MCTS(adapter, iterations=10, rollout_depth=6, seed=1)
    action, _ = mcts.select_action(game)
    assert action in game.legal_actions()


def test_adapter_weight_save_load_roundtrip(tmp_path) -> None:
    adapter = SatellitesAdapter()
    adapter.set_weight("move_bot_capture", 9.25)
    path = tmp_path / "weights.json"
    adapter.save_weights(str(path))

    adapter2 = SatellitesAdapter()
    adapter2.load_weights(str(path))
    assert abs(adapter2.get_weights()["move_bot_capture"] - 9.25) < 1e-9


def test_adapter_is_heuristic_neutral() -> None:
    game = SatellitesGame(headless=True)
    adapter = SatellitesAdapter()

    action = game.legal_actions()[0]
    assert adapter.evaluate(game, 0) == 0.0
    assert adapter.action_prior(game, action, 0) == 0.0
    assert adapter.tactical_priority(game, action, 0) == 0


def test_state_key_changes_with_state_mutation() -> None:
    game = SatellitesGame(headless=True)
    adapter = SatellitesAdapter()
    k0 = adapter.state_key(game)
    game.scores[0] += 1
    k1 = adapter.state_key(game)
    assert k0 != k1
