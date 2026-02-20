from engine import SatellitesGame


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

    ok = game.execute_add(4, 4)

    assert ok is True
    assert game.grid[(4, 4)] == {"owner": 0, "type": "tank", "count": 1}


def test_tank_drop_allows_own_tank_stack() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {(4, 4): {"owner": 0, "type": "tank", "count": 2}}

    ok = game.execute_add(4, 4)

    assert ok is True
    assert game.grid[(4, 4)]["count"] == 3


def test_tank_drop_rejects_opponent_start_hex() -> None:
    game = SatellitesGame(headless=True)
    _prep_add_tank(game, turn=0)
    game.grid = {}

    ok = game.execute_add(8, 3)

    assert ok is False
    assert (8, 3) not in game.grid
    assert "opponent start zone" in game.info_message

