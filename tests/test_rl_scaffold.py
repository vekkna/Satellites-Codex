import numpy as np
import pytest

from agents.alpha_mcts import AlphaMCTS
from engine import SatellitesGame
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder

torch = pytest.importorskip("torch")
from rl.model import SatellitesPolicyValueNet
from rl.selfplay import adjudicate_winner, run_selfplay_game, run_selfplay_games_parallel


def test_action_space_covers_legal_actions() -> None:
    game = SatellitesGame(headless=True)
    action_space = GlobalActionSpace(game)
    legal = game.legal_actions()
    assert legal
    for action in legal:
        idx = action_space.to_index(action)
        assert action_space.from_index(idx) == action


def test_encoder_shape_is_stable() -> None:
    game = SatellitesGame(headless=True)
    enc = FeatureEncoder(game)
    obs = enc.encode(game)
    assert obs.shape == (enc.feature_dim,)
    assert obs.dtype == np.float32


def test_model_forward_shapes() -> None:
    game = SatellitesGame(headless=True)
    action_space = GlobalActionSpace(game)
    enc = FeatureEncoder(game)
    model = SatellitesPolicyValueNet(enc.feature_dim, action_space.size)
    x = torch.from_numpy(enc.encode(game)).float().unsqueeze(0)
    logits, value = model(x)
    assert logits.shape == (1, action_space.size)
    assert value.shape == (1,)


def test_alpha_mcts_and_selfplay_smoke() -> None:
    game = SatellitesGame(headless=True)
    action_space = GlobalActionSpace(game)
    enc = FeatureEncoder(game)
    model = SatellitesPolicyValueNet(enc.feature_dim, action_space.size)
    mcts = AlphaMCTS(model, action_space, enc, simulations=8, seed=1)

    action, info = mcts.select_action(game, temperature=1.0)
    assert action in game.legal_actions()
    assert "policy" in info
    assert info["policy"].shape == (action_space.size,)

    examples = run_selfplay_game(mcts, enc, max_steps=8)
    assert examples
    assert examples[0].obs.shape == (enc.feature_dim,)
    assert examples[0].policy.shape == (action_space.size,)
    assert -1.0 <= examples[0].value <= 1.0


def test_parallel_selfplay_smoke() -> None:
    game = SatellitesGame(headless=True)
    action_space = GlobalActionSpace(game)
    enc = FeatureEncoder(game)
    model = SatellitesPolicyValueNet(enc.feature_dim, action_space.size)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    examples = run_selfplay_games_parallel(
        games=2,
        input_dim=enc.feature_dim,
        action_dim=action_space.size,
        model_state_dict=state,
        simulations=4,
        device="cpu",
        max_workers=2,
        max_steps=8,
        worker_torch_threads=1,
    )
    assert examples
    assert examples[0].obs.shape == (enc.feature_dim,)


def test_adjudicate_winner_prefers_score_then_tanks_then_bots() -> None:
    game = SatellitesGame(headless=True)
    game.winner = None

    game.scores = [4, 3]
    assert adjudicate_winner(game) == 0

    game.scores = [2, 2]
    game.grid = {}
    game.add_unit(0, 3, 0, "tank", 3)
    game.add_unit(8, 4, 1, "tank", 2)
    assert adjudicate_winner(game) == 0

    game.grid = {}
    game.add_unit(0, 3, 0, "tank", 2)
    game.add_unit(8, 4, 1, "tank", 2)
    game.add_unit(0, 4, 0, "bot", 3)
    game.add_unit(8, 3, 1, "bot", 1)
    assert adjudicate_winner(game) == 0


def test_selfplay_handles_no_legal_action_root_error(monkeypatch) -> None:
    game = SatellitesGame(headless=True)
    action_space = GlobalActionSpace(game)
    enc = FeatureEncoder(game)
    model = SatellitesPolicyValueNet(enc.feature_dim, action_space.size)
    mcts = AlphaMCTS(model, action_space, enc, simulations=4, seed=1)

    def _boom(*args, **kwargs):
        raise ValueError("No legal actions from root state.")

    monkeypatch.setattr(mcts, "select_action", _boom)
    examples = run_selfplay_game(mcts, enc, max_steps=8)
    assert examples == []
