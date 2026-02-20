import numpy as np
import pytest

from agents.alpha_mcts import AlphaMCTS
from engine import SatellitesGame
from rl.action_space import GlobalActionSpace
from rl.encode import FeatureEncoder

torch = pytest.importorskip("torch")
from rl.model import SatellitesPolicyValueNet
from rl.selfplay import run_selfplay_game


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

