import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, _model.wrap_observation_as_transition(obs), num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, _model.wrap_observation_as_transition(obs), num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


# Downloads the 11 GB pi0_base checkpoint and materialises the full model in host memory;
# too heavy for CI runners and laptops, so it is opt-in.
@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, _model.wrap_observation_as_transition(obs), num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_observation_subtask_id_round_trip():
    import numpy as np

    obs = _model.Observation.from_dict({"state": np.ones((2, 3), dtype = np.float32), "subtask_id": np.array([1, 2])})
    assert obs.subtask_id.tolist() == [1, 2]
    assert obs.to_dict()["subtask_id"].tolist() == [1, 2]
    assert _model.Observation.from_dict({"state": np.ones((2, 3), dtype = np.float32)}).subtask_id is None

    preprocessed = _model.preprocess_observation(
        None,
        _model.Observation(
            images = {key: jax.numpy.zeros((2, 8, 8, 3)) for key in _model.IMAGE_KEYS},
            image_masks = {},
            state = jax.numpy.ones((2, 3)),
            subtask_id = jax.numpy.array([0, 1]),
        ),
        image_resolution = (8, 8),
    )
    assert preprocessed.subtask_id.tolist() == [0, 1]
