import dataclasses

"""Tests for new value function architecture."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import best_of_n as _best_of_n
from openpi.models.model import IMAGE_KEYS
from openpi.models.model import Observation
from openpi.value_functions.base_value_functions import Transition
from openpi.value_functions.heads import RegressionHeadConfig
from openpi.value_functions.networks.base_networks import BaseValueNetwork
from openpi.value_functions.networks.paligemma import PaliGemmaNetworkConfig
from openpi.value_functions.value_function import MCValueFunctionConfig
from openpi.value_functions.value_function import SARSAValueFunctionConfig


def make_observation(state: jnp.ndarray) -> Observation:
    return Observation(images={}, image_masks={}, state=state)


def make_transition(batch_size: int, state_dim: int, action_dim: int = 4) -> Transition:
    observation = make_observation(jnp.ones((batch_size, state_dim)))
    next_observation = make_observation(jnp.ones((batch_size, state_dim)))
    return Transition(
        observation=observation,
        action=jnp.ones((batch_size, action_dim)),
        reward=jnp.zeros(batch_size),
        next_observation=next_observation,
        next_action=jnp.ones((batch_size, action_dim)),
        mc_return=jnp.ones(batch_size) * 0.5,
        termination=jnp.zeros(batch_size, dtype=bool),
        truncation=jnp.zeros(batch_size, dtype=bool),
        td_discount=None,
    )


@dataclasses.dataclass(frozen = True)
class TinyNetworkConfig:
    """State-only MLP standing in for the image backbones in these unit tests."""

    state_dim: int
    action_conditioned: bool = False
    action_dim: int = 4
    hidden_dim: int = 32

    def create(self, rng: jax.Array) -> "TinyNetwork":
        return TinyNetwork(self, rngs = nnx.Rngs(rng))


class TinyNetwork(BaseValueNetwork):
    def __init__(self, config: TinyNetworkConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.action_conditioned = config.action_conditioned
        input_dim = config.state_dim + (config.action_dim if config.action_conditioned else 0)
        self.hidden = nnx.Linear(input_dim, config.hidden_dim, rngs = rngs)
        self._feature_dim = config.hidden_dim

    def compute_features(self, observation: Observation, action = None, *, rng = None, **kwargs):
        del rng, kwargs
        inputs = observation.state
        if self.action_conditioned:
            batch_size = action.shape[0]
            inputs = jnp.concatenate([inputs, action.reshape(batch_size, action.size // batch_size)], axis = -1)
        return jax.nn.relu(self.hidden(inputs))

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class AuxNetwork(BaseValueNetwork):
    def __init__(self, feature_dim: int = 1, aux_tokens: int = 0):
        super().__init__()
        self.action_conditioned = True
        self._feature_dim = feature_dim
        self._aux_tokens = aux_tokens

    def compute_features(self, observation: Observation, action = None, *, rng = None):
        batch_size = observation.state.shape[0]
        features = jnp.ones((batch_size, self._feature_dim))
        if rng is not None and self._aux_tokens:
            return features, {
                "next_token_embeddings": jnp.zeros((batch_size, self._aux_tokens, 1)),
                "next_token_targets": jnp.zeros((batch_size, self._aux_tokens), dtype = jnp.int32),
                "next_token_mask": jnp.ones((batch_size, self._aux_tokens), dtype = jnp.bool_),
            }
        return features

    def decode(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        zeros = jnp.zeros(embeddings.shape[:-1] + (1,))
        return jnp.concatenate([zeros, zeros], axis = -1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class TestHeads:
    def test_regression_head(self):
        config = RegressionHeadConfig()
        head = config.create(64, jax.random.key(0))

        features = jnp.ones((4, 64))
        values = head(features)

        assert values.shape == (4,)


class TestMCValueFunction:
    def test_regression_head(self):
        config = MCValueFunctionConfig(
            network_config=TinyNetworkConfig(state_dim=10),
            head_config=RegressionHeadConfig(),
        )
        model = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((4, 10)))
        values = model.compute_value(obs)
        assert values.shape == (4,)

        transition = make_transition(4, 10)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert "predicted_value" in info


class TestSARSAValueFunction:
    def test_with_target_network(self):
        config = SARSAValueFunctionConfig(
            network_config=TinyNetworkConfig(state_dim=10, action_conditioned=True, action_dim=4),
            head_config=RegressionHeadConfig(),
            discount=0.99,
            tau=0.005,
        )
        model = config.create(jax.random.key(0))

        transition = make_transition(4, 10, action_dim=4)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (4,)
        assert "next_value" in info

        # Test target network update
        model.post_step_update()

    def test_aux_next_token_loss_is_added(self):
        transition = make_transition(4, 10, action_dim = 4)
        head = RegressionHeadConfig().create(1, jax.random.key(0))
        target_head = RegressionHeadConfig().create(1, jax.random.key(1))
        model = SARSAValueFunctionConfig(
            network_config=TinyNetworkConfig(state_dim=10, action_conditioned=True, action_dim=4),
            head_config=RegressionHeadConfig(),
            next_token_loss_weight=0.5,
        ).create(jax.random.key(2))
        model.network = AuxNetwork(aux_tokens = 2)
        model.target_network = AuxNetwork(aux_tokens = 0)
        model.head = head
        model.target_head = target_head

        model.next_token_loss_weight = 0.0
        baseline_loss, _ = model.compute_loss(transition, rng = jax.random.key(3))
        model.next_token_loss_weight = 0.5
        loss, info = model.compute_loss(transition, rng = jax.random.key(3))

        assert "next_token_loss" in info
        expected_aux_loss = np.full((4,), np.log(2.0))
        np.testing.assert_allclose(info["next_token_loss"], expected_aux_loss)
        np.testing.assert_allclose(loss - baseline_loss, 0.5 * expected_aux_loss, atol = 1e-5)


@pytest.mark.manual
def _make_paligemma_observation(
    batch_size: int,
    state_dim: int,
    action_horizon: int,
    max_token_len: int,
    image_size: tuple[int, int],
) -> Observation:
    images = {
        key: jnp.linspace(
            -1.0,
            1.0,
            num = batch_size * image_size[0] * image_size[1] * 3,
            dtype = jnp.float32,
        ).reshape(batch_size, image_size[0], image_size[1], 3)
        for key in IMAGE_KEYS
    }
    image_masks = {
        key: jnp.ones((batch_size,), dtype = jnp.bool_)
        for key in IMAGE_KEYS
    }
    tokenized_prompt = (jnp.arange(batch_size * max_token_len, dtype = jnp.int32).reshape(batch_size, max_token_len) % 128)
    tokenized_prompt_mask = jnp.ones((batch_size, max_token_len), dtype = jnp.bool_)
    state = jnp.linspace(-0.5, 0.5, num = batch_size * state_dim, dtype = jnp.float32).reshape(batch_size, state_dim)
    action_mask = jnp.ones((batch_size, action_horizon), dtype = jnp.bool_)
    return Observation(
        images = images,
        image_masks = image_masks,
        state = state,
        tokenized_prompt = tokenized_prompt,
        tokenized_prompt_mask = tokenized_prompt_mask,
        action_mask = action_mask,
    )


@pytest.mark.manual


def test_paligemma_prefix_cache_matches_uncached_features_across_samples():
    batch_size = 2
    num_samples = 3
    state_dim = 14
    action_dim = 14
    action_horizon = 4
    max_token_len = 16
    image_size = (224, 224)

    config = PaliGemmaNetworkConfig(
        state_dim = state_dim,
        num_cameras = len(IMAGE_KEYS),
        image_size = image_size,
        max_token_len = max_token_len,
        action_dim = action_dim,
        dtype = "float32",
    )
    network = config.create(jax.random.key(86), action_horizon = action_horizon)

    observation = _make_paligemma_observation(
        batch_size = batch_size,
        state_dim = state_dim,
        action_horizon = action_horizon,
        max_token_len = max_token_len,
        image_size = image_size,
    )
    expanded_observation = _best_of_n.expand_observation(observation, num_samples)
    flat_actions = jnp.linspace(
        -1.0,
        1.0,
        num = batch_size * num_samples * action_horizon * action_dim,
        dtype = jnp.float32,
    ).reshape(batch_size * num_samples, action_horizon, action_dim)

    uncached_result = network.compute_features(expanded_observation, flat_actions)
    uncached_features = uncached_result[0] if isinstance(uncached_result, tuple) else uncached_result

    raw_kv_cache, raw_prefix_mask, raw_subtask_mask = network.compute_prefix_cache(observation)

    assert raw_prefix_mask.shape == (batch_size, len(IMAGE_KEYS) * 256 + max_token_len + 1)
    cache_leaves = jax.tree_util.tree_leaves(raw_kv_cache)
    assert len(cache_leaves) > 0
    for leaf in cache_leaves:
        assert leaf.shape[1] == batch_size

    repeated_kv_cache = jax.tree.map(
        lambda x: jnp.repeat(x, num_samples, axis = 1),
        raw_kv_cache,
    )
    repeated_prefix_mask = jnp.repeat(raw_prefix_mask, num_samples, axis = 0)
    repeated_subtask_mask = None if raw_subtask_mask is None else jnp.repeat(raw_subtask_mask, num_samples, axis = 0)

    cached_features = network.compute_features(
        expanded_observation,
        flat_actions,
        prefix_cache = (repeated_kv_cache, repeated_prefix_mask, repeated_subtask_mask),
    )

    assert repeated_prefix_mask.shape == (batch_size * num_samples, raw_prefix_mask.shape[1])
    cached_cache_leaves = jax.tree_util.tree_leaves(repeated_kv_cache)
    assert len(cached_cache_leaves) == len(cache_leaves)
    for leaf in cached_cache_leaves:
        assert leaf.shape[1] == batch_size * num_samples

    assert cached_features.shape == (batch_size * num_samples, network.feature_dim)
    assert uncached_features.shape == (batch_size * num_samples, network.feature_dim)
    np.testing.assert_allclose(cached_features, uncached_features, atol = 1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
