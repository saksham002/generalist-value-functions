"""Tests for ActionBounds class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.shared.action_bounds import ActionBounds


class TestActionBoundsBasic:
    def test_create_with_tuples(self):
        bounds = ActionBounds(low=(-1.0, -2.0), high=(1.0, 2.0), is_normalized=True)

        assert bounds.low == (-1.0, -2.0)
        assert bounds.high == (1.0, 2.0)
        assert bounds.is_normalized is True
        assert bounds.action_dim == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ActionBounds(low=(-1.0,), high=(1.0, 2.0))

    def test_get_low_array(self):
        bounds = ActionBounds(low=(-1.0, -2.0, -3.0), high=(1.0, 2.0, 3.0))

        low_arr = bounds.get_low_array()

        assert low_arr.shape == (3,)
        np.testing.assert_allclose(low_arr, [-1.0, -2.0, -3.0])

    def test_get_high_array(self):
        bounds = ActionBounds(low=(-1.0, -2.0, -3.0), high=(1.0, 2.0, 3.0))

        high_arr = bounds.get_high_array()

        assert high_arr.shape == (3,)
        np.testing.assert_allclose(high_arr, [1.0, 2.0, 3.0])


class TestActionBoundsClip:
    def test_clip_within_bounds(self):
        bounds = ActionBounds(low=(-1.0, -1.0), high=(1.0, 1.0))
        actions = jnp.array([[0.5, -0.5], [0.0, 0.0]])

        clipped = bounds.clip(actions)

        np.testing.assert_allclose(clipped, actions)

    def test_clip_outside_bounds(self):
        bounds = ActionBounds(low=(-1.0, -1.0), high=(1.0, 1.0))
        actions = jnp.array([[2.0, -3.0], [0.5, 5.0]])

        clipped = bounds.clip(actions)

        expected = jnp.array([[1.0, -1.0], [0.5, 1.0]])
        np.testing.assert_allclose(clipped, expected)

    def test_clip_per_dimension_bounds(self):
        bounds = ActionBounds(low=(-1.0, -2.0, -3.0), high=(1.0, 2.0, 3.0))
        actions = jnp.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])

        clipped = bounds.clip(actions)

        expected = jnp.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(clipped, expected)

    def test_clip_broadcasts_over_batch_dims(self):
        bounds = ActionBounds(low=(-1.0, -2.0), high=(1.0, 2.0))
        # Shape: [batch=3, time=2, action_dim=2]
        actions = jnp.array(
            [
                [[5.0, 5.0], [0.0, 0.0]],
                [[-5.0, -5.0], [0.5, 1.5]],
                [[0.0, 3.0], [-0.5, -3.0]],
            ]
        )

        clipped = bounds.clip(actions)

        expected = jnp.array(
            [
                [[1.0, 2.0], [0.0, 0.0]],
                [[-1.0, -2.0], [0.5, 1.5]],
                [[0.0, 2.0], [-0.5, -2.0]],
            ]
        )
        np.testing.assert_allclose(clipped, expected)


class TestActionBoundsSampleUniform:
    def test_sample_shape(self):
        bounds = ActionBounds(low=(-1.0, -2.0, -3.0), high=(1.0, 2.0, 3.0))
        rng = jax.random.key(42)

        samples = bounds.sample_uniform(rng, shape=(10,))

        assert samples.shape == (10, 3)

    def test_sample_within_bounds(self):
        bounds = ActionBounds(low=(-1.0, -2.0), high=(1.0, 2.0))
        rng = jax.random.key(42)

        samples = bounds.sample_uniform(rng, shape=(100,))

        assert jnp.all(samples[:, 0] >= -1.0)
        assert jnp.all(samples[:, 0] <= 1.0)
        assert jnp.all(samples[:, 1] >= -2.0)
        assert jnp.all(samples[:, 1] <= 2.0)

    def test_sample_multi_dim_shape(self):
        bounds = ActionBounds(low=(-1.0, -1.0), high=(1.0, 1.0))
        rng = jax.random.key(42)

        # Shape: [batch, time, action_dim]
        samples = bounds.sample_uniform(rng, shape=(5, 3))

        assert samples.shape == (5, 3, 2)

    def test_sample_deterministic_with_key(self):
        bounds = ActionBounds(low=(-1.0,), high=(1.0,))
        rng1 = jax.random.key(42)
        rng2 = jax.random.key(42)

        samples1 = bounds.sample_uniform(rng1, shape=(10,))
        samples2 = bounds.sample_uniform(rng2, shape=(10,))

        np.testing.assert_allclose(samples1, samples2)


class TestActionBoundsLogProb:
    def test_uniform_log_prob(self):
        bounds = ActionBounds(low=(-1.0,), high=(1.0,))
        actions = jnp.ones((5, 1))

        log_prob = bounds.compute_uniform_log_prob(actions)

        # log(1 / (1 - (-1))) = log(1/2) = -log(2)
        expected = -jnp.log(2.0)
        np.testing.assert_allclose(log_prob, jnp.full((5,), expected))

    def test_uniform_log_prob_multi_dim(self):
        bounds = ActionBounds(low=(-1.0, -2.0), high=(1.0, 2.0))
        actions = jnp.ones((3, 2))

        log_prob = bounds.compute_uniform_log_prob(actions)

        # log(1 / (2 * 4)) = -log(8)
        expected = -jnp.log(2.0) - jnp.log(4.0)
        np.testing.assert_allclose(log_prob, jnp.full((3,), expected))


class TestActionBoundsFromArrays:
    def test_from_arrays_basic(self):
        low = np.array([-1.0, -2.0, -3.0])
        high = np.array([1.0, 2.0, 3.0])

        bounds = ActionBounds.from_arrays(low, high, is_normalized=False)

        assert bounds.low == (-1.0, -2.0, -3.0)
        assert bounds.high == (1.0, 2.0, 3.0)
        assert bounds.is_normalized is False
        assert bounds.action_dim == 3

    def test_from_arrays_flattens(self):
        low = np.array([[-1.0], [-2.0]])
        high = np.array([[1.0], [2.0]])

        bounds = ActionBounds.from_arrays(low, high, is_normalized=True)

        assert bounds.low == (-1.0, -2.0)
        assert bounds.high == (1.0, 2.0)


class TestActionBoundsFromUniform:
    def test_from_uniform_basic(self):
        bounds = ActionBounds.from_uniform(-1.0, 1.0, action_dim=4, is_normalized=True)

        assert bounds.low == (-1.0, -1.0, -1.0, -1.0)
        assert bounds.high == (1.0, 1.0, 1.0, 1.0)
        assert bounds.is_normalized is True
        assert bounds.action_dim == 4

    def test_from_uniform_asymmetric(self):
        bounds = ActionBounds.from_uniform(0.0, 10.0, action_dim=2, is_normalized=False)

        assert bounds.low == (0.0, 0.0)
        assert bounds.high == (10.0, 10.0)
        assert bounds.is_normalized is False


class TestActionBoundsToNormalized:
    def test_to_normalized_basic(self):
        bounds = ActionBounds(low=(-1.0, -2.0), high=(1.0, 2.0), is_normalized=False)
        mean = np.array([0.0, 0.0])
        std = np.array([0.5, 1.0])

        normalized = bounds.to_normalized(mean, std)

        # normalized = (raw - mean) / (std + epsilon)
        # With epsilon=1e-6, results are very close to expected
        np.testing.assert_allclose(normalized.low, (-2.0, -2.0), rtol=1e-5)
        np.testing.assert_allclose(normalized.high, (2.0, 2.0), rtol=1e-5)
        assert normalized.is_normalized is True

    def test_to_normalized_with_nonzero_mean(self):
        bounds = ActionBounds(low=(0.0, 0.0), high=(10.0, 20.0), is_normalized=False)
        mean = np.array([5.0, 10.0])
        std = np.array([5.0, 10.0])

        normalized = bounds.to_normalized(mean, std)

        # With epsilon=1e-6, results are very close to expected
        np.testing.assert_allclose(normalized.low, (-1.0, -1.0), rtol=1e-5)
        np.testing.assert_allclose(normalized.high, (1.0, 1.0), rtol=1e-5)

    def test_to_normalized_handles_flipped_bounds(self):
        """Test that bounds are reordered if normalization flips them."""
        bounds = ActionBounds(low=(-1.0,), high=(1.0,), is_normalized=False)
        mean = np.array([0.0])
        std = np.array([-1.0])  # Negative std would flip bounds

        normalized = bounds.to_normalized(mean, std)

        # With negative std, low and high would flip
        # The method ensures low <= high
        assert normalized.low[0] <= normalized.high[0]


class TestActionBoundsFrozen:
    def test_immutable(self):
        bounds = ActionBounds(low=(-1.0,), high=(1.0,))

        with pytest.raises(AttributeError):
            bounds.low = (0.0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
