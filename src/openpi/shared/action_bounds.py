"""Action space bounds with normalization tracking.

This module provides the ActionBounds class for representing and manipulating
action space bounds, supporting both normalized and unnormalized action spaces.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ActionBounds:
    """Action space bounds with normalization tracking.

    Stores per-dimension bounds for action spaces and tracks whether the bounds
    are for normalized or unnormalized actions. Provides utilities for clipping
    actions and sampling uniformly within bounds.

    Attributes:
        low: Per-dimension lower bounds as a tuple of floats.
        high: Per-dimension upper bounds as a tuple of floats.
        is_normalized: Whether bounds are for normalized actions (default True).
    """

    low: tuple[float, ...]
    high: tuple[float, ...]
    is_normalized: bool = True

    def __post_init__(self) -> None:
        if len(self.low) != len(self.high):
            raise ValueError(f"low and high must have the same length, got {len(self.low)} and {len(self.high)}")

    @property
    def action_dim(self) -> int:
        """Return the action dimension (inferred from tuple length)."""
        return len(self.low)

    @property
    def normalized(self) -> bool:
        """Alias for is_normalized for backwards compatibility."""
        return self.is_normalized

    def assert_normalized(self) -> None:
        """Assert that bounds are for normalized actions."""
        if not self.is_normalized:
            raise ValueError("ActionBounds must be normalized.")

    def get_low_array(self) -> at.Array:
        """Return lower bounds as a JAX array of shape [action_dim]."""
        return jnp.array(self.low, dtype=jnp.float32)

    def get_high_array(self) -> at.Array:
        """Return upper bounds as a JAX array of shape [action_dim]."""
        return jnp.array(self.high, dtype=jnp.float32)

    def clip(self, actions: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b d"]:
        """Clip actions to bounds, broadcasting over batch dimensions.

        Args:
            actions: Actions with shape [..., action_dim] where action_dim
                must match len(self.low).

        Returns:
            Clipped actions with the same shape as input.
        """
        low = self.get_low_array()
        high = self.get_high_array()
        return jnp.clip(actions, low, high)

    def sample_uniform(
        self,
        rng: at.KeyArrayLike,
        shape: tuple[int, ...],
    ) -> at.Float[at.Array, "*shape d"]:
        """Sample actions uniformly within bounds.

        Args:
            rng: JAX random key.
            shape: Shape of the output excluding the action dimension.
                For example, shape=(batch_size,) returns [batch_size, action_dim].

        Returns:
            Uniformly sampled actions with shape (*shape, action_dim).
        """
        low = self.get_low_array()
        high = self.get_high_array()
        full_shape = (*shape, self.action_dim)
        uniform_samples = jax.random.uniform(rng, shape=full_shape)
        return low + (high - low) * uniform_samples

    def compute_uniform_log_prob(self, actions: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b"]:
        """Compute log probability under uniform distribution over bounds.

        For uniform distribution U(low, high), the log probability is:
        log(1 / prod(high - low)) = -sum(log(high - low))

        This is constant for all actions within bounds.

        Args:
            actions: Actions with shape [..., action_dim]. The values are not
                used since log prob is constant for uniform distribution.

        Returns:
            Log probability with shape [...] (action dimension reduced).
        """
        low = self.get_low_array()
        high = self.get_high_array()
        log_prob_per_dim = -jnp.log(high - low)
        total_log_prob = jnp.sum(log_prob_per_dim)
        batch_shape = actions.shape[:-1]
        return jnp.full(batch_shape, total_log_prob)

    @classmethod
    def from_arrays(
        cls,
        low: np.ndarray,
        high: np.ndarray,
        *,
        is_normalized: bool = False,
    ) -> ActionBounds:
        """Create ActionBounds from numpy arrays.

        Args:
            low: Lower bounds as a 1D numpy array.
            high: Upper bounds as a 1D numpy array.
            is_normalized: Whether bounds are for normalized actions.

        Returns:
            ActionBounds instance with per-dimension bounds.
        """
        low_tuple = tuple(float(x) for x in low.flatten())
        high_tuple = tuple(float(x) for x in high.flatten())
        return cls(low=low_tuple, high=high_tuple, is_normalized=is_normalized)

    @classmethod
    def from_uniform(
        cls,
        low: float,
        high: float,
        action_dim: int,
        *,
        is_normalized: bool = True,
    ) -> ActionBounds:
        """Create ActionBounds with uniform bounds across all dimensions.

        Args:
            low: Lower bound for all dimensions.
            high: Upper bound for all dimensions.
            action_dim: Number of action dimensions.
            is_normalized: Whether bounds are for normalized actions.

        Returns:
            ActionBounds instance with the same bound for each dimension.
        """
        return cls(
            low=tuple(low for _ in range(action_dim)),
            high=tuple(high for _ in range(action_dim)),
            is_normalized=is_normalized,
        )

    def to_normalized(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        epsilon: float = 1e-6,
    ) -> ActionBounds:
        """Transform raw bounds to normalized space using z-score normalization.

        Computes normalized_bounds = (raw_bounds - mean) / (std + epsilon).

        Args:
            mean: Mean values for each dimension.
            std: Standard deviation for each dimension.
            epsilon: Small value added to std for numerical stability.

        Returns:
            New ActionBounds in normalized space with is_normalized=True.
        """
        low = np.array(self.low)
        high = np.array(self.high)
        std_safe = std + epsilon

        normalized_low = (low - mean) / std_safe
        normalized_high = (high - mean) / std_safe

        # For dimensions where std is very small, low and high might flip
        # Ensure low <= high after normalization
        final_low = np.minimum(normalized_low, normalized_high)
        final_high = np.maximum(normalized_low, normalized_high)

        return ActionBounds(
            low=tuple(float(x) for x in final_low),
            high=tuple(float(x) for x in final_high),
            is_normalized=True,
        )
