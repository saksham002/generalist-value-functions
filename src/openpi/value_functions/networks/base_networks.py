"""Base class for value network architectures."""

import abc

import flax.nnx as nnx

from openpi.models import model as _model
from openpi.shared import array_typing as at


class BaseValueNetwork(nnx.Module, abc.ABC):
    """Abstract base class for value network architectures.

    Networks compute features from observations (and optionally actions).
    The features are then passed to a value head for value/loss computation.
    """

    action_conditioned: bool  # Subclasses must set this as instance attribute

    @abc.abstractmethod
    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "*b feature_dim"]:
        """Compute features from observation (and optionally action).

        Args:
            observation: Observation containing state (and optionally images, prompt).
            action: Actions of shape [batch, action_horizon, action_dim].
                    Required if action_conditioned=True.
            rng: Optional random key for stochastic operations (e.g., image augmentation).

        Returns:
            Features of shape [batch, feature_dim].
        """

    @property
    @abc.abstractmethod
    def feature_dim(self) -> int:
        """Return the feature dimension output by this network."""
