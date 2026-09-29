"""Base classes for value functions.

This module defines the abstract base classes for value functions, following
the pattern established by BaseModel and BaseModelConfig in the models module.

Value functions can be configured to be action-conditioned (Q-function) or not
(V-function) using the same base class hierarchy.
"""

import abc
import dataclasses

from flax import struct
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at


def _convert_images_to_float(images: dict) -> dict:
    """Convert uint8 images to float32 in [-1, 1] range."""
    result = {}
    for key, img in images.items():
        img_arr = jnp.asarray(img)
        if img_arr.dtype == np.uint8:
            result[key] = img_arr.astype(jnp.float32) / 127.5 - 1.0
        else:
            result[key] = img_arr
    return result


def _extract_observations_from_batch(batch: dict) -> tuple[_model.Observation, _model.Observation]:
    """Extract observation and next_observation from a batch dictionary."""
    images = _convert_images_to_float(batch.get("image", {}))
    image_masks = {k: jnp.asarray(v) for k, v in batch.get("image_mask", {}).items()}
    next_images = _convert_images_to_float(batch.get("next_image", {}))
    next_image_masks = {k: jnp.asarray(v) for k, v in batch.get("next_image_mask", {}).items()}
    subtask_start_index = batch.get("subtask_start_index")
    next_subtask_start_index = batch.get("next_subtask_start_index", subtask_start_index)
    subtask_end_index = batch.get("subtask_end_index")
    next_subtask_end_index = batch.get("next_subtask_end_index", subtask_end_index)
    subtask_id = batch.get("subtask_id")
    next_subtask_id = batch.get("next_subtask_id", subtask_id)

    observation = _model.Observation(
        images = images,
        image_masks = image_masks,
        state = jnp.asarray(batch["state"]),
        tokenized_prompt = batch.get("tokenized_prompt"),
        tokenized_prompt_mask = batch.get("tokenized_prompt_mask"),
        action_mask = batch.get("action_mask"),
        subtask_start_index = jnp.asarray(subtask_start_index) if subtask_start_index is not None else None,
        subtask_end_index = jnp.asarray(subtask_end_index) if subtask_end_index is not None else None,
        subtask_id = jnp.asarray(subtask_id) if subtask_id is not None else None,
    )
    next_observation = _model.Observation(
        images = next_images,
        image_masks = next_image_masks,
        state = jnp.asarray(batch["next_state"]),
        tokenized_prompt = batch.get("next_tokenized_prompt", batch.get("tokenized_prompt")),
        tokenized_prompt_mask = batch.get("next_tokenized_prompt_mask", batch.get("tokenized_prompt_mask")),
        action_mask = batch.get("next_action_mask"),
        subtask_start_index = jnp.asarray(next_subtask_start_index) if next_subtask_start_index is not None else None,
        subtask_end_index = jnp.asarray(next_subtask_end_index) if next_subtask_end_index is not None else None,
        subtask_id = jnp.asarray(next_subtask_id) if next_subtask_id is not None else None,
    )
    return observation, next_observation


def masked_mean(values: at.Array, mask: at.Bool[at.Array, "..."] | None) -> at.Array:
    """Mean of ``values`` over the entries where ``mask`` is True.

    ``mask`` None means every entry counts. A mask with no True entry yields NaN rather
    than a made-up zero, so an all-invalid batch is visible in the logs.
    """
    if mask is None:
        return jnp.mean(values)
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.sum(mask)


@struct.dataclass
class Transition:
    """A transition (o, a, r, o', a') plus MC return for value function training.

    Different algorithms compute targets differently:
    - MC: target = mc_return
    - TD(0): target = reward + gamma * V(next_observation) * (1 - termination)
    - SARSA: target = reward + gamma * Q(next_observation, next_action) * (1 - termination)

    Note: For bootstrapping, only mask with termination, not truncation.
    Truncated states should still bootstrap to avoid bias.
    """

    observation: _model.Observation
    action: at.Float[at.Array, "*b action_dim"] | None = None
    reward: at.Float[at.Array, "*b"] | None = None
    next_observation: _model.Observation | None = None
    next_action: at.Float[at.Array, "*b action_dim"] | None = None
    mc_return: at.Float[at.Array, "*b"] | None = None
    # True where mc_return is a trustworthy target: False on frames whose subtask never
    # reached its success state (partial demos), where the countdown-based return is
    # fiction. None means every return is valid. Only MC diagnostics honour it so far.
    mc_return_mask: at.Bool[at.Array, "*b"] | None = None
    termination: at.Bool[at.Array, "*b"] | None = None
    truncation: at.Bool[at.Array, "*b"] | None = None
    td_discount: at.Float[at.Array, "*b"] | None = None
    # Pre-computed counterfactual actions for best-of-n evaluation at current state
    # Shape: [batch, num_samples, action_horizon, action_dim]
    counterfactual_actions: at.Float[at.Array, "*b k ah ad"] | None = None
    # Pre-computed counterfactual next actions for best-of-n TD backup
    # Shape: [batch, num_samples, action_horizon, action_dim]
    counterfactual_next_actions: at.Float[at.Array, "*b k ah ad"] | None = None

    @classmethod
    def from_batch(cls, batch: dict) -> "Transition":
        """Create a Transition from a batch dictionary.

        Expects batch to contain:
        - state, next_state: Low-dim state observations
        - action, next_action: Actions
        - reward, mc_return: Reward signals
        - termination, truncation: Episode boundary flags
        - tokenized_prompt, tokenized_prompt_mask: Optional text prompts
        - (optional) counterfactual_actions / counterfactual_next_actions: Pre-computed
          counterfactual actions for BestOfN at the current / next state
        """
        observation, next_observation = _extract_observations_from_batch(batch)
        counterfactual_actions = batch.get("counterfactual_actions")
        if counterfactual_actions is not None:
            counterfactual_actions = jnp.asarray(counterfactual_actions)
        counterfactual_next_actions = batch.get("counterfactual_next_actions")
        if counterfactual_next_actions is not None:
            counterfactual_next_actions = jnp.asarray(counterfactual_next_actions)

        return cls(
            observation=observation,
            action=jnp.asarray(batch["actions"]),
            reward=jnp.asarray(batch["reward"]),
            next_observation=next_observation,
            next_action=jnp.asarray(batch["next_actions"]),
            mc_return=jnp.asarray(batch["mc_return"]),
            mc_return_mask=jnp.asarray(batch["mc_return_mask"]) if "mc_return_mask" in batch else None,
            termination=jnp.asarray(batch["termination"]),
            truncation=jnp.asarray(batch["truncation"]),
            td_discount = jnp.asarray(batch["td_discount"]) if "td_discount" in batch else None,
            counterfactual_actions=counterfactual_actions,
            counterfactual_next_actions=counterfactual_next_actions,
        )


@dataclasses.dataclass(frozen=True)
class BaseValueFunctionConfig(abc.ABC):
    """Configuration shared by all value functions.

    Specific value function implementations should inherit from this class
    and implement the `create` method to instantiate the corresponding model.

    Value functions can be action-conditioned (Q-function) or not (V-function)
    based on the implementation's configuration.
    """

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseValueFunction":
        """Create a new value function, initializing parameters."""

    @abc.abstractmethod
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        """Returns the input specification for the value function.

        For V(s): returns (observation_spec, target_spec)
        For Q(s,a): returns (observation_spec, action_spec, target_spec)
        """


@dataclasses.dataclass
class BaseValueFunction(nnx.Module, abc.ABC):
    """Base class for value function implementations.

    Value functions estimate expected returns. They can be:
    - V(s): State-only value functions
    - Q(s,a): Action-conditioned value functions (Q-functions)

    The action parameter is optional to support both cases with the same interface.
    """

    @abc.abstractmethod
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute the value for each observation (and optionally action) in the batch.

        Args:
            observation: Observation containing state (and optionally images, prompt).
            action: Actions of shape [batch, action_horizon, action_dim].
                    Required for action-conditioned value functions.

        Returns:
            Estimated values of shape [batch].
        """

    @abc.abstractmethod
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute the target value for each observation (and optionally action).

        Implementations that do not maintain target networks should return
        `compute_value` outputs.
        """

    @abc.abstractmethod
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute the loss for value function training.

        Each algorithm implementation determines how to compute targets:
        - MC: uses transition.mc_return directly
        - TD: computes target from reward + gamma * V(next_state)
        - SARSA: computes target from reward + gamma * Q(next_state, next_action)
        - SAC: computes soft Bellman target with entropy bonus

        Args:
            transition: Full SARSA transition with (s, a, r, s', a', mc_return).
            train: Whether in training mode.
            rng: Random key for algorithms that need stochasticity (e.g., SAC).
            policy: Optional policy for algorithms that need action sampling (e.g., CQL).

        Returns:
            Tuple of:
            - Per-sample loss of shape [batch]
            - Dict of additional info to log (e.g., predicted values, TD errors)
        """

    def post_step_update(self) -> None:
        """Called after each training step.

        Override this method for operations that should happen after each
        gradient update, such as target network updates (Polyak averaging).

        Default implementation does nothing.
        """


