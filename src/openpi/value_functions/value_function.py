"""Unified value function with config hierarchy.

Provides composable value functions using network + head + objective pattern.
Each objective-specific config inherits from the base and adds only the
parameters it needs.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import best_of_n as _best_of_n
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared.action_bounds import ActionBounds
from openpi.value_functions import value_function_objectives as _objectives
from openpi.value_functions.base_value_functions import BaseValueFunction
from openpi.value_functions.base_value_functions import BaseValueFunctionConfig
from openpi.value_functions.base_value_functions import Transition
from openpi.value_functions.heads import HeadConfig
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base_networks import BaseValueNetwork
from openpi.value_functions.networks.paligemma import PaliGemmaNetworkConfig
from openpi.value_functions.networks.resnet import ResNetNetworkConfig

# =============================================================================
# Single-Transition Value Functions
# =============================================================================


# Image-based network configs share the PaliGemma conventions: `create(rng, action_horizon = ...)`
# and a `dtype` field that determines the value function's weight dtype.
_IMAGE_NETWORK_CONFIGS = (PaliGemmaNetworkConfig, ResNetNetworkConfig)


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for value functions.

    Composes a network config with a head config. Subclasses define
    the objective and any objective-specific parameters.
    """

    network_config: PaliGemmaNetworkConfig | ResNetNetworkConfig
    head_config: HeadConfig

    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> ValueFunction:
        raise NotImplementedError("Use a specific config subclass (MCValueFunctionConfig, etc.)")

    @property
    def weight_dtype(self) -> str:
        """Dtype for model weights, derived from the network config."""
        if isinstance(self.network_config, _IMAGE_NETWORK_CONFIGS):
            return self.network_config.dtype
        return "float32"

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        nc = self.network_config
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, nc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size], jnp.float32)
        if self.action_horizon is not None:
            actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, nc.action_dim], jnp.float32)
            return obs, actions, target
        return obs, target


@dataclasses.dataclass(frozen=True)
class MCValueFunctionConfig(ValueFunctionConfig):
    """Monte-Carlo value function config.

    Uses mc_return from transitions as target.
    """

    next_token_loss_weight: float = 0.0

    @override
    def create(self, rng: at.KeyArrayLike) -> MCValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        # PaliGemmaNetworkConfig / ResNetNetworkConfig receive action_horizon at creation time instead of
        # storing it as a config field.
        if isinstance(self.network_config, _IMAGE_NETWORK_CONFIGS):
            network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MCValueFunction(
            network=network,
            head=head,
            next_token_loss_weight=self.next_token_loss_weight,
        )


@dataclasses.dataclass(frozen=True)
class SARSAValueFunctionConfig(ValueFunctionConfig):
    """SARSA value function config with target network."""

    discount: float = 0.99
    tau: float = 0.005
    next_token_loss_weight: float = 0.0
    # Read by init_critic in train_value_function.py to cast params after init.
    # Polyak updates cast the
    # aggregated target back to target_dtype so the dtype is preserved.
    target_dtype: str = "float32"

    @override
    def create(self, rng: at.KeyArrayLike) -> SARSAValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng, 2)

        if isinstance(self.network_config, _IMAGE_NETWORK_CONFIGS):
            network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
            target_network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            network = self.network_config.create(net_rng)
            target_network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_head = self.head_config.create(target_network.feature_dim, head_rng)

        return SARSAValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
            next_token_loss_weight=self.next_token_loss_weight,
        )


class ValueFunction(BaseValueFunction):
    """Base value function class with network + head composition."""

    network: BaseValueNetwork
    head: ValueHead

    def __init__(self, network: BaseValueNetwork, head: ValueHead):
        super().__init__()
        self.network = network
        self.head = head

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"] | tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b _n"]]:
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        out = self.network.compute_features(observation, action, **feature_kwargs)
        if isinstance(out, tuple):
            features, attn_scores = out[0], out[1]
            val = self.head(features)
            return val, attn_scores
        val = self.head(out)
        return val

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """For value functions without target network, return the same as compute_value."""
        result = self.compute_value(
            observation, action,
            prefix_cache = prefix_cache,
        )
        if isinstance(result, tuple):
            return result[0]
        return result


class MCValueFunction(ValueFunction):
    """Monte-Carlo value function."""

    next_token_loss_weight: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        next_token_loss_weight: float = 0.0,
    ):
        super().__init__(network, head)
        self.next_token_loss_weight = next_token_loss_weight

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del policy
        return _objectives.mc_objective(
            self.network,
            self.head,
            transition,
            next_token_loss_weight=self.next_token_loss_weight,
            rng=rng,
        )

    def compute_prefix_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[at.Array, at.Array, at.Array | None]:
        """Expose the network's prefix-cache fast path, as the SARSA and CQL critics do.

        Lets one encoder pass over images + prompt be shared across the N candidate actions
        of a Best-of-N evaluation instead of re-encoding per candidate. No ``use_target``
        argument: a Monte-Carlo critic bootstraps off nothing and so has no target network.
        """
        if not hasattr(self.network, "compute_prefix_cache"):
            raise AttributeError(
                f"{type(self.network).__name__} does not support prefix caching. "
                "Callers should check hasattr before invoking."
            )
        return self.network.compute_prefix_cache(observation)


class SARSAValueFunction(ValueFunction):
    """SARSA value function with target network."""

    target_network: BaseValueNetwork
    target_head: ValueHead
    discount: float
    tau: float
    next_token_loss_weight: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        target_network: BaseValueNetwork,
        target_head: ValueHead,
        discount: float,
        tau: float,
        next_token_loss_weight: float,
    ):
        super().__init__(network, head)
        self.target_network = target_network
        self.target_head = target_head
        self.discount = discount
        self.tau = tau
        self.next_token_loss_weight = next_token_loss_weight

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target value using target network."""
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        target_out = self.target_network.compute_features(observation, action, **feature_kwargs)
        target_features = target_out[0] if isinstance(target_out, tuple) else target_out
        val = self.target_head(target_features)
        return val

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del train, policy
        return _objectives.sarsa_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            discount=self.discount,
            next_token_loss_weight=self.next_token_loss_weight,
            rng=rng,
        )

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_network, self.network, self.tau)
        _polyak_update(self.target_head, self.head, self.tau)

    def compute_prefix_cache(
        self,
        observation: _model.Observation,
        use_target: bool = False,
    ) -> tuple[at.Array, at.Array, at.Array | None]:
        """Wrapper that exposes the underlying network's prefix-cache fast path.

        Used by `BestOfNWrapper.sample_actions` to compute the critic's
        per-prefix KV cache once and reuse it across the N candidate-action
        evaluations (instead of re-encoding images + prompt N times). Mirrors
        the sibling repo's wrapper at value_function.py:393. Returns
        ``(kv_cache, prefix_mask, subtask_mask)``; ``subtask_mask`` is the
        ``[B, prefix_len]`` boolean over prefix columns marking subtask-text
        positions (or None when subtask boundaries weren't supplied).
        """
        network = self.target_network if use_target else self.network
        if not hasattr(network, "compute_prefix_cache"):
            raise AttributeError(
                f"{type(network).__name__} does not support prefix caching. "
                "Callers should check hasattr before invoking."
            )
        return network.compute_prefix_cache(observation)


@dataclasses.dataclass(frozen=True)
class CQLValueFunctionConfig(BaseValueFunctionConfig):
    """CQL config with Q-network and target Q-network.

    Supports any network config implementing the create/feature_dim protocol.
    """

    q_network_config: PaliGemmaNetworkConfig | ResNetNetworkConfig
    q_head_config: HeadConfig

    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    discount: float = 0.99
    tau: float = 0.005
    next_token_loss_weight: float = 0.0

    action_bounds: ActionBounds = dataclasses.field(
        default_factory = lambda: ActionBounds.from_uniform(-1.0, 1.0, action_dim = 1, is_normalized = True)
    )
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    cql_n_actions: int = 4
    cql_action_sample_method: Literal["uniform", "normal"] = "uniform"
    cql_importance_sample: bool = True
    only_use_next_actions_for_cql: bool = False
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf
    use_calql: bool = False
    use_calql_on_random_actions: bool = True

    @property
    def weight_dtype(self) -> str:
        """Dtype for model weights, derived from the network config."""
        if isinstance(self.q_network_config, _IMAGE_NETWORK_CONFIGS):
            return self.q_network_config.dtype
        return "float32"

    @override
    def create(self, rng: at.KeyArrayLike) -> CQLValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng, 2)

        if isinstance(self.q_network_config, _IMAGE_NETWORK_CONFIGS):
            q_network = self.q_network_config.create(net_rng, action_horizon = self.action_horizon)
            target_q_network = self.q_network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            q_network = self.q_network_config.create(net_rng)
            target_q_network = self.q_network_config.create(net_rng)
        q_head = self.q_head_config.create(q_network.feature_dim, head_rng)
        target_q_head = self.q_head_config.create(target_q_network.feature_dim, head_rng)

        return CQLValueFunction(
            q_network=q_network,
            q_head=q_head,
            target_q_network=target_q_network,
            target_q_head=target_q_head,
            discount=self.discount,
            tau=self.tau,
            action_bounds=self.action_bounds,
            cql_alpha=self.cql_alpha,
            cql_temp=self.cql_temp,
            cql_n_actions=self.cql_n_actions,
            cql_action_sample_method=self.cql_action_sample_method,
            cql_importance_sample=self.cql_importance_sample,
            only_use_next_actions_for_cql=self.only_use_next_actions_for_cql,
            cql_max_target_backup=self.cql_max_target_backup,
            cql_clip_diff_min=self.cql_clip_diff_min,
            cql_clip_diff_max=self.cql_clip_diff_max,
            use_calql=self.use_calql,
            use_calql_on_random_actions=self.use_calql_on_random_actions,
            next_token_loss_weight=self.next_token_loss_weight,
        )

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]:
        qc = self.q_network_config
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, qc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size], jnp.float32)
        action_horizon = self.action_horizon if self.action_horizon is not None else 1
        actions = jax.ShapeDtypeStruct([batch_size, action_horizon, qc.action_dim], jnp.float32)
        return obs, actions, target


class CQLValueFunction(BaseValueFunction):
    """CQL Q-function with target network and conservative penalty.

    Policy is not owned by this class. Pass policy to compute_loss().
    """

    q_network: BaseValueNetwork
    q_head: ValueHead
    target_q_network: BaseValueNetwork
    target_q_head: ValueHead
    discount: float
    tau: float
    next_token_loss_weight: float

    action_bounds: ActionBounds
    cql_alpha: float
    cql_temp: float
    cql_n_actions: int
    cql_action_sample_method: Literal["uniform", "normal"]
    cql_importance_sample: bool
    only_use_next_actions_for_cql: bool
    cql_max_target_backup: bool
    cql_clip_diff_min: float
    cql_clip_diff_max: float
    use_calql: bool
    use_calql_on_random_actions: bool

    def __init__(
        self,
        q_network: BaseValueNetwork,
        q_head: ValueHead,
        target_q_network: BaseValueNetwork,
        target_q_head: ValueHead,
        discount: float,
        tau: float,
        *,
        action_bounds: ActionBounds,
        cql_alpha: float,
        cql_temp: float,
        cql_n_actions: int,
        cql_action_sample_method: Literal["uniform", "normal"],
        cql_importance_sample: bool,
        only_use_next_actions_for_cql: bool,
        cql_max_target_backup: bool,
        cql_clip_diff_min: float,
        cql_clip_diff_max: float,
        use_calql: bool,
        use_calql_on_random_actions: bool,
        next_token_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.q_network = q_network
        self.q_head = q_head
        self.target_q_network = target_q_network
        self.target_q_head = target_q_head
        self.discount = discount
        self.tau = tau
        self.action_bounds = action_bounds
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.cql_n_actions = cql_n_actions
        self.cql_action_sample_method = cql_action_sample_method
        self.cql_importance_sample = cql_importance_sample
        self.only_use_next_actions_for_cql = only_use_next_actions_for_cql
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.use_calql = use_calql
        self.use_calql_on_random_actions = use_calql_on_random_actions
        self.next_token_loss_weight = next_token_loss_weight

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"] | tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b _n"]]:
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        out = self.q_network.compute_features(observation, action, **feature_kwargs)
        if isinstance(out, tuple):
            features, attn_scores = out[0], out[1]
            val = self.q_head(features)
            return val, attn_scores
        val = self.q_head(out)
        return val

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target Q-value using target network."""
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        target_out = self.target_q_network.compute_features(observation, action, **feature_kwargs)
        features = target_out[0] if isinstance(target_out, tuple) else target_out
        val = self.target_q_head(features)
        return val

    def compute_prefix_cache(
        self,
        observation: _model.Observation,
        use_target: bool = False,
    ) -> tuple[at.Array, at.Array, at.Array | None]:
        network = self.target_q_network if use_target else self.q_network
        if not hasattr(network, "compute_prefix_cache"):
            raise AttributeError(
                f"{type(network).__name__} does not support prefix caching. "
                "Callers should check hasattr before invoking."
            )
        return network.compute_prefix_cache(observation)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if policy is None:
            raise ValueError("CQLValueFunction requires a policy for action sampling.")
        if rng is None:
            raise ValueError("CQLValueFunction requires rng for action sampling.")
        del train
        only_use_next_actions_for_cql = self.only_use_next_actions_for_cql
        if isinstance(policy, _best_of_n.BestOfNWrapper) and policy.base_model is None:
            only_use_next_actions_for_cql = True
        q_loss, cql_loss, info = _objectives.cql_objective(
            self.q_network,
            self.q_head,
            self.target_q_network,
            self.target_q_head,
            transition,
            policy,
            rng=rng,
            discount=self.discount,
            action_bounds=self.action_bounds,
            cql_alpha=self.cql_alpha,
            cql_temp=self.cql_temp,
            cql_n_actions=self.cql_n_actions,
            cql_action_sample_method=self.cql_action_sample_method,
            cql_importance_sample=self.cql_importance_sample,
            only_use_next_actions_for_cql=only_use_next_actions_for_cql,
            cql_max_target_backup=self.cql_max_target_backup,
            cql_clip_diff_min=self.cql_clip_diff_min,
            cql_clip_diff_max=self.cql_clip_diff_max,
            use_calql=self.use_calql,
            use_calql_on_random_actions=self.use_calql_on_random_actions,
            next_token_loss_weight=self.next_token_loss_weight,
            value_function=self,
        )
        total_loss = q_loss + self.cql_alpha * cql_loss
        info["cql_alpha"] = jnp.array(self.cql_alpha)
        return total_loss, info

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_q_network, self.q_network, self.tau)
        _polyak_update(self.target_q_head, self.q_head, self.tau)




# =============================================================================
# Utilities
# =============================================================================


def _polyak_update(target_module: nnx.Module, online_module: nnx.Module, tau: float) -> None:
    """Polyak averaging: target = tau * online + (1 - tau) * target.

    JAX type-promotes the aggregate to the wider of the two operand dtypes, so a
    bf16 target mixed with an fp32 online would silently drift to fp32 after one
    update. Cast the final aggregate back to the target's original dtype to keep
    the target stack's dtype stable across steps.
    """
    target_state = nnx.state(target_module)
    online_state = nnx.state(online_module)
    new_target_state = jax.tree.map(
        lambda t, o: (tau * o + (1.0 - tau) * t).astype(t.dtype),
        target_state,
        online_state,
    )
    nnx.update(target_module, new_target_state)
