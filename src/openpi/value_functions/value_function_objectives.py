"""Value function training objectives.

All objectives are pure functions over a batch of transitions; the operations are
element-wise over the batch dimension.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import best_of_n as _best_of_n
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared.action_bounds import ActionBounds
from openpi.value_functions import base_value_functions as _base_vf
from openpi.value_functions.base_value_functions import Transition
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base_networks import BaseValueNetwork


def _compute_value_loss(head: ValueHead, features: at.Array, target: at.Array) -> at.Array:
    """Per-sample squared error of the regression head."""
    return jnp.square(head(features) - target)


def next_token_objective(
    network: BaseValueNetwork,
    next_token_embeddings: at.Array,
    next_token_targets: at.Array,
    next_token_mask: at.Array,
) -> at.Array:
    if not hasattr(network, "decode"):
        raise AttributeError(f"{type(network).__name__} does not support next-token decoding.")
    logits = network.decode(next_token_embeddings)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, next_token_targets)
    masked_loss = jnp.where(next_token_mask, per_token_loss, 0.0)
    denom = jnp.maximum(jnp.sum(next_token_mask, axis = -1), 1)
    return jnp.sum(masked_loss, axis = -1) / denom


def mc_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
    *,
    next_token_loss_weight: float = 0.0,
    rng: at.KeyArrayLike | None = None,
) -> tuple[at.Array, dict[str, at.Array]]:
    """Monte-Carlo regression: target = mc_return.


    Args:
        network: Value network for feature extraction.
        head: Value head for value prediction.
        transition: Transition.
        next_token_loss_weight: Weight on the auxiliary next-token loss. Only applies when
            the network emits next-token aux (PaliGemma with subtask indices in the batch).
        rng: Optional random key for stochastic operations (e.g., image augmentation).

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    action = transition.action if network.action_conditioned else None
    # PaliGemma returns (features, aux) when the observation carries subtask indices; aux
    # carries the next-token-prediction targets. Unwrap exactly like sarsa_objective so the
    # head sees features (not the tuple) and the aux loss can be composed.
    network_out = network.compute_features(transition.observation, action, rng=rng)
    if isinstance(network_out, tuple) and len(network_out) == 2 and isinstance(network_out[1], dict):
        features, aux = network_out
    else:
        features, aux = network_out, {}
    # Captured before the auxiliary next-token term is folded in. Subtracting it back
    # out afterwards would be equivalent only up to floating point, and would silently
    # go wrong the moment another auxiliary term joins the sum.
    value_loss = _compute_value_loss(head, features, transition.mc_return)
    loss = value_loss
    next_token_embeddings = aux.get("next_token_embeddings")
    next_token_loss = None
    if next_token_embeddings is not None:
        next_token_loss = next_token_objective(
            network,
            next_token_embeddings,
            aux["next_token_targets"],
            aux["next_token_mask"],
        )
        loss = loss + next_token_loss_weight * next_token_loss

    pred = head(features)
    td_error = pred - transition.mc_return

    info = {
        "value_loss": value_loss,
        "predicted_value": pred,
        "target_value": transition.mc_return,
        "td_error": td_error,
    }
    if next_token_loss is not None:
        info["next_token_loss"] = next_token_loss
    return loss, info


def sarsa_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    *,
    discount: float = 0.99,
    next_token_loss_weight: float = 0.0,
    rng: at.KeyArrayLike | None = None,
) -> tuple[at.Array, dict[str, at.Array]]:
    """SARSA: target = r + gamma * Q(s', a').


    Args:
        network: Online value network.
        head: Online value head.
        transition: Transition. If transition.td_discount is set
            (per-sample discount from the dataloader), it takes precedence over discount.
        target_network: Target network for stable target computation.
        target_head: Target head.
        discount: Fallback discount factor used when transition.td_discount is None.
        rng: Optional random key for stochastic operations (e.g., image augmentation).

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    # Target network doesn't need augmentation (no gradient flow).
    # PaliGemma returns (features, attn_scores) when rng=None; unwrap if so.
    target_out = target_network.compute_features(transition.next_observation, transition.next_action)
    target_features = target_out[0] if isinstance(target_out, tuple) else target_out
    target_value = target_head(target_features)

    effective_discount = transition.td_discount if transition.td_discount is not None else discount
    not_done = 1.0 - transition.termination.astype(jnp.float32)
    target = transition.reward + effective_discount * not_done * target_value
    target = jax.lax.stop_gradient(target)

    action = transition.action if network.action_conditioned else None
    network_out = network.compute_features(transition.observation, action, rng=rng)
    if isinstance(network_out, tuple) and len(network_out) == 2 and isinstance(network_out[1], dict):
        features, aux = network_out
    else:
        features, aux = network_out, {}
    # Captured before the auxiliary next-token term is folded in. Subtracting it back
    # out afterwards would be equivalent only up to floating point, and would silently
    # go wrong the moment another auxiliary term joins the sum.
    value_loss = _compute_value_loss(head, features, target)
    loss = value_loss
    next_token_embeddings = aux.get("next_token_embeddings")
    next_token_loss = None
    if next_token_embeddings is not None:
        next_token_loss = next_token_objective(
            network,
            next_token_embeddings,
            aux["next_token_targets"],
            aux["next_token_mask"],
        )
        loss = loss + next_token_loss_weight * next_token_loss

    pred = head(features)
    td_error = pred - target

    mc_loss = jnp.square(pred - transition.mc_return)

    info = {
        "value_loss": value_loss,
        "predicted_value": pred,
        "target_value": target,
        "next_value": target_value,
        "td_error": td_error,
        "mc_loss": mc_loss,
        "effective_discount": jnp.mean(effective_discount),
    }
    if next_token_loss is not None:
        info["next_token_loss"] = next_token_loss
    return loss, info


# =============================================================================
# CQL Helpers
# =============================================================================


def _flatten_observation_for_policy(
    observation: _model.Observation,
) -> tuple[_model.Observation, int, int | None]:
    """Flatten multi-transition observations for policy sampling."""
    state = observation.state
    if state.ndim == 3:
        batch_size, num_transitions = state.shape[:2]
        flat_obs = jax.tree_map(
            lambda x: x.reshape((batch_size * num_transitions,) + x.shape[2:]) if x is not None else None,
            observation,
        )
        return flat_obs, batch_size, num_transitions
    batch_size = state.shape[0]
    return observation, batch_size, None


def _flatten_transition_for_policy(
    transition: Transition,
) -> tuple[Transition, int, int | None]:
    """Flatten multi-transition for policy sampling."""
    state = transition.observation.state
    if state.ndim == 3:
        batch_size, num_transitions = state.shape[:2]

        def flatten(x):
            if x is None:
                return None
            return x.reshape((batch_size * num_transitions,) + x.shape[2:])

        flat_obs = jax.tree_map(flatten, transition.observation)
        flat_next_obs = (
            jax.tree_map(flatten, transition.next_observation) if transition.next_observation is not None else None
        )
        flat_transition = Transition(
            observation=flat_obs,
            action=flatten(transition.action),
            reward=flatten(transition.reward),
            next_observation=flat_next_obs,
            next_action=flatten(transition.next_action),
            mc_return=flatten(transition.mc_return),
            mc_return_mask=flatten(transition.mc_return_mask),
            termination=flatten(transition.termination),
            truncation=flatten(transition.truncation),
            td_discount=flatten(transition.td_discount) if transition.td_discount is not None else None,
            counterfactual_next_actions=(
                flatten(transition.counterfactual_next_actions)
                if transition.counterfactual_next_actions is not None
                else None
            ),
        )
        return flat_transition, batch_size, num_transitions
    batch_size = state.shape[0]
    return transition, batch_size, None


def _reshape_policy_actions(
    actions_flat: at.Array,
    *,
    batch_size: int,
    num_transitions: int | None,
    action_horizon: int,
    action_dim: int,
) -> at.Array:
    if num_transitions is None:
        return actions_flat.reshape(batch_size, action_horizon, action_dim)
    return actions_flat.reshape(batch_size, num_transitions, action_horizon, action_dim)


def _reshape_policy_actions_with_samples(
    actions_flat: at.Array,
    *,
    batch_size: int,
    num_transitions: int | None,
    num_samples: int,
    action_horizon: int,
    action_dim: int,
) -> at.Array:
    if num_transitions is None:
        return actions_flat.reshape(batch_size, num_samples, action_horizon, action_dim)
    return actions_flat.reshape(batch_size, num_transitions, num_samples, action_horizon, action_dim)


def _reshape_log_prob(
    log_prob_flat: at.Array,
    *,
    batch_size: int,
    num_transitions: int | None,
) -> at.Array:
    if num_transitions is None:
        return log_prob_flat
    return log_prob_flat.reshape(batch_size, num_transitions)


def _reshape_log_prob_with_samples(
    log_prob_flat: at.Array,
    *,
    batch_size: int,
    num_transitions: int | None,
    num_samples: int,
) -> at.Array:
    if num_transitions is None:
        return log_prob_flat.reshape(batch_size, num_samples)
    return log_prob_flat.reshape(batch_size, num_transitions, num_samples)


def _sample_policy_actions(
    policy: _model.BaseModel,
    transition: Transition,
    rng: at.KeyArrayLike,
    *,
    num_actions: int | None,
    compute_next_action: bool = False,
    value_function: _base_vf.BaseValueFunction | None = None,
) -> tuple[at.Array, at.Array]:
    """Sample actions (and log-probs) from policy for single or multi transitions.

    Args:
        policy: Policy model to sample from.
        transition: Transition containing observations.
        rng: Random key for sampling.
        num_actions: Number of actions to sample per observation (None for single sample).
        compute_next_action: If True, use next_observation for action sampling.
        value_function: Optional value function (forwarded to policy if supported).

    Returns:
        Tuple of (actions, log_probs).
    """
    flat_transition, batch_size, num_transitions = _flatten_transition_for_policy(transition)
    flat_batch = batch_size if num_transitions is None else batch_size * num_transitions

    dist = policy.action_distribution(rng, flat_transition, compute_next_action = compute_next_action, value_function = value_function)

    if num_actions is None:
        actions_flat, log_prob_flat = dist.sample_and_log_prob(seed=rng)
        actions_flat = actions_flat.reshape(flat_batch, policy.action_horizon, policy.action_dim)
        actions = _reshape_policy_actions(
            actions_flat,
            batch_size=batch_size,
            num_transitions=num_transitions,
            action_horizon=policy.action_horizon,
            action_dim=policy.action_dim,
        )
        log_prob = _reshape_log_prob(log_prob_flat, batch_size=batch_size, num_transitions=num_transitions)
        return actions, log_prob

    actions_flat, log_prob_flat = dist.sample_and_log_prob(seed=rng, sample_shape=num_actions)
    actions_flat = jnp.transpose(actions_flat, (1, 0, 2))
    log_prob_flat = jnp.transpose(log_prob_flat, (1, 0))
    actions_flat = actions_flat.reshape(flat_batch, num_actions, policy.action_horizon, policy.action_dim)
    actions = _reshape_policy_actions_with_samples(
        actions_flat,
        batch_size=batch_size,
        num_transitions=num_transitions,
        num_samples=num_actions,
        action_horizon=policy.action_horizon,
        action_dim=policy.action_dim,
    )
    log_prob = _reshape_log_prob_with_samples(
        log_prob_flat,
        batch_size=batch_size,
        num_transitions=num_transitions,
        num_samples=num_actions,
    )
    return actions, log_prob


def _expand_observation_for_actions(observation: _model.Observation, num_actions: int) -> _model.Observation:
    """Broadcast observation to include a sampled-action dimension."""
    state = observation.state
    if state.ndim == 2:

        def expand(x: at.Array | None) -> at.Array | None:
            if x is None:
                return None
            x = x[:, None, ...]
            return jnp.broadcast_to(x, (x.shape[0], num_actions) + x.shape[2:])

        return jax.tree_map(expand, observation)

    if state.ndim == 3:

        def expand(x: at.Array | None) -> at.Array | None:
            if x is None:
                return None
            x = x[:, :, None, ...]
            return jnp.broadcast_to(x, (x.shape[0], x.shape[1], num_actions) + x.shape[3:])

        return jax.tree_map(expand, observation)

    raise ValueError(f"Unsupported observation state ndim: {state.ndim}")


def _compute_q_values_for_samples(
    q_network: BaseValueNetwork,
    q_head: ValueHead,
    observation: _model.Observation,
    actions: at.Array,
    rng: at.KeyArrayLike | None = None,
) -> at.Array:
    """Compute Q values for multiple actions per observation.

    Supports both single-transition [batch, k, ...] and multi-transition
    [batch, n, k, ...] action shapes.
    """
    state = observation.state
    if state.ndim == 2:
        num_actions = actions.shape[1]
        obs_for_actions = _expand_observation_for_actions(observation, num_actions)
        out = q_network.compute_features(obs_for_actions, actions, rng = rng)
        features = out[0] if isinstance(out, tuple) else out
        return q_head(features)

    if state.ndim == 3:
        batch_size, num_transitions, num_actions = actions.shape[:3]
        action_horizon = actions.shape[3]
        action_dim = actions.shape[4]

        actions_reshaped = actions.transpose(0, 2, 1, 3, 4).reshape(
            batch_size * num_actions, num_transitions, action_horizon, action_dim
        )
        obs_repeated = jax.tree_map(
            lambda x: jnp.repeat(x, num_actions, axis=0) if x is not None else None, observation
        )
        out = q_network.compute_features(obs_repeated, actions_reshaped, rng = rng)
        features = out[0] if isinstance(out, tuple) else out
        q_values = q_head(features)

        q_values = q_values.reshape(batch_size, num_actions, num_transitions)
        return jnp.transpose(q_values, (0, 2, 1))

    raise ValueError(f"Unsupported observation state ndim: {state.ndim}")


def _sample_random_actions(
    rng: at.KeyArrayLike,
    *,
    shape: tuple[int, ...],
    method: Literal["uniform", "normal"],
    action_bounds: ActionBounds,
) -> at.Array:
    if method == "uniform":
        return action_bounds.sample_uniform(rng, shape[:-1])
    if method == "normal":
        return jax.random.normal(rng, shape=shape)
    raise ValueError(f"Unknown cql_action_sample_method: {method}")


def _random_action_log_prob(
    random_actions: at.Array,
    *,
    method: Literal["uniform", "normal"],
    action_bounds: ActionBounds,
) -> at.Array:
    if method == "uniform":
        per_timestep_log_prob = action_bounds.compute_uniform_log_prob(random_actions)
        return jnp.sum(per_timestep_log_prob, axis=-1)
    if method == "normal":
        log_two_pi = jnp.log(2.0 * jnp.pi)
        return -0.5 * jnp.sum(random_actions**2 + log_two_pi, axis=(-1, -2))
    raise ValueError(f"Unknown cql_action_sample_method: {method}")


# =============================================================================
# CQL Objective
# =============================================================================


def cql_objective(
    q_network: BaseValueNetwork,
    q_head: ValueHead,
    target_q_network: BaseValueNetwork,
    target_q_head: ValueHead,
    transition: Transition,
    policy: _model.BaseModel,
    *,
    rng: at.KeyArrayLike,
    discount: float = 0.99,
    next_token_loss_weight: float = 0.0,
    action_bounds: ActionBounds,
    cql_alpha: float,
    cql_temp: float = 1.0,
    cql_n_actions: int = 4,
    cql_action_sample_method: Literal["uniform", "normal"] = "uniform",
    cql_importance_sample: bool = True,
    only_use_next_actions_for_cql: bool = False,
    cql_max_target_backup: bool = False,
    cql_clip_diff_min: float = -np.inf,
    cql_clip_diff_max: float = np.inf,
    use_calql: bool = False,
    use_calql_on_random_actions: bool = True,
    value_function: _base_vf.BaseValueFunction | None = None,
) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
    """CQL objective with fixed alpha (returned separately for weighting)."""
    rng = jax.random.key(rng) if isinstance(rng, int) else rng

    # Q(s, a) prediction
    rng, q_rng = jax.random.split(rng)
    # PaliGemma returns (features, aux) when rng is set; aux carries the
    # next-token-prediction targets. Unwrap exactly like sarsa_objective so the
    # head sees features (not the tuple) and the aux loss can be composed.
    network_out = q_network.compute_features(transition.observation, transition.action, rng = q_rng)
    if isinstance(network_out, tuple) and len(network_out) == 2 and isinstance(network_out[1], dict):
        q_features, aux = network_out
    else:
        q_features, aux = network_out, {}
    q_pred = q_head(q_features)
    q_pred_for_logging = q_pred

    # Target Q(s', a') for TD backup
    rng, target_rng = jax.random.split(rng)
    if cql_max_target_backup:
        next_actions, _ = _sample_policy_actions(
            policy,
            transition,
            target_rng,
            num_actions=cql_n_actions,
            compute_next_action=True,
            value_function=value_function,
        )
        target_q_values = _compute_q_values_for_samples(
            target_q_network, target_q_head, transition.next_observation, next_actions
        )
        target_q_values = jnp.max(target_q_values, axis=-1)
    elif (
        isinstance(policy, _best_of_n.BestOfNWrapper)
        and policy.base_model is None
        and policy.use_target_value
    ):
        # BestOfN already ran the target network over all N cached candidates inside
        # sample_actions; reuse its argmax q-value as the Bellman target instead of
        # re-forwarding target_q_network on the winning action.
        next_actions, target_q_values = policy.select_best_action_and_q(
            target_rng,
            transition,
            compute_next_action=True,
            value_function=value_function,
        )
    else:
        next_actions, _ = _sample_policy_actions(
            policy,
            transition,
            target_rng,
            num_actions=None,
            compute_next_action=True,
            value_function=value_function,
        )
        target_out = target_q_network.compute_features(transition.next_observation, next_actions)
        target_features = target_out[0] if isinstance(target_out, tuple) else target_out
        target_q_values = target_q_head(target_features)

    assert target_q_values.shape == transition.reward.shape, (
        f"Expected target_q_values shape {transition.reward.shape}, got {target_q_values.shape}"
    )
    not_done = 1.0 - transition.termination.astype(jnp.float32)
    effective_discount = transition.td_discount if transition.td_discount is not None else discount
    target = transition.reward + effective_discount * not_done * target_q_values
    target = jax.lax.stop_gradient(target)

    # TD loss
    assert target.shape == transition.reward.shape, (
        f"Expected target shape {transition.reward.shape}, got {target.shape}"
    )
    # Captured before the auxiliary next-token term is folded in. Subtracting it back
    # out afterwards would be equivalent only up to floating point, and would silently
    # go wrong the moment another auxiliary term joins the sum.
    value_loss = _compute_value_loss(q_head, q_features, target)
    q_loss = value_loss
    next_token_embeddings = aux.get("next_token_embeddings")
    next_token_loss = None
    if next_token_embeddings is not None:
        next_token_loss = next_token_objective(
            q_network,
            next_token_embeddings,
            aux["next_token_targets"],
            aux["next_token_mask"],
        )
        q_loss = q_loss + next_token_loss_weight * next_token_loss
    td_error = q_pred_for_logging - target

    q_pred_mc_diff = (q_pred_for_logging - transition.mc_return) if transition.mc_return is not None else None

    # Early return if CQL is disabled
    if cql_alpha == 0:
        cql_loss = jnp.zeros_like(q_loss)
        info = {
            "value_loss": value_loss,
            "q_mean": jnp.mean(q_pred_for_logging),
            "q_std": jnp.std(q_pred_for_logging),
            "target_mean": jnp.mean(target),
            "target_std": jnp.std(target),
            "td_error_mean": jnp.mean(td_error),
            "td_error_std": jnp.std(td_error),
        }
        if q_pred_mc_diff is not None:
            info["mc_loss"] = _base_vf.masked_mean(jnp.square(q_pred_mc_diff), transition.mc_return_mask)
            info["avg_q_minus_mc"] = _base_vf.masked_mean(q_pred_mc_diff, transition.mc_return_mask)
            info["q_pred_per_sample"] = q_pred_for_logging
            info["mc_return_per_sample"] = transition.mc_return
        if next_token_loss is not None:
            info["next_token_loss"] = next_token_loss
        return q_loss, cql_loss, info

    # CQL samples: random, next, (optional) current
    rng, random_rng, current_rng, next_rng = jax.random.split(rng, 4)
    if transition.observation.state.ndim == 3:
        batch_size, num_transitions = transition.observation.state.shape[:2]
        random_shape = (batch_size, num_transitions, cql_n_actions, policy.action_horizon, policy.action_dim)
        sample_axis = 2
    else:
        batch_size = transition.observation.state.shape[0]
        random_shape = (batch_size, cql_n_actions, policy.action_horizon, policy.action_dim)
        sample_axis = 1

    cql_random_actions = _sample_random_actions(
        random_rng,
        shape=random_shape,
        method=cql_action_sample_method,
        action_bounds=action_bounds,
    )

    if not only_use_next_actions_for_cql:
        cql_current_actions, cql_current_log_pis = _sample_policy_actions(
            policy,
            transition,
            current_rng,
            num_actions=cql_n_actions,
            compute_next_action=False,
            value_function=value_function,
        )
    else:
        cql_current_actions = None
        cql_current_log_pis = None

    cql_next_actions, cql_next_log_pis = _sample_policy_actions(
        policy,
        transition,
        next_rng,
        num_actions=cql_n_actions,
        compute_next_action=True,
        value_function=value_function,
    )

    action_blocks = [cql_random_actions, cql_next_actions]
    if not only_use_next_actions_for_cql:
        action_blocks.append(cql_current_actions)

    all_sampled_actions = jnp.concatenate(action_blocks, axis=sample_axis)
    expected_actions = cql_n_actions * (2 if only_use_next_actions_for_cql else 3)
    assert all_sampled_actions.shape[sample_axis] == expected_actions, (
        f"Expected {expected_actions} sampled actions, got {all_sampled_actions.shape[sample_axis]}"
    )

    # Q values for sampled actions
    rng, cql_q_rng = jax.random.split(rng)
    cql_q_samples = _compute_q_values_for_samples(q_network, q_head, transition.observation, all_sampled_actions, rng = cql_q_rng)
    if transition.observation.state.ndim == 3:
        expected_shape = (batch_size, num_transitions, expected_actions)
    else:
        expected_shape = (batch_size, expected_actions)
    assert cql_q_samples.shape == expected_shape, (
        f"Expected cql_q_samples shape {expected_shape}, got {cql_q_samples.shape}"
    )

    info: dict[str, at.Array] = {
        "all_sampled_action_values": jnp.mean(cql_q_samples),
        "random_action_values": jnp.mean(cql_q_samples[..., :cql_n_actions]),
        "next_action_values": jnp.mean(cql_q_samples[..., cql_n_actions : 2 * cql_n_actions]),
    }
    if not only_use_next_actions_for_cql:
        info["current_action_values"] = jnp.mean(cql_q_samples[..., 2 * cql_n_actions :])

    # Cal-QL lower bound
    if use_calql:
        n_actions_for_calql = cql_n_actions * 3
        if not use_calql_on_random_actions:
            n_actions_for_calql -= cql_n_actions
        if only_use_next_actions_for_cql:
            n_actions_for_calql -= cql_n_actions

        if transition.mc_return.ndim == 2:
            mc_lower_bound = jnp.repeat(transition.mc_return[:, :, None], n_actions_for_calql, axis=2)
        else:
            mc_lower_bound = jnp.repeat(transition.mc_return[:, None], n_actions_for_calql, axis=1)

        num_vals = jnp.size(cql_q_samples[..., :n_actions_for_calql])
        if use_calql_on_random_actions:
            calql_bound_rate = jnp.sum(cql_q_samples < mc_lower_bound) / num_vals
            cql_q_samples = jnp.maximum(cql_q_samples, mc_lower_bound)
        else:
            calql_bound_rate = jnp.sum(cql_q_samples[..., cql_n_actions:] < mc_lower_bound) / num_vals
            cql_q_samples = jnp.concatenate(
                [
                    cql_q_samples[..., :cql_n_actions],
                    jnp.maximum(cql_q_samples[..., cql_n_actions:], mc_lower_bound),
                ],
                axis=-1,
            )
        info["calql_bound_rate"] = calql_bound_rate

    # Importance sampling correction
    if cql_importance_sample:
        random_log_prob = _random_action_log_prob(
            cql_random_actions,
            method=cql_action_sample_method,
            action_bounds=action_bounds,
        )
        importance_prob = jnp.concatenate(
            [random_log_prob, cql_next_log_pis],
            axis=sample_axis,
        )
        if not only_use_next_actions_for_cql:
            importance_prob = jnp.concatenate([importance_prob, cql_current_log_pis], axis=sample_axis)
        cql_q_samples = cql_q_samples - importance_prob
    else:
        q_pred_expanded = jnp.expand_dims(q_pred, axis=-1)
        cql_q_samples = jnp.concatenate([cql_q_samples, q_pred_expanded], axis=-1)
        cql_q_samples = cql_q_samples - jnp.log(cql_q_samples.shape[-1]) * cql_temp

    # Log-sum-exp over actions
    cql_ood_values = jax.scipy.special.logsumexp(cql_q_samples / cql_temp, axis=-1) * cql_temp
    cql_q_diff = cql_ood_values - q_pred

    cql_loss = jnp.clip(cql_q_diff, cql_clip_diff_min, cql_clip_diff_max)
    assert cql_loss.shape == transition.reward.shape, (
        f"Expected cql_loss shape {transition.reward.shape}, got {cql_loss.shape}"
    )

    info.update(
        {
            "value_loss": value_loss,
            "q_mean": jnp.mean(q_pred_for_logging),
            "q_std": jnp.std(q_pred_for_logging),
            "target_mean": jnp.mean(target),
            "target_std": jnp.std(target),
            "td_error_mean": jnp.mean(td_error),
            "td_error_std": jnp.std(td_error),
            "cql_ood_values_mean": jnp.mean(cql_ood_values),
            "cql_q_diff_mean": jnp.mean(cql_q_diff),
            "cql_loss_mean": jnp.mean(cql_loss),
        }
    )
    if q_pred_mc_diff is not None:
        info["mc_loss"] = _base_vf.masked_mean(jnp.square(q_pred_mc_diff), transition.mc_return_mask)
        info["avg_q_minus_mc"] = _base_vf.masked_mean(q_pred_mc_diff, transition.mc_return_mask)
    if next_token_loss is not None:
        info["next_token_loss"] = next_token_loss
    return q_loss, cql_loss, info
