"""Best-of-N sampling wrapper for value-guided action selection.

This module provides a wrapper around any BaseModel policy that samples N action
candidates and selects the best one according to a value function (Q-function)
passed at runtime. This enables value-guided policy improvement without modifying
the underlying policy architecture.
"""

import dataclasses
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import yam_eef_jax as _yam_eef_jax
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
import openpi.transforms as _transforms

# Set OPENPI_DEBUG=1 in the environment to enable observation/action-stat logging
# during `BestOfNWrapper.sample_actions` and the policy-only path in
# `BestOfNPolicy._run_jit_inference`. Read once at module import.
_DEBUG: bool = os.environ.get("OPENPI_DEBUG", "0") == "1"
from openpi.value_functions import base_value_functions as _base_vf

# Chunk-wise-delta layout shared by the joint and EEF sides of the YAM bimanual
# 14D vector: 6 delta dims + 1 absolute gripper per arm. Only the EEF side carries
# rotations, at dims 3:6 and 10:13. Mirrors `LeRobotRldsDataConfig.create`.
_JOINT_DELTA_MASK = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))
_EEF_RPY_INDEX_START = (3, 10)


def _log_model_inputs(
    tag: str,
    observation: _model.Observation,
    *,
    actions: jnp.ndarray | None = None,
) -> None:
    """Log key Observation fields (and optionally an action chunk) via jax.debug.print.

    Used to inspect what the policy and the critic actually see at sample time. Only the
    first element of the leading batch axis is shown to keep log lines compact. Fields
    that are None are skipped at trace time (not added to the JIT graph).
    """
    jax.debug.print(
        f"[debug] {tag} state shape={observation.state.shape} value={{v}}",
        v = observation.state[0],
    )
    if observation.action_mask is not None:
        jax.debug.print(
            f"[debug] {tag} action_mask shape={observation.action_mask.shape} value={{v}}",
            v = observation.action_mask[0],
        )
    if observation.tokenized_prompt is not None:
        jax.debug.print(
            f"[debug] {tag} tokenized_prompt shape={observation.tokenized_prompt.shape} value={{v}}",
            v = observation.tokenized_prompt[0],
        )
        if observation.tokenized_prompt_mask is not None:
            # Detokenize via host callback — PaligemmaTokenizer is numpy-only and
            # can't run under jit. Lazy-import to avoid pulling SentencePiece into
            # the import path of every BestOfN consumer.
            from openpi.rlds_utils.utils import detokenize_prompt

            def _log_detokenized(token_ids, mask):
                print(
                    f"[debug] {tag} untokenized_prompt: "
                    f"{detokenize_prompt(token_ids, mask)!r}"
                )

            jax.debug.callback(
                _log_detokenized,
                observation.tokenized_prompt[0],
                observation.tokenized_prompt_mask[0],
            )
    if observation.tokenized_prompt_mask is not None:
        jax.debug.print(
            f"[debug] {tag} tokenized_prompt_mask shape={observation.tokenized_prompt_mask.shape} value={{v}}",
            v = observation.tokenized_prompt_mask[0],
        )
    for cam_name, img in observation.images.items():
        jax.debug.print(
            f"[debug] {tag} image[{cam_name}] shape={img.shape} min={{mn:.4f}} max={{mx:.4f}}",
            mn = jnp.min(img[0]), mx = jnp.max(img[0]),
        )
    for mask_name, mask in observation.image_masks.items():
        jax.debug.print(
            f"[debug] {tag} image_mask[{mask_name}] shape={mask.shape} value={{v}}",
            v = mask[0],
        )
    if actions is not None:
        # Print the full action chunk for the first batch element.
        chunk = actions[0]
        jax.debug.print(
            f"[debug] {tag} actions[0] shape={chunk.shape} value={{v}}",
            v = chunk,
        )


@dataclasses.dataclass(frozen=True)
class BestOfNWrapperConfig(_model.BaseModelConfig):
    """Config for Best-of-N wrapper. Value function is passed at runtime.

    If base_model_config is None, the wrapper uses cached counterfactual actions
    from the transition instead of sampling from a base model. In this case,
    action_dim and action_horizon must be provided explicitly.
    """

    action_dim: int = 0
    action_horizon: int = 0
    max_token_len: int = 0

    # Optional: if None, uses cached counterfactual actions from transition
    base_model_config: _model.BaseModelConfig | None = None

    num_samples: int = 10

    use_target_value: bool = False

    # Norm stats referenced for the action-dim slice and the policy/critic
    # equality guard. Both None (cached path) or both provided.
    policy_norm_stats: dict[str, NormStats] | None = None
    critic_norm_stats: dict[str, NormStats] | None = None

    # Policy/critic chunk-wise-delta format. Asserted equal in the wrapper: a
    # shared format means the policy's normalized output is already in the
    # critic's space, so no renormalization is applied.
    policy_use_chunk_wise_delta: bool = False
    critic_use_chunk_wise_delta: bool = False

    # Critic-side knobs packaged for the wrapper. Recognized keys:
    #   - use_quantile_norm (bool): selects the clip range for policy candidates.
    #   - subsample (bool): critic's data.subsample; matters for the 2-D norm-stat
    #     equality check (`policy[1::2]` vs `critic`) and the (60, 60) regime.
    #   - action_dim_offset (int | None): policy → critic action-dim slice
    #     [offset : offset + critic_action_dim]. Required when dims differ.
    #   - action_horizon (int | None): critic's expected action horizon. Drives
    #     the wrapper's subsample/pad adapter. Supported (action_horizon,
    #     critic_action_horizon) pairs: (60, 50), (30, 50), (50, 50), (60, 60).
    #   - policy_subsample (bool): policy's data.subsample. When True in the
    #     (60, 60) regime, the wrapper skips the stride-2 subsample (the policy
    #     already emits half-cadence actions) and ensures both policy and
    #     critic action masks are 30 ones + 30 zeros.
    critic_kwargs: dict | None = None



    # Joint-space policy scored by an EEF critic: convert the candidates into the
    # critic's 14D EEF space before scoring. See `BestOfNWrapper.__init__`.
    convert_policy_actions_to_eef: bool = False
    policy_use_quantile_norm: bool = False

    def __post_init__(self):
        if self.base_model_config is not None:
            # Sync inherited fields from base_model_config (frozen dataclass requires object.__setattr__).
            object.__setattr__(self, "action_dim", self.base_model_config.action_dim)
            object.__setattr__(self, "action_horizon", self.base_model_config.action_horizon)
            object.__setattr__(self, "max_token_len", self.base_model_config.max_token_len)
        # Cached-only mode: action_dim and action_horizon must be provided
        elif self.action_dim == 0 or self.action_horizon == 0:
            raise ValueError("action_dim and action_horizon must be provided when base_model_config is None")
        if (self.policy_norm_stats is None) != (self.critic_norm_stats is None):
            raise ValueError(
                "policy_norm_stats and critic_norm_stats must both be provided or both be None. "
                f"Got policy_norm_stats={'set' if self.policy_norm_stats is not None else 'None'}, "
                f"critic_norm_stats={'set' if self.critic_norm_stats is not None else 'None'}."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.BEST_OF_N

    @override
    def create(self, rng: at.KeyArrayLike) -> "BestOfNWrapper":
        base_model = self.base_model_config.create(rng) if self.base_model_config is not None else None
        return BestOfNWrapper(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            base_model=base_model,
            num_samples=self.num_samples,
            use_target_value=self.use_target_value,
            policy_norm_stats=self.policy_norm_stats,
            critic_norm_stats=self.critic_norm_stats,
            policy_use_chunk_wise_delta=self.policy_use_chunk_wise_delta,
            critic_use_chunk_wise_delta=self.critic_use_chunk_wise_delta,
            critic_kwargs=self.critic_kwargs,
            convert_policy_actions_to_eef=self.convert_policy_actions_to_eef,
            policy_use_quantile_norm=self.policy_use_quantile_norm,
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.base_model_config is not None:
            return self.base_model_config.inputs_spec(batch_size=batch_size)
        # Cached-only mode: return minimal spec
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, 1], jnp.float32),
            )
        actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return obs, actions


@dataclasses.dataclass
class BestOfNWrapper(_model.BaseModel):
    """Wrapper that samples N actions and selects best via Q-value.

    Value function is passed at runtime to sample_actions(), not stored.

    If base_model is None, uses cached counterfactual actions from the transition
    instead of sampling from a base model.
    """

    base_model: _model.BaseModel | None
    num_samples: int
    use_target_value: bool
    policy_norm_stats: dict[str, NormStats] | None
    critic_norm_stats: dict[str, NormStats] | None
    policy_use_chunk_wise_delta: bool
    critic_use_chunk_wise_delta: bool
    critic_use_quantile_norm: bool
    critic_subsample: bool
    policy_subsample: bool
    critic_action_dim_offset: int | None
    critic_action_horizon: int | None
    convert_policy_actions_to_eef: bool
    policy_use_quantile_norm: bool

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        max_token_len: int,
        *,
        base_model: _model.BaseModel | None,
        num_samples: int,
        use_target_value: bool,
        policy_norm_stats: dict[str, NormStats] | None = None,
        critic_norm_stats: dict[str, NormStats] | None = None,
        policy_use_chunk_wise_delta: bool = False,
        critic_use_chunk_wise_delta: bool = False,
        critic_kwargs: dict | None = None,
        convert_policy_actions_to_eef: bool = False,
        policy_use_quantile_norm: bool = False,
    ):
        super().__init__(action_dim, action_horizon, max_token_len)
        self.base_model = base_model
        self.num_samples = num_samples
        self.use_target_value = use_target_value
        self.policy_norm_stats = policy_norm_stats
        self.critic_norm_stats = critic_norm_stats
        self.policy_use_chunk_wise_delta = policy_use_chunk_wise_delta
        self.critic_use_chunk_wise_delta = critic_use_chunk_wise_delta
        self.convert_policy_actions_to_eef = convert_policy_actions_to_eef
        self.policy_use_quantile_norm = policy_use_quantile_norm
        ck = critic_kwargs or {}
        self.critic_use_quantile_norm = ck.get("use_quantile_norm", False)
        self.critic_subsample = ck.get("subsample", False)
        self.policy_subsample = ck.get("policy_subsample", False)
        critic_action_dim_offset = ck.get("action_dim_offset", None)
        critic_action_horizon = ck.get("action_horizon", None)
        self.critic_action_dim_offset = critic_action_dim_offset
        # This path scores the policy's normalized output with the critic
        # directly (no renorm), so it requires both ends to share the
        # chunk-wise-delta format and the exact norm stats. Fail fast otherwise.
        if policy_use_chunk_wise_delta != critic_use_chunk_wise_delta:
            raise ValueError(
                "BestOfNWrapper requires policy and critic to use the same "
                f"use_chunk_wise_delta. Got policy={policy_use_chunk_wise_delta}, "
                f"critic={critic_use_chunk_wise_delta}."
            )
        if (policy_norm_stats is None) != (critic_norm_stats is None):
            raise ValueError(
                "policy_norm_stats and critic_norm_stats must both be provided or both be None."
            )
        # Joint-space policy + EEF critic. The candidates are carried into the
        # critic's space inside `sample_actions` (unnormalize -> absolute -> FK ->
        # EEF delta -> renormalize), so the two sides deliberately reference
        # different norm-stat assets and the equality guard below does not apply.
        # `Normalize` / `Unnormalize` are pure arithmetic over dict leaves, so the
        # same transforms the host-side pipeline uses run on traced arrays. Only the
        # stats they need are stored: a frozen-dataclass transform object held as an
        # nnx attribute would be hashed as static graphdef and its dict field is
        # unhashable, so the transforms are constructed at trace time instead.
        self._conversion_stats = None
        if convert_policy_actions_to_eef:
            if policy_norm_stats is None:
                raise ValueError(
                    "convert_policy_actions_to_eef=True requires policy_norm_stats and "
                    "critic_norm_stats (the conversion runs through both)."
                )
            self._conversion_stats = {
                "policy": {k: policy_norm_stats[k] for k in ("state", "actions")},
                "critic": {k: critic_norm_stats[k] for k in ("state", "actions")},
            }
        elif policy_norm_stats is not None:
            # Only the keys used when scoring Q(s, a) at BestOfN inference — state and
            # actions — must match between policy and critic norm stats. Critic-only keys
            # carried from training (next_state / next_actions for TD/SARSA targets, and
            # 'action_diff' for chunk-wise-delta critics) are never used here, so they are
            # ignored for both the key-set check and the per-key value comparison.
            _core_keys = {"state", "actions"}
            p_core = set(policy_norm_stats) & _core_keys
            c_core = set(critic_norm_stats) & _core_keys
            if p_core != c_core:
                raise ValueError(
                    "BestOfNWrapper requires identical policy/critic core norm-stat keys "
                    f"(state/actions); differ: {sorted(p_core)} vs {sorted(c_core)}."
                )
            for stats_key in sorted(p_core):
                p_stats = policy_norm_stats[stats_key]
                c_stats = critic_norm_stats[stats_key]
                for field_name in ("mean", "std", "q01", "q99"):
                    p_val = getattr(p_stats, field_name)
                    c_val = getattr(c_stats, field_name)
                    if (p_val is None) != (c_val is None):
                        raise ValueError(
                            "BestOfNWrapper requires identical policy/critic norm stats; "
                            f"'{stats_key}'.{field_name} presence differs."
                        )
                    if p_val is not None:
                        p_arr = np.asarray(p_val)
                        c_arr = np.asarray(c_val)
                        # If the critic was trained with subsample=True, its
                        # per-position 2-D stats correspond to the stride-2 slice
                        # of the policy's, so compare that slice to match positions.
                        if self.critic_subsample and not self.policy_subsample and p_arr.ndim == 2:
                            p_arr = p_arr[1::2]
                            c_arr = c_arr[: p_arr.shape[0]]
                        if not np.array_equal(p_arr, c_arr):
                            raise ValueError(
                                "BestOfNWrapper requires identical policy/critic norm stats; "
                                f"'{stats_key}'.{field_name} differs."
                            )
        # Five supported (action_horizon, critic_action_horizon) regimes:
        #   - (60, 50): 60 Hz policy + 30 Hz critic. Stride-2 subsample
        #     (60 -> 30), then zero-pad to 50.
        #   - (30, 50): same-fps policy + critic. Skip subsample, zero-pad
        #     30 -> 50.
        #   - (50, 50): matched-horizon policy + critic. No subsample, no
        #     pad — the chunk is forwarded as-is to the critic. The 30 / 20
        #     valid / invalid split is carried entirely by `action_mask`.
        #   - (60, 60): 60 Hz policy + 60-step critic (HDF5 60 Hz critics,
        #     e.g. sim_bimanual_assembly). Stride-2 subsample (60 -> 30),
        #     then zero-pad to 60; action_mask is 30 True + 30 False.
        #   - (20, 20): matched-horizon, no subsample, no pad; action_mask
        #     is all-True (every slot is a real action).
        if critic_action_horizon is not None and not (
            (critic_action_horizon == 50 and action_horizon in (30, 50, 60))
            or (critic_action_horizon == 60 and action_horizon == 60)
            or (critic_action_horizon == 20 and action_horizon == 20)
        ):
            raise ValueError(
                "Only (action_horizon=30, critic_action_horizon=50), "
                "(action_horizon=50, critic_action_horizon=50), "
                "(action_horizon=60, critic_action_horizon=50), "
                "(action_horizon=60, critic_action_horizon=60), "
                "and (action_horizon=20, critic_action_horizon=20) are supported. "
                f"Got action_horizon={action_horizon}, critic_action_horizon={critic_action_horizon}."
            )
        self.critic_action_horizon = critic_action_horizon
        if action_horizon == 60 and critic_action_horizon == 60:
            assert self.critic_subsample, "(60, 60) regime requires critic_subsample=True"
        if convert_policy_actions_to_eef and not (
            policy_use_chunk_wise_delta and critic_use_chunk_wise_delta
        ):
            raise ValueError(
                "convert_policy_actions_to_eef=True is only implemented for chunk-wise-delta "
                "policies and critics (the conversion undoes the joint delta and re-applies "
                "it in EEF space). Got "
                f"policy_use_chunk_wise_delta={policy_use_chunk_wise_delta}, "
                f"critic_use_chunk_wise_delta={critic_use_chunk_wise_delta}."
            )

    def _to_eef(self, actions: at.Array, state_joint: at.Array) -> tuple[at.Array, at.Array]:
        """Carry normalized joint candidates into the critic's unnormalized EEF delta space.

        `actions` is [B, N, ah, 14] in the policy's normalized chunk-wise-delta space and
        `state_joint` is [B, 14] normalized. Undoes the joint delta, runs forward
        kinematics, and re-applies the delta in EEF space (where rotations compose as
        `R_action @ R_state.inv()`). Returns the EEF-delta actions alongside the absolute
        EEF state, which the caller still needs to normalize for the critic observation.

        Both delta steps are elementwise over the chunk axis, so this commutes with the
        caller's stride-2 subsample and can run on the full chunk.
        """
        unnormalize = _transforms.Unnormalize(
            self._conversion_stats["policy"], use_quantiles = self.policy_use_quantile_norm,
        )
        unnormalized = unnormalize({"actions": actions, "state": state_joint})
        absolute_joint = _yam_eef_jax.apply_absolute(
            unnormalized["actions"],
            unnormalized["state"][:, None, :],
            _JOINT_DELTA_MASK,
            None,
        )
        origins, axes = _yam_eef_jax.chain_constants()
        absolute_eef = _yam_eef_jax.joint_to_eef(absolute_joint, origins, axes)
        state_eef = _yam_eef_jax.joint_to_eef(unnormalized["state"], origins, axes)
        delta_eef = _yam_eef_jax.apply_delta(
            absolute_eef, state_eef[:, None, :], _JOINT_DELTA_MASK, _EEF_RPY_INDEX_START,
        )
        return delta_eef, state_eef

    def _get_cached_actions(
        self,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool,
    ) -> at.Array:
        """Get cached counterfactual actions from transition."""
        if compute_next_action:
            if transition.counterfactual_next_actions is None:
                raise ValueError(
                    "compute_next_action=True but transition.counterfactual_next_actions is None. "
                    "Ensure the data pipeline populates this field."
                )
            actions = transition.counterfactual_next_actions
        else:
            if transition.counterfactual_actions is None:
                raise ValueError(
                    "compute_next_action=False but transition.counterfactual_actions is None. "
                    "Ensure the data pipeline populates this field."
                )
            actions = transition.counterfactual_actions

        if actions.shape[1] < self.num_samples:
            raise ValueError(
                "Cached counterfactual actions provide fewer samples than BestOfN requires: "
                f"expected at least {self.num_samples}, got {actions.shape[1]} "
                f"for shape {actions.shape}."
            )
        if actions.shape[1] > self.num_samples:
            actions = actions[:, : self.num_samples]
        # Slice to model's action_horizon if cached actions have longer horizon
        if actions.shape[2] > self.action_horizon:
            actions = actions[:, :, : self.action_horizon, :]
        return actions

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool = False,
        value_function: _base_vf.BaseValueFunction | None = None,
        critic_observation: _model.Observation | None = None,
        sample_rngs: at.KeyArrayLike | None = None,
        **kwargs,
    ) -> _model.Actions:
        """Sample N actions and select best via Q-value.

        If base_model is provided, samples from it. Otherwise, uses cached
        counterfactual actions from the transition.

        Args:
            rng: Random key for sampling.
            transition: Transition containing observation and next_observation.
            compute_next_action: If True, use next_observation.
            value_function: Action-conditioned value function (required).
            critic_observation: Optional `Observation` for the critic forward. Use this when
                any critic-facing field differs from the policy's (e.g. critic-tokenized prompt,
                critic-pipeline images, critic-normalized state). Falls back to `transition.observation`
                when None.
            sample_rngs: Optional pre-split rng array of shape (N, ...). If provided, used
                as the per-sample rngs directly (skipping the in-JIT `jax.random.split` of
                `rng`). Intended for the multi-host serve path: the caller hands in a
                DATA_AXIS-sharded rng array so each chip gets a unique rng (and therefore
                unique noise) without the cross-host all-gather collapsing per-host
                divergence into a "Frankenstein" tensor. Pass None for the legacy
                same-rng-per-call path (used by training / single-host eval).
            **kwargs: Forwarded to base model's sample_actions (if base_model exists).

        Returns:
            Tuple of:
              - selected actions of shape [batch, action_horizon, action_dim].
              - q_values of shape [batch, num_samples] (B always = 1 in the eval flow).
        """
        if value_function is None:
            raise ValueError("value_function is required for BestOfNWrapper")

        observation = transition.next_observation if compute_next_action else transition.observation
        assert observation.state.ndim == 2, (
            f"Expected observation with a single batch dimension [B, state_dim], got shape {observation.state.shape}"
        )

        rng_sample, rng_select = jax.random.split(rng)
        batch_size = observation.state.shape[0]

        if self.policy_subsample and self.critic_action_horizon == 60 and self.action_horizon == 60:
            mask_30_30 = jnp.concatenate(
                [
                    jnp.ones((batch_size, 30), dtype = jnp.bool_),
                    jnp.zeros((batch_size, 30), dtype = jnp.bool_),
                ],
                axis = 1,
            )
            observation = dataclasses.replace(observation, action_mask = mask_30_30)
            transition = dataclasses.replace(
                transition,
                **({"next_observation": observation} if compute_next_action else {"observation": observation}),
            )
        elif self.action_horizon == 20 and self.critic_action_horizon == 20:
            # Matched-horizon, no subsample, no pad → every slot is a real action.
            mask_all_true = jnp.ones((batch_size, 20), dtype = jnp.bool_)
            observation = dataclasses.replace(observation, action_mask = mask_all_true)
            transition = dataclasses.replace(
                transition,
                **({"next_observation": observation} if compute_next_action else {"observation": observation}),
            )

        if self.base_model is not None:
            if _DEBUG:
                _log_model_inputs("policy", observation)
                # Probe a representative leaf of the underlying pi0/pi05 base model
                # so we can verify the restored params on the server match the
                # diagnostic's restored params bit-for-bit.
                if hasattr(self.base_model, "action_out_proj"):
                    aop_kernel = self.base_model.action_out_proj.kernel.value
                    aop_bias = self.base_model.action_out_proj.bias.value
                    jax.debug.print(
                        "[debug] base_model.action_out_proj/kernel shape={s} dtype={d} "
                        "mean={mn:.6e} std={st:.6e} min={mi:.6e} max={mx:.6e} sum={sm:.6e} "
                        "first8={f}",
                        s = aop_kernel.shape, d = aop_kernel.dtype,
                        mn = jnp.mean(aop_kernel.astype(jnp.float32)),
                        st = jnp.std(aop_kernel.astype(jnp.float32)),
                        mi = jnp.min(aop_kernel.astype(jnp.float32)),
                        mx = jnp.max(aop_kernel.astype(jnp.float32)),
                        sm = jnp.sum(aop_kernel.astype(jnp.float32)),
                        f = aop_kernel.reshape((-1,))[:8].astype(jnp.float32),
                    )
                    jax.debug.print(
                        "[debug] base_model.action_out_proj/bias shape={s} dtype={d} "
                        "mean={mn:.6e} std={st:.6e} sum={sm:.6e} value={v}",
                        s = aop_bias.shape, d = aop_bias.dtype,
                        mn = jnp.mean(aop_bias.astype(jnp.float32)),
                        st = jnp.std(aop_bias.astype(jnp.float32)),
                        sm = jnp.sum(aop_bias.astype(jnp.float32)),
                        v = aop_bias.astype(jnp.float32),
                    )
            # Two paths for picking the per-sample rngs that get fed to the
            # vmap'd policy.sample_actions call:
            #   - sample_rngs is None → legacy path: split `rng_sample` into
            #     ceil(num_samples / num_workers) keys per worker. Same rng on
            #     every host, so every host computes the same per-worker n
            #     candidates (no cross-worker diversity). Used by training /
            #     single-host eval.
            #   - sample_rngs is not None → caller hands in a sharded array of
            #     shape (N, ...) where N is leading-axis-sharded across the
            #     mesh's DATA_AXIS. Each chip gets a unique rng → unique noise.
            #     Used by the multi-host serve path to fan out diverse noise
            #     per chip without triggering the cross-host all-gather
            #     "Frankenstein" failure mode.
            if sample_rngs is None:
                num_workers = jax.process_count()
                n = max(1, -(-self.num_samples // num_workers))
                sample_rngs = jax.random.split(rng_sample, n)

            # Match main's vmap pattern: use `eqx.filter_vmap` with explicit
            # `in_axes=(0, None, None, None)`, passing `model`, `trans`, and
            # `next_action` as positional args (broadcast, not vmapped). This
            # avoids closing the NNX module / transition over `jax.vmap` —
            # which (under the previous `jax.vmap(sample_with_rng)(sample_rngs)`
            # closure pattern) was producing wrong outputs on the live server.
            @eqx.filter_vmap(in_axes = (0, None, None, None))
            def sample_with_rng(rng_i, model, trans, next_action):
                return model.sample_actions(
                    rng_i, trans, compute_next_action = next_action, **kwargs,
                )

            all_actions = sample_with_rng(sample_rngs, self.base_model, transition, compute_next_action)
            # all_actions shape: [N, B, ah, ad]
            all_actions = jnp.moveaxis(all_actions, 0, 1)  # [B, N, ah, ad]
            if _DEBUG:
                # Raw policy output in policy-normalized space, before the
                # critic-side renormalize / horizon-pad / action-dim slice.
                jax.debug.print(
                    f"[debug] policy raw_actions shape={all_actions.shape} sample0={{v}}",
                    v = all_actions[0, 0],
                )
            if self.convert_policy_actions_to_eef:
                # No clip here: the candidates are still in the policy's joint space,
                # and the clip that matters is the one applied after the conversion,
                # in the critic's range. Clipping joint values before forward
                # kinematics would also distort the resulting pose in a way the
                # offline caching pipeline (which converts straight after
                # Unnormalize + AbsoluteActions) never does.
                eval_actions = all_actions
            else:
                # Clip into the critic's training range for CRITIC SCORING ONLY —
                # `all_actions` (what we return for execution) stays at the raw
                # policy output. Cached counterfactuals are pre-clipped at load time.
                clip_bound = 1.25 if self.critic_use_quantile_norm else 5.0
                eval_actions = jnp.clip(all_actions, -clip_bound, clip_bound)
        else:
            # Use cached counterfactual actions (already pre-clipped).
            all_actions = self._get_cached_actions(
                transition, compute_next_action=compute_next_action
            )  # [B, N, ah, ad]
            eval_actions = all_actions

        n = all_actions.shape[1]
        action_horizon = all_actions.shape[2]

        # Policy and critic share norm stats + chunk-wise-delta format (asserted
        # in __init__), so the policy's normalized output is already in the
        # critic's space — only the action-dim slice (the policy may zero-pad
        # above the critic's action dim) and the horizon adapt remain.
        if self.critic_norm_stats is not None:
            critic_action_dim = self.critic_norm_stats["actions"].mean.shape[-1]
            if eval_actions.shape[-1] != critic_action_dim:
                if self.critic_action_dim_offset is None:
                    raise ValueError(
                        f"Action dim mismatch: input has {eval_actions.shape[-1]} but critic "
                        f"expects {critic_action_dim}. Set critic_action_dim_offset to specify "
                        f"which slice of the input to use."
                    )
                start = self.critic_action_dim_offset
                eval_actions = eval_actions[..., start : start + critic_action_dim]

        # Joint-space policy + EEF critic: carry the candidates into the critic's
        # unnormalized EEF delta space. Renormalization is deferred until after the
        # horizon adapt below so the critic's full-length stats apply to the padded
        # chunk in one go.
        state_eef = None
        if self.convert_policy_actions_to_eef:
            eval_actions, state_eef = self._to_eef(eval_actions, observation.state)

        # Adapt the policy chunk to the critic's expected horizon. Regimes are
        # supported (validated in __init__):
        #   - 60 Hz policy + 30 Hz critic (action_horizon=60, critic_action_horizon=50):
        #     stride-2 subsample (60 → 30 real actions), then zero-pad to 50,
        #     matching the fps_mask_30 mask the critic was trained on.
        #   - same-fps policy + critic (action_horizon=30, critic_action_horizon=50):
        #     skip the subsample, just zero-pad 30 → 50.
        #   - 60 Hz policy + 60-step critic (action_horizon=60, critic_action_horizon=60):
        #     stride-2 subsample (60 → 30), then zero-pad to 60 → action_mask
        #     30 True + 30 False. Equal horizons but still subsampled (unlike
        #     the 50/50 forward-as-is case), so it needs the explicit gate below.
        # In all cases the action_mask is also subsampled (when applicable) and
        # zero-padded with False so the critic only attends to the real actions.
        critic_action_mask = None
        subsample_60_60 = action_horizon == 60 and self.critic_action_horizon == 60
        if self.critic_action_horizon is not None and (
            self.critic_action_horizon != action_horizon or subsample_60_60
        ):
            if subsample_60_60:
                do_subsample = not self.policy_subsample
            else:
                do_subsample = 6 * self.critic_action_horizon == 5 * action_horizon
            if do_subsample:
                subsampled_actions = eval_actions[:, :, 1::2, :]
            else:
                subsampled_actions = eval_actions
            subsampled_horizon = subsampled_actions.shape[2]
            pad_amount = self.critic_action_horizon - subsampled_horizon
            eval_actions = jnp.pad(
                subsampled_actions,
                ((0, 0), (0, 0), (0, pad_amount), (0, 0)),
            )

            if observation.action_mask is not None:
                subsampled_mask = (
                    observation.action_mask[:, 1::2] if do_subsample else observation.action_mask
                )
            else:
                subsampled_mask = jnp.ones((batch_size, subsampled_horizon), dtype = jnp.bool_)
            subsampled_mask_expanded = jnp.repeat(
                subsampled_mask[:, None, :], n, axis = 1,
            ).reshape(batch_size * n, subsampled_horizon)
            critic_action_mask = jnp.pad(
                subsampled_mask_expanded,
                ((0, 0), (0, pad_amount)),
                constant_values = False,
            )

            action_horizon = self.critic_action_horizon

        # Normalize the EEF candidates into the critic's space. Padding first lets
        # the critic's full-length 2-D stats apply as-is: the padded rows are the
        # degenerate `q01 == q99 == 0` ones that `_sanitize_quantile_norm_stats`
        # expands to (-1, 1), so the zero padding still normalizes to 0.
        critic_state = None
        if state_eef is not None:
            normalize = _transforms.Normalize(
                self._conversion_stats["critic"], use_quantiles = self.critic_use_quantile_norm,
            )
            normalized = normalize({"actions": eval_actions, "state": state_eef})
            critic_clip_bound = 1.25 if self.critic_use_quantile_norm else 5.0
            eval_actions = jnp.clip(normalized["actions"], -critic_clip_bound, critic_clip_bound)
            critic_state = normalized["state"]

        # Fallback to policy obs when no critic override (shared norm stats make this safe).
        effective_critic_obs = critic_observation if critic_observation is not None else observation
        if critic_state is not None:
            # The joint-space state the policy pipeline normalized is meaningless to
            # an EEF critic. This critic has no_state=True so it never reads it, but
            # leaving it in place would silently mislead the next one that does.
            effective_critic_obs = dataclasses.replace(effective_critic_obs, state = critic_state)

        expanded_obs = expand_observation(effective_critic_obs, n)
        if critic_action_mask is not None:
            expanded_obs = dataclasses.replace(expanded_obs, action_mask = critic_action_mask)
        # actions to evaluate may have a different last dim than the policy actions when the
        # critic consumes a sliced subset of the policy's action vector.
        flat_actions = eval_actions.reshape(batch_size * n, action_horizon, eval_actions.shape[-1])

        if _DEBUG:
            _log_model_inputs("critic", expanded_obs, actions = flat_actions)

        # Critic prefix encodes images + prompt + state KVs. The critic was
        # trained on its OWN tokenized_prompt (different buffer length / vocab
        # than the policy's) and on critic-normalized state, so feeding the
        # policy's observation here would silently bake the policy's tokens +
        # policy-normalized state into the cached prefix and leave the critic
        # off-distribution. The caller supplies the critic's view via
        # `critic_observation`; falls back to `observation` when not provided.
        prefix_observation = effective_critic_obs

        prefix_cache = None
        network = getattr(value_function, "q_network", getattr(value_function, "network", None))
        # The sibling repo's value_function classes expose a top-level
        # `compute_prefix_cache` that delegates to the underlying network. The
        # in-repo SARSAValueFunction doesn't expose that wrapper, so checking
        # only the network's attribute and then calling on `value_function`
        # (which is what the sibling code does) raises AttributeError. Guard
        # both: only enter this branch when both the network supports the
        # KV-cache fast path AND the value_function exposes the wrapper call.
        if (
            network is not None
            and hasattr(network, "compute_prefix_cache")
            and hasattr(value_function, "compute_prefix_cache")
        ):
            # Only bootstrapping critics (SARSA, CQL) take `use_target`; a Monte-Carlo
            # critic has no target network and its wrapper rejects the keyword, so pass
            # it only when a target prefix is actually requested.
            if self.use_target_value:
                raw_kv_cache, raw_prefix_mask, raw_subtask_mask = value_function.compute_prefix_cache(
                    prefix_observation, use_target = True
                )
            else:
                raw_kv_cache, raw_prefix_mask, raw_subtask_mask = value_function.compute_prefix_cache(
                    prefix_observation
                )
            # Gemma 4 KV cache has batch at axis 0 (per-layer dict, 1-D end_index leaf);
            # gemma_2b stacks layers with batch at axis 1.
            kv_batch_axis = 1
            # Networks with a non-KV prefix cache (e.g. ResNet image features) declare their batch axis.
            kv_batch_axis = getattr(network, "prefix_cache_batch_axis", kv_batch_axis)
            repeated_kv_cache = jax.tree.map(
                lambda x: jnp.repeat(x, n, axis = kv_batch_axis), raw_kv_cache
            )
            repeated_prefix_mask = jnp.repeat(raw_prefix_mask, n, axis = 0)
            repeated_subtask_mask = (
                jnp.repeat(raw_subtask_mask, n, axis = 0) if raw_subtask_mask is not None else None
            )
            prefix_cache = (repeated_kv_cache, repeated_prefix_mask, repeated_subtask_mask)

        value_kwargs = {}
        if prefix_cache is not None:
            value_kwargs["prefix_cache"] = prefix_cache

        if self.use_target_value:
            result = value_function.compute_target_value(
                expanded_obs,
                flat_actions,
                **value_kwargs,
            )
        else:
            result = value_function.compute_value(
                expanded_obs,
                flat_actions,
                **value_kwargs,
            )
        q_values = result[0] if isinstance(result, tuple) else result
        q_values = q_values.reshape(batch_size, n)

        # Selection happens in the caller (host-side) so multi-host can
        # process_allgather + argmax across workers.
        return all_actions, q_values

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Delegate loss computation to base model (enables BC pre-training)."""
        if self.base_model is None:
            batch_size = actions.shape[0]
            return jnp.zeros((batch_size, self.action_horizon))
        return self.base_model.compute_loss(rng, observation, actions, train=train)

    @override
    def action_distribution(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool = False,
        value_function: _base_vf.BaseValueFunction | None = None,
        **kwargs,
    ):
        """Return action distribution.

        If base_model exists, delegates to it. Otherwise, returns a deterministic
        distribution at the argmax action from cached counterfactual actions.
        """
        if self.base_model is not None:
            return self.base_model.action_distribution(
                rng, transition, compute_next_action=compute_next_action, **kwargs
            )

        # No base model: return deterministic distribution at best cached action
        import distrax

        best_action, _ = self.select_best_action_and_q(
            rng, transition,
            compute_next_action=compute_next_action,
            value_function=value_function,
            **kwargs,
        )
        return distrax.Deterministic(loc=best_action)

    def select_best_action_and_q(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool,
        value_function: _base_vf.BaseValueFunction | None,
        **kwargs,
    ) -> tuple[at.Array, at.Array]:
        """Select the argmax-Q candidate from cached counterfactual actions.

        Returns (best_action[B, ah, ad], best_q[B]). The Q value is whichever
        flavor sample_actions used: target Q when use_target_value=True,
        online Q otherwise. Callers that need a *target*-Q value (e.g. the CQL
        Bellman backup) MUST check self.use_target_value before consuming
        best_q. Only supports base_model=None: with a base_model present,
        sample_actions samples fresh candidates per call and the q values
        returned here are not reusable in the same way.
        """
        if self.base_model is not None:
            raise ValueError(
                "select_best_action_and_q is only defined when base_model is None "
                "(used by the BestOfN-over-cached-counterfactual-actions training path)."
            )
        all_actions, q_values = self.sample_actions(
            rng, transition,
            compute_next_action=compute_next_action,
            value_function=value_function,
            **kwargs,
        )
        best_idx = jnp.argmax(q_values, axis = -1)
        best_action = jnp.take_along_axis(
            all_actions, best_idx[:, None, None, None], axis = 1,
        ).squeeze(1)
        best_q = jnp.take_along_axis(q_values, best_idx[:, None], axis = -1).squeeze(-1)
        return best_action, best_q


def expand_observation(observation: _model.Observation, num_samples: int) -> _model.Observation:
    """Repeat each observation num_samples times along the batch dimension.

    Given an observation with batch dimension B, produces an observation with
    batch dimension B*num_samples, where each original observation is repeated
    num_samples times contiguously.

    E.g., for B=2, N=3: [obs0, obs0, obs0, obs1, obs1, obs1]
    """

    def _repeat(x):
        if x is None:
            return None
        expanded = jnp.repeat(x[:, None], num_samples, axis=1)
        return expanded.reshape(x.shape[0] * num_samples, *x.shape[1:])

    return _model.Observation(
        images={k: _repeat(v) for k, v in observation.images.items()},
        image_masks={k: _repeat(v) for k, v in observation.image_masks.items()},
        state=_repeat(observation.state),
        tokenized_prompt=_repeat(observation.tokenized_prompt),
        tokenized_prompt_mask=_repeat(observation.tokenized_prompt_mask),
        token_ar_mask=_repeat(observation.token_ar_mask),
        token_loss_mask=_repeat(observation.token_loss_mask),
        action_mask=_repeat(observation.action_mask),
        subtask_start_index=_repeat(observation.subtask_start_index),
        subtask_end_index=_repeat(observation.subtask_end_index),
        subtask_id=_repeat(observation.subtask_id),
    )
