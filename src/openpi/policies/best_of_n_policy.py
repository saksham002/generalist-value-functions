"""Policy wrapper that loads a policy plus an optional best-of-N critic.

Exposes a `Policy`-compatible `infer(obs)` interface so the `WebsocketPolicyServer`
and its clients work unchanged when the server is launched with `--critic.*` flags.

Two modes:

- **Policy-only**: only `policy_*` args provided. The wrapper loads the policy
  through `load_model_utils.load_policy` and applies the explicit action-dim
  slice + output-transform path, so action-dim padding (32-D model output ->
  14-D bimanual EEF slot) is unwrapped consistently with the best-of-N path.
- **Best-of-N**: `critic_*` args also provided. Wraps the loaded policy in
  BestOfNWrapper, JITs the sample closure, and re-tokenizes the prompt with the
  critic's tokenizer at infer time. The policy and critic share norm stats and the
  chunk-wise-delta format, so the policy's normalized output is scored by the
  critic directly (no renormalization).

The returned `infer(obs)` dict is `{"actions": np.ndarray, "policy_timing":
{"infer_ms": float}, "q_values": np.ndarray | None}`; a client that only reads
`"actions"` keeps working.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import time
from typing import Any

import etils.epath as _epath
import flax.nnx as nnx
import jax
from jax.experimental import multihost_utils as _multihost
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.models.best_of_n import _DEBUG
from openpi.models.best_of_n import BestOfNWrapper
from openpi.models.best_of_n import _log_model_inputs
import openpi.models.model as _model
from openpi.policies import policy as _policy_module
from openpi.policies import subtask_decoder as _subtask_decoder
from openpi.rlds_utils.load_model_utils import LoadPolicyConfig
from openpi.rlds_utils.load_model_utils import load_critic as _load_critic
from openpi.rlds_utils.load_model_utils import load_policy as _load_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.value_functions import base_value_functions as _base_vf

logger = logging.getLogger(__name__)

INFERENCE_ITER_FILENAME = "inference_iter.txt"


def inference_iter_uri(checkpoint_dir: str) -> str:
    """Where the sampling counter lives for a served checkpoint.

    Beside the checkpoint root rather than inside a step directory: the counter belongs to
    the eval, not to any one step, and a step directory carries a commit marker that must
    keep describing exactly what training wrote.
    """
    return f"{checkpoint_dir.rstrip('/')}/{INFERENCE_ITER_FILENAME}"


def _read_inference_iter(path: str | None) -> int:
    if not path:
        return 0
    try:
        text = _epath.Path(path).read_text().strip()
    except Exception as e:  # absent on the first launch, which is not an error
        logger.info("No inference_iter at %s (%s); starting from 0", path, type(e).__name__)
        return 0
    try:
        return int(text)
    except ValueError:
        logger.warning("inference_iter at %s is not an int (%r); starting from 0", path, text)
        return 0


# Norm-stat keys the policy's Normalize / Unnormalize transforms reference at
# inference time. Mirrors LocalPolicy in the sibling repo.
_INFERENCE_NORM_KEYS = {"state", "actions", "next_state", "next_actions"}


# Keys whose dtype `broadcast_one_to_all` up-promotes during host transfer
# (e.g. uint8 -> uint32, bool -> int32) and that we have to cast back to keep
# Pi0 / BestOfN's jaxtyping annotations happy. Mapping is target-dtype.
_BROADCAST_RESTORE_DTYPES: dict[str, Any] = {
    # Observation fields
    "image": jnp.uint8,
    "image_mask": jnp.bool_,
    "tokenized_prompt_mask": jnp.bool_,
    "action_mask": jnp.bool_,
    "token_ar_mask": jnp.bool_,
    "token_loss_mask": jnp.bool_,
    # Critic-side extras
    "critic_token_mask": jnp.bool_,
    # Subtask-AR critic extras
    "decode_critic_token_mask": jnp.bool_,
    "subtask_start_index": jnp.int32,
    "subtask_end_index": jnp.int32,
}


def _restore_inference_dtypes(d: dict[str, Any]) -> dict[str, Any]:
    """Cast known fields back to their expected dtypes after a broadcast."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        target = _BROADCAST_RESTORE_DTYPES.get(k)
        if target is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {dk: dv.astype(target) if hasattr(dv, "astype") else dv for dk, dv in v.items()}
        elif hasattr(v, "astype"):
            out[k] = v.astype(target)
        else:
            out[k] = v
    return out


def _compute_acs(actions: np.ndarray) -> float:
    """Average cosine similarity across a BestOfN candidate action pool.

    `actions` is the (n, ah, ad) pool of candidate action chunks. For every
    unordered pair of the n candidates, the per-timestep cosine similarity
    between the two chunks is averaged across the ah horizon; those C(n, 2)
    pair scalars are then averaged into a single value. Returns NaN when n < 2.
    """
    n = actions.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(actions, axis = -1, keepdims = True)
    unit = actions / np.clip(norms, 1e-8, None)
    # per_timestep_cos[i, j, t] = <unit[i, t], unit[j, t]>
    per_timestep_cos = np.einsum("itd,jtd->ijt", unit, unit)
    cos_per_pair = per_timestep_cos.mean(axis = -1)  # (n, n): mean over horizon
    upper = np.triu_indices(n, k = 1)  # unique unordered pairs
    return float(cos_per_pair[upper].mean())


def _build_critic_kwargs(critic_config: Any) -> dict[str, Any]:
    """Pack the critic-side attributes BestOfNWrapper needs from the TrainConfig.

    Uses `getattr` for `use_quantile_norm` so a data factory without that field
    falls back to the default (False).
    """
    # SARSAValueFunctionConfig names the backbone `network_config`; CQLValueFunctionConfig
    # uses `q_network_config`. Resolve either so BestOfN works with both critic flavors.
    network_config = getattr(critic_config.model, "network_config", None)
    if network_config is None:
        network_config = getattr(critic_config.model, "q_network_config", None)
    assert network_config is not None, (
        f"Critic model {type(critic_config.model).__name__} has neither "
        "`network_config` (SARSA) nor `q_network_config` (CQL)."
    )
    return {
        # FineTuneConfig overrides land on TrainConfig.action_horizon; fall back
        # to the model's action_horizon when the TrainConfig field is unset.
        "action_horizon": critic_config.action_horizon or critic_config.model.action_horizon,
        "use_chunk_wise_delta": critic_config.data.use_chunk_wise_delta,
        "subsample": getattr(critic_config.data, "subsample", False),
        "use_quantile_norm": getattr(critic_config.data, "use_quantile_norm", False),
        "tokenizer": network_config.get_tokenizer(),
    }


class BestOfNPolicy(_base_policy.BasePolicy):
    """Loads a policy (and optional critic) and serves an `infer(obs)` API over the websocket server."""

    def __init__(
        self,
        policy_config_name: str,
        policy_checkpoint_dir: str,
        *,
        policy_step: int | None = None,
        policy_fine_tune_config: str | None = None,
        policy_task_description: str | None = None,
        critic_config_name: str | None = None,
        critic_checkpoint_dir: str | None = None,
        critic_step: int | None = None,
        critic_fine_tune_config: str | None = None,
        num_samples: int = 8,
        default_prompt: str | None = None,
        return_q_values: bool = True,
        prewarm: bool = True,
        sample_parallel: bool = False,
        fsdp_devices: int | None = None,
        subtask_decode_every: int = 20,
        num_steps: int | None = None,
        inference_iter_path: str | None = None,
        start_inference_iter: int | None = None,
    ) -> None:
        # Both critic-config and critic-checkpoint must be provided together.
        critic_args_set = (critic_config_name is not None) or (critic_checkpoint_dir is not None)
        if critic_args_set and (critic_config_name is None or critic_checkpoint_dir is None):
            raise ValueError(
                "BestOfNPolicy: critic_config_name and critic_checkpoint_dir must be "
                "provided together (both or neither)."
            )
        self._return_q_values = return_q_values

        # Subtask-AR decoding state (populated in the critic-load block when the
        # critic's value network has predict_subtask_ar=True). Defaults keep the
        # feature off for non-subtask_ar critics and the policy-only path. The
        # policy always uses the client-sent prompt; the decoded subtask only
        # conditions the critic's value prompt.
        self._subtask_decode_every = int(subtask_decode_every)
        self._critic_predict_subtask_ar = False
        self._critic_uses_subtask_id = False
        self._subtask_decoder: _subtask_decoder.SubtaskPredictor | None = None
        self._cached_subtask_str: str | None = None
        self._subtask_iter = 0

        # sample_parallel mode (opt-in): shard the per-sample rngs (and JIT
        # outputs) along BATCH_AXIS so num_samples unique candidates are
        # produced (1 per batch position when num_samples == BATCH_AXIS, or
        # num_samples/BATCH_AXIS per position batched via vmap when num_samples
        # > BATCH_AXIS). JIT outputs are with_sharding_constraint'd to fully
        # replicated so host gather is a one-shard read.
        #
        # Default fsdp_devices selection — caller can override via the
        # `fsdp_devices` kwarg. The override is useful when the auto-computed
        # value (device_count // num_samples) is too small to fit the
        # restored params (e.g. v5e-32 with N=8 → auto fsdp=4 puts the policy +
        # critic params on each chip at 4× the per-chip footprint of the
        # saved fsdp=16 sharding, which exceeds 16 GB HBM and OOMs).
        device_count = jax.device_count()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self._sample_parallel = bool(sample_parallel)
        if self._sample_parallel:
            if fsdp_devices is None:
                # Auto: when num_samples >= device_count, FSDP=1 (each chip a
                # batch position); when num_samples < device_count, FSDP =
                # device_count/num_samples (one chip per FSDP shard per sample).
                if num_samples >= device_count:
                    fsdp_override = 1
                else:
                    if device_count % num_samples != 0:
                        raise ValueError(
                            f"sample_parallel=True auto fsdp requires device_count "
                            f"({device_count}) divisible by num_samples ({num_samples}); "
                            f"pass fsdp_devices explicitly to use a different mesh."
                        )
                    fsdp_override = device_count // num_samples
            else:
                if fsdp_devices <= 0 or device_count % fsdp_devices != 0:
                    raise ValueError(
                        f"sample_parallel fsdp_devices override ({fsdp_devices}) must be a "
                        f"positive divisor of device_count ({device_count})."
                    )
                fsdp_override = fsdp_devices
            batch_axis_size = device_count // fsdp_override
            self._fsdp_devices_override: int | None = fsdp_override
            self._sample_parallel_batch_axis = batch_axis_size
            logger.info(
                "BestOfNPolicy sample_parallel=True: num_samples=%d, mesh=(%d batch, %d fsdp), "
                "device_count=%d, fsdp_devices_override=%s",
                num_samples, batch_axis_size, fsdp_override, device_count,
                "auto" if fsdp_devices is None else str(fsdp_devices),
            )
        else:
            # Honor the caller's fsdp_devices even when sample_parallel=False —
            # it's still consumed by LoadPolicyConfig below to pick the load
            # mesh (and to skip the same-topology fast path in load_policy when
            # the saved sharding metadata can't be re-used as-is).
            self._fsdp_devices_override = fsdp_devices
            self._sample_parallel_batch_axis = None

        # ---- Load policy + build the Policy with full transforms.
        logger.info(f"Loading policy '{policy_config_name}' from {policy_checkpoint_dir} (step={policy_step})...")
        load_config = LoadPolicyConfig(
            config_name = policy_config_name,
            checkpoint_path = policy_checkpoint_dir,
            fine_tune = policy_fine_tune_config,
            step = policy_step,
            fsdp_devices = self._fsdp_devices_override,
        )
        model, config = _load_policy(load_config)
        logger.info("Policy checkpoint restored.")

        # The policy model lives on `config.policy` for IQL/SARSA-style
        # multi-component configs, otherwise on `config.model` directly.
        policy_model_config = config.policy if config.policy is not None else config.model

        data_config = config.data.create(config.assets_dirs, policy_model_config)
        asset_id = data_config.asset_id
        if policy_step is not None:
            step_dir = str(policy_step)
        else:
            step_dirs = sorted(
                (d for d in os.listdir(policy_checkpoint_dir) if d.isdigit()),
                key = int,
            )
            step_dir = step_dirs[-1]
            logger.info(f"No policy step specified, using latest step {step_dir}.")
        norm_stats_dir = os.path.join(policy_checkpoint_dir, step_dir, "assets", asset_id)
        logger.info(f"Loading policy norm stats from {norm_stats_dir}")
        all_norm_stats = _normalize.load(norm_stats_dir)
        norm_stats = {k: v for k, v in all_norm_stats.items() if k in _INFERENCE_NORM_KEYS}
        # BC-trained policies (critic_mode=False data configs, e.g.
        # LeRobotRldsDataConfig) save norm stats without the next_state /
        # next_actions aliases that critic-mode stats carry. BestOfNWrapper
        # asserts identical core key sets, so replicate the aliasing the
        # critic-mode configs apply (next_* stats are pure aliases of
        # state/actions everywhere in this repo).
        if "state" in norm_stats and "next_state" not in norm_stats:
            norm_stats["next_state"] = norm_stats["state"]
        if "actions" in norm_stats and "next_actions" not in norm_stats:
            norm_stats["next_actions"] = norm_stats["actions"]

        policy = _policy_module.Policy(
            model,
            transforms = [
                _transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
                *(
                    [_transforms.Clip(data_config.clip_normalized_bounds)]
                    if data_config.clip_normalized_bounds is not None
                    else []
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms = [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            metadata = config.policy_metadata,
        )

        # Borrow the prepared transform pipeline + model from the Policy
        # instance (mirrors LocalPolicy). This avoids reimplementing the
        # transform composition while still letting us slice the action_dim
        # padding before Unnormalize the way the sibling does.
        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
        # Override the flow-matching integration step count. Only consumed by the
        # policy-only (BC) sampling path; the BestOfN-with-critic path samples
        # via the BestOfN model, which uses the model's default num_steps.
        if num_steps is not None:
            self._sample_kwargs["num_steps"] = num_steps
        self._rng = policy._rng  # noqa: SLF001
        # Surface image sizes + critic-image flag to client metadata for EvalImageHelper.
        self._metadata = dict(policy.metadata)
        # image_size may be a scalar int or an (H, W) tuple. Normalize to a [H, W]
        # list either way.
        _img_size = getattr(config.data, "image_size", 224)
        if isinstance(_img_size, int):
            _img_size = (_img_size, _img_size)
        self._metadata["policy_image_size"] = list(_img_size)
        # Surface the policy's training-time subsample flag so eval clients can
        # set env_control_freq / replan_steps to match without keying off the
        # config name (subsample policy → 30 Hz env + 15-step replan).
        self._metadata["policy_subsample"] = bool(getattr(config.data, "subsample", False))

        # Policy.__init__ skips this on the JAX path; force eval-mode here.
        self._model.eval()

        # Disable classifier-free guidance at eval time (matches LocalPolicy).
        if hasattr(self._model, "config") and hasattr(self._model.config, "guidance"):
            object.__setattr__(self._model.config, "guidance", 0.0)
            logger.info("Disabled classifier-free guidance (guidance=0.0).")

        # Wrap sample_actions so the debug print runs INSIDE the JIT graph.
        # Outside the JIT the actions tensor is multi-host sharded with empty
        # shards on some processes, and `jax.debug.print` fails on `device_put`
        # of a non-fully-addressable array. Inside the graph `jax.debug.print`
        # handles sharded arrays correctly (same as the BestOfN.debug path in
        # best_of_n.py). Mirrors the `_bestofn_sample` pattern: pass the model
        # as a positional argument to `@nnx.jit` rather than capture by closure
        # so the NNX state is sharded correctly.
        if _DEBUG:
            @nnx.jit
            def _sample_with_debug(model, rng_in, transition_in, noise = None):
                # ---- DIAGNOSTIC: fingerprints INSIDE the JIT graph for the
                # actual model inputs. Lets us detect any divergence between
                # V9 and V4 paths at the MODEL boundary even if the batched
                # dict was bit-identical at the host-side boundary.
                obs_in = transition_in.observation
                jax.debug.print(
                    "[mdl.debug] state sum={s:.6e}  rng={r}",
                    s = jnp.sum(obs_in.state.astype(jnp.float64)),
                    r = jax.random.key_data(rng_in),
                )
                # Image is (1, 224, 224, 3) → flatten size = 150528. Offsets
                # below all fall within [0, 150527]. Two images with same sum
                # but different content will diverge in these specific pixel
                # values.
                _img_offsets = jnp.array([0, 100, 1000, 10000, 50000, 100000, 140000, 150527])
                for cam in sorted(obs_in.images.keys()):
                    img_f64 = obs_in.images[cam].astype(jnp.float64)
                    img_flat = img_f64.reshape(-1)
                    jax.debug.print(
                        "[mdl.debug] image[{c}] sum={s:.6e}  shape={sh}",
                        c = cam,
                        s = jnp.sum(img_f64),
                        sh = obs_in.images[cam].shape,
                    )
                    jax.debug.print(
                        "[mdl.debug] image[{c}] pixels @ [0,100,1000,10000,50000,100000,140000,150527] = {v}",
                        c = cam,
                        v = img_flat[_img_offsets],
                    )
                if obs_in.tokenized_prompt is not None:
                    jax.debug.print(
                        "[mdl.debug] tokenized_prompt sum={s}  shape={sh}",
                        s = jnp.sum(obs_in.tokenized_prompt),
                        sh = obs_in.tokenized_prompt.shape,
                    )
                if obs_in.action_mask is not None:
                    jax.debug.print(
                        "[mdl.debug] action_mask sum={s}  shape={sh}",
                        s = jnp.sum(obs_in.action_mask.astype(jnp.int32)),
                        sh = obs_in.action_mask.shape,
                    )
                # action_out_proj kernel fingerprint inside JIT — proves whether
                # the loaded params' state differs between V9 and V4 calls.
                if hasattr(model, "action_out_proj"):
                    jax.debug.print(
                        "[mdl.debug] action_out_proj/kernel sum={s:.6e}  bias_sum={b:.6e}",
                        s = jnp.sum(model.action_out_proj.kernel.value.astype(jnp.float64)),
                        b = jnp.sum(model.action_out_proj.bias.value.astype(jnp.float64)),
                    )
                if hasattr(model, "action_in_proj"):
                    jax.debug.print(
                        "[mdl.debug] action_in_proj/kernel sum={s:.6e}  bias_sum={b:.6e}",
                        s = jnp.sum(model.action_in_proj.kernel.value.astype(jnp.float64)),
                        b = jnp.sum(model.action_in_proj.bias.value.astype(jnp.float64)),
                    )
                # Aggregate fingerprints over the entire NNX state — if these
                # differ between V9 and V4 calls despite the same model object,
                # that's the bug. Use abs-sum + sum-of-squares to catch sign-
                # cancelling differences. Only iterate float-leaves with
                # `.value`.
                _state_leaves = [
                    v.value for v in jax.tree.leaves(nnx.state(model))
                    if hasattr(v, "value") and hasattr(v.value, "shape")
                    and jnp.issubdtype(v.value.dtype, jnp.floating)
                ]
                jax.debug.print(
                    "[mdl.debug] full state: #leaves={n}  abs_sum={a:.6e}  sq_sum={s:.6e}",
                    n = len(_state_leaves),
                    a = sum(jnp.sum(jnp.abs(x.astype(jnp.float64))) for x in _state_leaves),
                    s = sum(jnp.sum(x.astype(jnp.float64) * x.astype(jnp.float64)) for x in _state_leaves),
                )

                # ALSO fingerprint the FULL transition (not just observation
                # fields we logged). Catches discrepancies in transition.action,
                # transition.next_observation, transition.next_action, etc.
                _trans_leaves = [
                    leaf for leaf in jax.tree.leaves(transition_in)
                    if hasattr(leaf, "shape") and jnp.issubdtype(leaf.dtype, jnp.floating)
                ]
                _trans_int_leaves = [
                    leaf for leaf in jax.tree.leaves(transition_in)
                    if hasattr(leaf, "shape") and (jnp.issubdtype(leaf.dtype, jnp.integer)
                                                   or jnp.issubdtype(leaf.dtype, jnp.bool_))
                ]
                jax.debug.print(
                    "[mdl.debug] transition: #float_leaves={nf}  abs_sum={a:.6e}  #int_leaves={ni}  int_sum={i}",
                    nf = len(_trans_leaves),
                    a = sum(jnp.sum(jnp.abs(x.astype(jnp.float64))) for x in _trans_leaves),
                    ni = len(_trans_int_leaves),
                    i = sum(jnp.sum(x.astype(jnp.int64)) for x in _trans_int_leaves),
                )
                actions = (
                    model.sample_actions(rng_in, transition_in, noise = noise)
                    if noise is not None
                    else model.sample_actions(rng_in, transition_in)
                )
                jax.debug.print(
                    f"[debug] policy raw_actions shape={actions.shape} sample0={{v}}",
                    v = actions[0],
                )
                return actions

            def _sample_jit_wrapper(rng_in, transition_in, **sample_kwargs):
                return _sample_with_debug(
                    self._model, rng_in, transition_in,
                    noise = sample_kwargs.get("noise"),
                )

            self._sample_actions_jit = _sample_jit_wrapper
        else:
            self._sample_actions_jit = nnx_utils.module_jit(self._model.sample_actions)

        # action_dim_offset + eef_action_dim define the slice from the model's
        # 32-D padded output back to the 14-D bimanual EEF block. norm_stats
        # for "actions" is stored at the unpadded (14) dim.
        self._action_dim_offset = policy_model_config.action_dim_offset
        self._eef_action_dim = norm_stats["actions"].mean.shape[-1]
        # State dim for the prewarm dummy (norm_stats["state"] is the
        # authoritative shape; fall back to the action dim if absent).
        self._state_dim = norm_stats["state"].mean.shape[-1]

        # JAX rank 0 binds the websocket; every other rank runs
        # participate_loop so the JIT'd inference doesn't deadlock waiting
        # for collectives. Rank → TPU worker IP mapping is deterministic
        # per pod; serve_policy.py logs it at startup.
        self._process_index = jax.process_index()
        self._num_processes = jax.process_count()
        self._is_multi_host = self._num_processes > 1
        # Seeded rather than always zero: the flow-matching x_T for call n is
        # fold_in(rng, num_processes * n), so a server that restarts after preemption and
        # resumes an eval would otherwise replay the remaining episodes with the noise of
        # calls 0..k instead of continuing the sequence. Explicit value wins; otherwise the
        # counter persisted by the preemption handler is picked up.
        self._inference_iter_path = inference_iter_path
        if start_inference_iter is not None:
            self._inference_iter = int(start_inference_iter)
        else:
            self._inference_iter = _read_inference_iter(inference_iter_path)
        if self._inference_iter:
            logger.info("Starting inference_iter at %d", self._inference_iter)
        self._install_preemption_handler()

        # Mesh used for the FSDP-sharded JIT call. Re-built from `config.fsdp_devices`
        # — defaults to 16 (saved checkpoint topology). With sample_parallel=True
        # config.fsdp_devices was set to device_count // num_samples so the
        # BATCH_AXIS exactly matches num_samples and the rng/leading axis maps
        # 1:1 onto batch positions with no padding.
        from openpi.training import sharding as _sharding_mod
        self._sharding_mod = _sharding_mod
        self._mesh = _sharding_mod.make_mesh(config.fsdp_devices)
        if self._sample_parallel:
            # sample_rngs.shape[0] must be a multiple of BATCH_AXIS so the
            # leading axis shards cleanly: the BATCH_AXIS either matches
            # num_samples (1 per batch position, parallel) or divides it
            # (multiple per batch position, sequential vmap). When num_samples
            # is not a multiple (e.g. the n=1 BC-only phase on a (4 batch,
            # 16 fsdp) v5e-64 mesh), pad up to the next multiple: the JIT
            # samples + scores num_samples_padded candidates on the trained
            # fsdp topology and the host-side `[:n]` slice in `infer` keeps
            # only the first num_samples. (Raising fsdp_devices to
            # device_count instead compiles a fresh mesh topology, which hit
            # cross-host XLA-compilation nondeterminism — "unexpected peer in
            # the launch group" — on v5e-64.)
            batch_axis = self._sample_parallel_batch_axis
            self._num_samples_padded = ((num_samples + batch_axis - 1) // batch_axis) * batch_axis
            if self._num_samples_padded != num_samples:
                logger.info(
                    "BestOfNPolicy sample_parallel: padding num_samples %d -> %d (next multiple "
                    "of BATCH_AXIS=%d); extra candidates are sampled+scored then discarded.",
                    num_samples, self._num_samples_padded, batch_axis,
                )
        else:
            # Pad num_samples to a multiple of jax.device_count() so a single
            # PartitionSpec(DATA_AXIS, None) cleanly splits the leading axis across
            # all chips (1 unique sample per chip). User's num_samples is the lower
            # bound; the critic ends up scoring `num_samples_padded` candidates and
            # we argmax over all of them downstream.
            self._num_samples_padded = ((num_samples + device_count - 1) // device_count) * device_count
            if self._num_samples_padded != num_samples:
                logger.info(
                    "BestOfNPolicy: padding num_samples %d -> %d (next multiple of device_count=%d) "
                    "so the per-sample rngs cleanly shard along DATA_AXIS.",
                    num_samples, self._num_samples_padded, device_count,
                )

        # Cache the obs schema (shapes / dtypes) so participating ranks can
        # construct dummies of the exact structure broadcast_one_to_all expects.
        # PaligemmaTokenizer writes max_token_len at construction time; read it off
        # the model.
        self._max_token_len = self._model.max_token_len
        self._policy_task_description = policy_task_description
        # Critic-side prompt override; set below after critic load when the
        # critic's data factory uses prompt_mode="task_description_predict_current_subtask"
        # (training prompt = constant task description; the subtask text is only
        # used for the auxiliary next-token loss, not for the input prompt).
        self._critic_task_description: str | None = None
        self._image_size = 224  # The eval transforms always emit 224x224.
        self._image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        # Critic prompt length (only used when BestOfN + critic are active);
        # known after critic load below. Default to the policy len in the
        # absence of a critic so participate_loop's dummies still work.
        self._critic_max_token_len = self._max_token_len

        # ---- Optional: load critic + wrap in BestOfN.
        self._critic_model = None
        self._critic_tokenizer = None
        self._bestofn_sample = None
        self._bestofn = None
        self._policy_data_config = data_config
        self._policy_subsample = getattr(config.data, "subsample", False)
        # Prompt modes gate the per-call task_description override below.
        self._policy_prompt_mode = (
            getattr(config.data, "prompt_mode", None)
            or getattr(config.data, "subtask_prompt_mode", None)
        )
        self._critic_prompt_mode: str | None = None

        if critic_args_set:
            logger.info(
                f"Loading critic '{critic_config_name}' from {critic_checkpoint_dir} "
                f"(step={critic_step}, fine_tune={critic_fine_tune_config})..."
            )
            critic_model, critic_norm_stats, critic_config, _ = _load_critic(
                critic_config_name,
                critic_checkpoint_dir,
                fine_tune = critic_fine_tune_config,
                step = critic_step,
                fsdp_devices = self._fsdp_devices_override if self._fsdp_devices_override is not None else 16,
            )
            critic_kwargs = _build_critic_kwargs(critic_config)
            # Critic-side image_size: surfaced to clients + used to shape dummies
            # (HLO must match across hosts). SARSAValueFunctionConfig exposes the
            # PaliGemma config as `network_config`; CQLValueFunctionConfig exposes
            # it as `q_network_config` — accept either.
            # _build_critic_kwargs above has already asserted one of these exists.
            critic_network_config = getattr(critic_config.model, "network_config", None)
            if critic_network_config is None:
                critic_network_config = critic_config.model.q_network_config
            critic_image_size = getattr(critic_network_config, "image_size", None)
            if critic_image_size is not None:
                self._metadata["critic_image_size"] = list(critic_image_size)
            self._critic_model = critic_model
            self._critic_model.eval()
            self._critic_tokenizer = critic_kwargs["tokenizer"]
            # Encode the "null" sentinel the same way the dynamic-prompt critic
            # path tokenizes a live prompt (see _prepare_inputs_rank0), so the
            # BestOfNWrapper can recognize a null-prompt step and skip value-based
            # selection.
            # A critic without a text pathway (ResNet network, get_tokenizer() -> None)
            # detects the null prompt on the raw string instead and predicts its own
            # categorical subtask id from the images.
            if self._critic_tokenizer is not None:
                _null_tokens, _ = self._critic_tokenizer.tokenize("null", None)
                self._critic_null_prompt_tokens = np.asarray(_null_tokens)
            else:
                self._critic_null_prompt_tokens = None
                logger.info("Critic has no tokenizer; prompt tokens are zero placeholders for the critic.")
            # Critic prompt token length (determines the dummy shape we
            # broadcast on participating workers).
            self._critic_max_token_len = getattr(critic_network_config, "max_token_len", self._critic_max_token_len)

            # Auto-route the constant task description into critic tokenization
            # when the critic was trained with prompt_mode="task_description_predict_current_subtask".
            # In that training mode the input prompt was the task description (the
            # subtask text only appears in the next-token auxiliary loss labels),
            # so at eval time we must feed the same task description rather than
            # the client's dynamic subtask.
            critic_prompt_mode = getattr(critic_config.data, "prompt_mode", None)
            self._critic_prompt_mode = critic_prompt_mode
            # prompt_mode="task_description" trains on the constant task string too (via
            # plain TokenizePrompt), so it needs the same routing or the critic would be
            # scored on the client's dynamic subtask instead.
            if (
                critic_prompt_mode in ("task_description", "task_description_predict_current_subtask")
                and policy_task_description is not None
            ):
                self._critic_task_description = policy_task_description
                logger.info(
                    f"Critic prompt_mode={critic_prompt_mode!r}: routing constant "
                    f"task_description into critic tokenization."
                )

            # A critic that predicts its own subtask was trained with Q conditioned on
            # it: a predict_subtask_ar network on `task_description + current_subtask
            # + "\n"` (the subtask suffix visible to CLS/action), a categorical network
            # on the embedded `subtask_id`. At eval we predict the current subtask off
            # the same observation every `subtask_decode_every` calls and feed the
            # cached prediction back into the critic observation -- as prompt suffix
            # or as id, whichever the network reads -- so the Q matches training. A
            # critic with neither head leaves the decoder None and the value prompt
            # task-only (unchanged).
            _critic_net = _subtask_decoder.critic_value_network(self._critic_model)
            _critic_net_config = getattr(_critic_net, "config", None)
            self._critic_predict_subtask_ar = bool(getattr(_critic_net_config, "predict_subtask_ar", False))
            self._critic_uses_subtask_id = bool(getattr(_critic_net_config, "uses_subtask_id", False))
            self._subtask_decoder = _subtask_decoder.build_subtask_predictor(
                self._critic_model, self._critic_tokenizer,
                decode_every = self._subtask_decode_every, max_tokens = 16,
            )
            if self._subtask_decoder is not None:
                logger.info(
                    f"Critic predicts its subtask ({type(self._subtask_decoder).__name__}): "
                    f"subtask prediction enabled (decode_every={self._subtask_decode_every})."
                )

            # `use_chunk_wise_delta` lives on the data FACTORY (RoboCoinRldsDataConfig
            # / Hdf5RldsDataConfig), not the runtime DataConfig instance, so
            # we read it from `config.data`. BestOfNWrapper asserts the policy
            # and critic values match (and that their norm stats are identical).
            policy_use_chunk_wise_delta = config.data.use_chunk_wise_delta
            critic_use_chunk_wise_delta = critic_kwargs["use_chunk_wise_delta"]
            critic_subsample = critic_kwargs["subsample"]

            # A joint-space policy scored by an EEF critic (e.g. lego_pi05_subtask vs
            # lego_paligemma_cql_rlds_finetune_subtask_ar) needs its candidates carried
            # through forward kinematics before the critic sees them; the wrapper does
            # this in-graph. The reverse has no inverse-kinematics path.
            policy_use_eef = getattr(config.data, "use_eef", False)
            critic_use_eef = getattr(critic_config.data, "use_eef", False)
            if policy_use_eef and not critic_use_eef:
                raise ValueError(
                    "Policy is EEF-space but the critic is joint-space; there is no "
                    "inverse-kinematics path to convert candidates for scoring."
                )
            convert_policy_actions_to_eef = critic_use_eef and not policy_use_eef

            resolved_offset = self._action_dim_offset
            critic_action_dim = critic_norm_stats["actions"].mean.shape[-1]
            policy_action_horizon = self._model.action_horizon
            critic_action_horizon = critic_kwargs["action_horizon"]
            logger.info(
                f"Building BestOfNWrapper (num_samples={num_samples}, "
                f"action_dim_offset={resolved_offset}, "
                f"critic_action_dim={critic_action_dim} (policy action_dim={self._model.action_dim}), "
                f"policy_action_horizon={policy_action_horizon}, "
                f"critic_action_horizon={critic_action_horizon}, "
                f"policy_use_chunk_wise_delta={policy_use_chunk_wise_delta}, "
                f"critic_use_chunk_wise_delta={critic_use_chunk_wise_delta}, "
                f"convert_policy_actions_to_eef={convert_policy_actions_to_eef})"
            )
            if convert_policy_actions_to_eef:
                logger.info(
                    "Joint-space policy + EEF critic: candidates are converted via YAM "
                    "forward kinematics before scoring. Executed actions stay joint-space, "
                    "so the logged acs is a joint-space similarity."
                )
            wrapper_critic_kwargs = {
                "use_quantile_norm": critic_kwargs["use_quantile_norm"],
                "subsample": critic_subsample,
                "policy_subsample": self._policy_subsample,
                "action_dim_offset": resolved_offset,
                "action_horizon": critic_action_horizon,
            }
            self._bestofn = BestOfNWrapper(
                action_dim = self._model.action_dim,
                action_horizon = self._model.action_horizon,
                max_token_len = self._model.max_token_len,
                base_model = self._model,
                num_samples = num_samples,
                use_target_value = False,
                policy_norm_stats = norm_stats,
                critic_norm_stats = critic_norm_stats,
                policy_use_chunk_wise_delta = policy_use_chunk_wise_delta,
                critic_use_chunk_wise_delta = critic_use_chunk_wise_delta,
                critic_kwargs = wrapper_critic_kwargs,
                convert_policy_actions_to_eef = convert_policy_actions_to_eef,
                policy_use_quantile_norm = data_config.use_quantile_norm,
            )

            # Capture for closure (avoids referencing `self` inside JIT body):
            _mesh = self._mesh
            _sample_parallel = self._sample_parallel
            _critic_predict_subtask_ar = self._critic_predict_subtask_ar
            _critic_uses_subtask_id = self._critic_uses_subtask_id

            @nnx.jit
            def _bestofn_sample(bon, vf, rng, sample_rngs, transition, critic_prompt, critic_prompt_mask, critic_is_null_prompt, subtask_start_index, subtask_end_index, critic_subtask_id):
                # BestOfNWrapper.sample_actions returns (all_actions, q_values).
                # `rng` is replicated;
                # `sample_rngs` is leading-axis-sharded so each parallel sample
                # group gets unique initial noise. Forwarded as the optional
                # `sample_rngs=` kwarg so the wrapper skips its in-JIT split and
                # uses these directly.
                # Critic view: critic-tokenized prompt, shared images and image_masks.
                replace_kwargs = {
                    "tokenized_prompt": critic_prompt,
                    "tokenized_prompt_mask": critic_prompt_mask,
                }
                # For a predict_subtask_ar critic, set the subtask span so the
                # critic conditions Q on the (decoded) subtask suffix — matching
                # training. The flag is a Python constant captured here, so the
                # non-subtask_ar graph never references the indices and stays
                # byte-identical to before.
                if _critic_predict_subtask_ar:
                    replace_kwargs["subtask_start_index"] = subtask_start_index
                    replace_kwargs["subtask_end_index"] = subtask_end_index
                # The categorical analog: the cached predicted id conditions Q, and the
                # network's -1 sentinel means "none cached yet, resolve it from the images".
                if _critic_uses_subtask_id:
                    replace_kwargs["subtask_id"] = critic_subtask_id
                critic_obs = dataclasses.replace(transition.observation, **replace_kwargs)
                all_actions, q_values = bon.sample_actions(
                    rng, transition, compute_next_action = False,
                    value_function = vf,
                    critic_observation = critic_obs,
                    sample_rngs = sample_rngs,
                )
                # Detected host-side to keep the raw token array off the JIT'd module.
                q_values = jnp.where(critic_is_null_prompt, jnp.zeros_like(q_values), q_values)
                if _sample_parallel:
                    # Replicate outputs across all chips so host gather just reads
                    # one addressable shard — no cross-host process_allgather on
                    # the candidate axis.
                    replicated = jax.sharding.NamedSharding(_mesh, jax.sharding.PartitionSpec())
                    all_actions = jax.lax.with_sharding_constraint(all_actions, replicated)
                    q_values = jax.lax.with_sharding_constraint(q_values, replicated)
                return all_actions, q_values

            self._bestofn_sample = _bestofn_sample
            logger.info("Critic loaded; BestOfN sample closure JIT-registered.")

        if prewarm:
            self._prewarm_jit()

        logger.info("BestOfNPolicy ready.")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Run a single inference step on rank 0 (or single-host).

        Mirrors LocalPolicy.predict but is callable from WebsocketPolicyServer
        (single-arg dict input, dict output).

        Multi-host: this method is called only on rank 0 (the websocket-bound
        host). It builds the inference inputs, broadcasts them to all other
        hosts via `jax.experimental.multihost_utils.broadcast_one_to_all`, and
        then triggers the JIT'd sampling closure. Ranks 1..N-1 are sitting in
        `participate_loop` and will reach the same broadcast in lockstep so
        the JAX collective ops complete.
        """
        # Make a copy since transforms mutate in place. Capture the prompt
        # string BEFORE _input_transform runs (TokenizePrompt pops it) so we
        # can re-tokenize for the critic.
        obs = jax.tree.map(lambda x: x, obs)
        batched, extras = self._prepare_inputs_rank0(obs, noise = noise)
        # Subtask the critic's value prompt was built from in the prepare above
        # (None on the cold-start call before the first decode).
        used_subtask = self._cached_subtask_str

        # NOTE: order matches participate_loop (broadcast_inputs → decode →
        # next_sample_rng) so collective ops on multi-host hit in the same
        # sequence on every rank.
        batched, extras = self._broadcast_inputs(batched, extras)

        # Subtask-AR critics: autoregressively decode the current subtask off
        # this call's observation (cadence-gated) BEFORE the policy forward.
        # Every rank hits the decode collectives at the same all-ranks point
        # (see _maybe_decode_subtask).
        did_decode, decode_out = self._maybe_decode_subtask(batched, extras, on_rank0 = True)

        sample_rng, sample_rngs = self._next_sample_rng()

        # ---- DIAGNOSTIC: dump fingerprints of batched + rng + extras keys
        # right before _run_jit_inference. Lets us compare against the SAME
        # path constructed manually in test scripts.
        if _DEBUG and self._process_index == 0:
            import hashlib as _hashlib
            def _fp(name, v):
                if hasattr(v, "shape"):
                    arr = np.asarray(jax.device_get(v))
                    h = _hashlib.sha256(arr.tobytes()).hexdigest()[:16]
                    logger.info(f"[infer.debug] {name} shape={arr.shape} dtype={arr.dtype} sha256[:16]={h} sum={float(arr.astype(np.float64).sum()):.6e}")
                else:
                    logger.info(f"[infer.debug] {name} = {v!r}")
            for k in sorted(batched.keys()):
                v = batched[k]
                if isinstance(v, dict):
                    for dk in sorted(v.keys()):
                        _fp(f"batched['{k}']['{dk}']", v[dk])
                else:
                    _fp(f"batched['{k}']", v)
            _fp("sample_rng", jax.random.key_data(sample_rng))
            # Don't fingerprint `sample_rngs` here — it's DATA_AXIS-sharded across
            # all hosts, so `jax.device_get` would try to fetch a non-process-local
            # array and raise. Per-host divergence is already verified by the
            # `[rng]` log inside `_next_sample_rng` (each host prints its
            # addressable shards).
            logger.info(f"[infer.debug] extras keys = {sorted(extras.keys())}")

        start_time = time.monotonic()
        # BestOfN path returns local [B, n_per_worker, ah, ad] + [B, n_per_worker];
        # policy-only path returns [B, ah, ad] + None.
        actions_out, q_values = self._run_jit_inference(batched, extras, sample_rng, sample_rngs)
        actions_out = jax.block_until_ready(actions_out)
        q_values_np: np.ndarray | None = None
        acs_value: float | None = None
        if q_values is not None:
            q_values = jax.block_until_ready(q_values)
            # actions_pool: (total, ah, ad), q_pool: (total,) — B=1 dropped.
            # `total == num_samples` because the BestOfNWrapper
            # trims its candidate pool to `num_samples` before the critic runs;
            # the padding to `num_samples_padded` happens only at the per-chip
            # rng layer to align with DATA_AXIS sharding, and the extra
            # candidates are dropped in the JIT graph itself.
            actions_pool, q_pool = self._gather_local_candidates(actions_out, q_values)
            n = int(self._bestofn.num_samples)
            actions_pool = actions_pool[:n]
            q_pool = q_pool[:n]
            best_idx = int(np.argmax(q_pool))
            actions_out = actions_pool[best_idx]
            q_values_np = q_pool
            # Average cosine similarity across the candidate pool, computed on
            # the 14-D EEF action slice (the non-EEF padded dims carry other
            # robot DOFs and would skew the similarity).
            eef_pool = actions_pool[
                ..., self._action_dim_offset : self._action_dim_offset + self._eef_action_dim
            ]
            acs_value = _compute_acs(np.asarray(eef_pool))
            logger.info(f"BestOfN q_values (n={n}): {q_values_np.tolist()}  acs={acs_value:.4f}")
        else:
            # Policy-only path: drop the always-1 leading batch dim so the
            # downstream slice + output_transform see (ah, ad) like the
            # BestOfN path does after _gather_local_candidates.
            actions_out = np.asarray(jax.device_get(actions_out))[0]
        infer_ms = (time.monotonic() - start_time) * 1000.0

        # Build a host-side observation snapshot for the output transform's
        # state passthrough (we already have the values on rank 0 in `obs`).
        observation = _model.Observation.from_dict(batched)

        # Slice the model's padded action vector down to the 14-D bimanual EEF
        # block before Unnormalize sees it (norm stats live at the unpadded
        # dim; without this slice Unnormalize would assert shape mismatch).
        # actions_out is now (ah, ad) — gather already dropped the B=1 dim.
        start = self._action_dim_offset
        actions_np = np.asarray(actions_out[..., start : start + self._eef_action_dim])

        decoded = self._output_transform({
            "state": np.asarray(observation.state[0]),
            "actions": actions_np,
            # `state`/`next_state` are passed through some transforms (e.g.
            # AbsoluteActions only touches actions, but mirroring the sibling
            # avoids any "missing key" failure for transforms that index
            # both).
            "next_state": np.asarray(observation.state[0]),
            "next_actions": actions_np,
        })

        result: dict[str, Any] = {
            "actions": np.asarray(decoded["actions"], dtype = np.float32),
            "policy_timing": {"infer_ms": infer_ms},
        }
        if self._return_q_values and q_values_np is not None:
            result["q_values"] = q_values_np
            if acs_value is not None:
                result["acs"] = acs_value

        # Subtask the critic conditioned its value forward on this call: the
        # cached subtask used to build the value prompt in the (re-)prepare
        # above (None on the cold-start call before the first decode). Only
        # emitted for critics that predict their subtask (a predictor is
        # configured); other critics and the policy-only path are unchanged.
        if self._subtask_decoder is not None:
            result["critic_subtask"] = used_subtask

        if decode_out is not None:
            result.update(decode_out)
        return result

    # ---------------------------------------------------------------------
    # Multi-host coordination internals
    # ---------------------------------------------------------------------

    def _persist_inference_iter(self) -> None:
        """Write the sampling counter beside the checkpoint, from rank 0 only.

        Every host bumps the counter in lockstep, so one writer is enough and eight racing
        on the same object is not.
        """
        if not self._inference_iter_path or self._process_index != 0:
            return
        try:
            _epath.Path(self._inference_iter_path).write_text(str(self._inference_iter))
            logger.info("Persisted inference_iter=%d to %s", self._inference_iter, self._inference_iter_path)
        except Exception as e:
            logger.error("Failed to persist inference_iter to %s: %s", self._inference_iter_path, e)

    def _install_preemption_handler(self) -> None:
        """Save the sampling counter when the pod is reclaimed.

        Spot preemption arrives as SIGTERM ahead of the shutdown, the same signal orbax
        turns into `reached_preemption` on the training side. A server has no step loop to
        check a flag from, so the write happens in the handler itself and then the previous
        disposition runs, letting the process die as it otherwise would.
        """
        if not self._inference_iter_path:
            return
        previous = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            logger.warning("SIGTERM at inference_iter=%d; persisting before shutdown", self._inference_iter)
            self._persist_inference_iter()
            if callable(previous) and previous not in (signal.SIG_IGN, signal.SIG_DFL):
                previous(signum, frame)
            else:
                raise SystemExit(0)

        try:
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            # Only the main thread may install handlers; a server loaded from a worker
            # thread simply keeps the default disposition.
            logger.warning("Could not install SIGTERM handler (not main thread); inference_iter will not persist")

    def _maybe_decode_subtask(
        self, batched: dict[str, Any], extras: dict[str, Any], *, on_rank0: bool,
    ) -> tuple[bool, dict[str, Any] | None]:
        """All-ranks pre-inference subtask decode for predict_subtask_ar critics.

        Gated by `self._subtask_iter % self._subtask_decode_every`, advanced
        exactly once per inference on EVERY rank (rank 0 via `infer`, workers via
        `participate_loop`/`_prewarm_jit`), so the decode JIT collectives fire on
        the identical step on every host. On rank 0 the decoded subtask is cached
        (consumed by the next call's value prompt) and the decode dict is returned; on
        workers the same JIT chain runs in lockstep. Returns `(did_run, out)` —
        `did_run` is identical on every rank (shared counter); `out` is non-None only
        on rank 0 decode calls. No-op `(False, None)` when no decoder is
        configured (non-subtask_ar critic / policy-only path).
        """
        if self._subtask_decoder is None:
            return False, None
        run = (self._subtask_iter % self._subtask_decode_every == 0)
        self._subtask_iter += 1
        if not run:
            return False, None
        critic_obs = _model.Observation.from_dict(batched)
        critic_obs = dataclasses.replace(
            critic_obs,
            tokenized_prompt = extras["decode_critic_tokens"],
            tokenized_prompt_mask = extras["decode_critic_token_mask"],
        )
        if on_rank0:
            out = self._subtask_decoder.predict(critic_obs)
            self._cached_subtask_str = out["predicted_subtask"]
            return True, out
        self._subtask_decoder.run_lockstep(critic_obs)
        return True, None

    def _prepare_inputs_rank0(
        self, obs: dict, *, noise: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rank-0 only: run the input transform pipeline + build the broadcast
        package (`batched` for the Observation, `extras` for everything else
        the JIT'd inference closure consumes).
        """
        prompt_str = obs.get("prompt")
        # Per-call task_description override. Policy + critic each gated on
        # their own prompt_mode; the override is popped before _input_transform
        # so the transform pipeline doesn't see an unexpected field.
        per_call_task_desc = (
            obs.get("task_description") if isinstance(obs, dict) else None
        )
        if isinstance(obs, dict) and "task_description" in obs:
            obs = {k: v for k, v in obs.items() if k != "task_description"}
        policy_task_desc = None
        if self._policy_prompt_mode == "task_description":
            policy_task_desc = per_call_task_desc or self._policy_task_description
        critic_task_desc = None
        if self._critic_prompt_mode in ("task_description", "task_description_predict_current_subtask"):
            critic_task_desc = per_call_task_desc or self._critic_task_description
        # A subtask-decoding critic REQUIRES a task description: without one the
        # extras package drops the decode_critic_* leaves, so rank 0 broadcasts
        # a structurally different package than the workers' dummies and the
        # multi-host launch group desyncs (workers hang / one rank halts).
        # Fail loudly instead.
        if (
            self._subtask_decoder is not None
            and self._subtask_decoder.needs_task_description
            and critic_task_desc is None
        ):
            raise ValueError(
                "Subtask-decoding critic needs a task description on every call: "
                "send obs['task_description'] from the client or serve with "
                "--task-description."
            )
        if policy_task_desc is not None:
            obs = {**obs, "prompt": policy_task_desc}
        transformed = self._input_transform(obs)

        batched: dict[str, Any] = {}
        for k, v in transformed.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                batched[k] = {dk: jnp.asarray(dv)[None, ...] for dk, dv in v.items()}
            else:
                batched[k] = jnp.asarray(v)[None, ...]

        # Only fall back to all-True masks when the input transform did not
        # already supply them: an input transform may mask a camera that does not
        # exist on the platform, and that must be preserved or the model attends
        # to a zero-image.
        action_horizon = self._model.action_horizon
        if "action_mask" not in batched:
            # Horizons default to all-True; the adapt-time mask logic in
            # BestOfNWrapper.sample_actions handles subsample / pad.
            if self._policy_subsample and action_horizon == 60:
                # Stride-2 policy emits 30 real actions zero-padded to 60.
                fps_mask = jnp.concatenate(
                    [jnp.ones(30, dtype = jnp.bool_), jnp.zeros(30, dtype = jnp.bool_)]
                )
                batched["action_mask"] = fps_mask[None, :]
            else:
                batched["action_mask"] = jnp.ones(action_horizon, dtype = jnp.bool_)[None, :]
        if "image_mask" not in batched:
            batched["image_mask"] = {k: jnp.array([True]) for k in batched.get("image", {})}

        extras: dict[str, Any] = {}
        if self._bestofn is not None and self._critic_tokenizer is None:
            # Tokenizer-free critic (ResNet): keep the broadcast package shape-stable with
            # zero token placeholders and detect the null prompt on the raw string.
            null_prompt = isinstance(prompt_str, str) and prompt_str.strip().lower() == "null"
            extras["critic_tokens"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.int32)
            extras["critic_token_mask"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.bool_)
            extras["subtask_start_index"] = jnp.zeros((1,), dtype = jnp.int32)
            extras["subtask_end_index"] = jnp.zeros((1,), dtype = jnp.int32)
            extras["critic_is_null_prompt"] = jnp.asarray(null_prompt, dtype = jnp.bool_)
            if self._subtask_decoder is not None:
                # The same decode leaves the AR path carries, so `_maybe_decode_subtask`
                # builds the critic observation identically; a categorical head reads the
                # images only and ignores the (zero) tokens.
                extras["decode_critic_tokens"] = extras["critic_tokens"]
                extras["decode_critic_token_mask"] = extras["critic_token_mask"]
        elif self._bestofn is not None:
            # subtask_start/end_index default to 0 (ignored unless predict_subtask_ar).
            s_start, s_end = 0, 0
            if critic_task_desc is not None and self._critic_prompt_mode == "task_description":
                # Trained with plain TokenizePrompt on the task string; the subtask
                # tokenizer's prefix separator would not reproduce those token ids.
                critic_tokens, critic_token_mask = self._critic_tokenizer.tokenize(critic_task_desc, None)
            elif critic_task_desc is not None:
                if self._subtask_decoder is not None:
                    # Task-only prefix the AR decoder reads (its NTP training input).
                    decode_tokens, decode_token_mask, _, _ = _transforms._tokenize_subtask_prompt(
                        self._critic_tokenizer, critic_task_desc, "", append_newline = False,
                    )
                    extras["decode_critic_tokens"] = jnp.asarray(decode_tokens)[None, ...]
                    extras["decode_critic_token_mask"] = jnp.asarray(decode_token_mask)[None, ...]
                if self._critic_predict_subtask_ar and self._cached_subtask_str:
                    # Value prompt conditions Q on the cached (predicted) subtask
                    # + trailing "\n" (append_newline=True), exactly like training.
                    critic_tokens, critic_token_mask, s_start, s_end = _transforms._tokenize_subtask_prompt(
                        self._critic_tokenizer, critic_task_desc, self._cached_subtask_str, append_newline = True,
                    )
                else:
                    # Cold start / non-subtask_ar: task-only prompt (no suffix).
                    # The returned indices point one past the prefix → an empty
                    # suffix; with predict_subtask_ar=True this is value-neutral.
                    critic_tokens, critic_token_mask, s_start, s_end = _transforms._tokenize_subtask_prompt(
                        self._critic_tokenizer, critic_task_desc, "", append_newline = False,
                    )
            else:
                critic_tokens, critic_token_mask = self._critic_tokenizer.tokenize(prompt_str, None)
            extras["critic_tokens"] = jnp.asarray(critic_tokens)[None, ...]
            extras["critic_token_mask"] = jnp.asarray(critic_token_mask)[None, ...]
            extras["subtask_start_index"] = jnp.asarray([s_start], dtype = jnp.int32)
            extras["subtask_end_index"] = jnp.asarray([s_end], dtype = jnp.int32)
            # Host-side null-prompt detection: keeps the raw token array off
            # the JIT'd BestOfNWrapper (nnx.split rejects bare np.ndarray leaves).
            critic_tokens_np = np.asarray(critic_tokens)
            null_ref = self._critic_null_prompt_tokens
            is_null = (
                critic_tokens_np.shape == null_ref.shape
                and bool(np.array_equal(critic_tokens_np, null_ref))
            )
            extras["critic_is_null_prompt"] = jnp.asarray(is_null, dtype = jnp.bool_)
        if self._bestofn is not None:
            extras["critic_subtask_id"] = jnp.asarray([self._cached_subtask_id()], dtype = jnp.int32)
        if noise is not None:
            noise_arr = jnp.asarray(noise)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            extras["noise"] = noise_arr
        return batched, extras

    def _cached_subtask_id(self) -> int:
        """Categorical id of the cached predicted subtask for the critic observation.

        -1 is the network's "resolve it from the images" sentinel: before the first
        prediction, and for every critic without a categorical subtask head, where the
        leaf is carried only to keep the broadcast package shape-stable and is never read.
        """
        if not self._critic_uses_subtask_id or self._cached_subtask_str is None:
            return -1
        return self._subtask_decoder.vocab_index(self._cached_subtask_str)

    def _make_dummy_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Workers 1..N-1: construct broadcast-package dummies of the exact
        shapes/dtypes that rank 0 produces, so `broadcast_one_to_all` can
        merge them. Values are placeholders; rank 0's values overwrite them.
        """
        action_horizon = self._model.action_horizon
        batched: dict[str, Any] = {
            "state": jnp.zeros((1, self._eef_action_dim), dtype = jnp.float32),
            "image": {k: jnp.zeros((1, self._image_size, self._image_size, 3), dtype = jnp.uint8) for k in self._image_keys},
            "image_mask": {k: jnp.array([True], dtype = jnp.bool_) for k in self._image_keys},
            "tokenized_prompt": jnp.zeros((1, self._max_token_len), dtype = jnp.int32),
            "tokenized_prompt_mask": jnp.zeros((1, self._max_token_len), dtype = jnp.bool_),
            "action_mask": jnp.ones((1, action_horizon), dtype = jnp.bool_),
        }
        extras: dict[str, Any] = {}
        if self._bestofn is not None:
            extras["critic_tokens"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.int32)
            extras["critic_token_mask"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.bool_)
            extras["critic_is_null_prompt"] = jnp.asarray(False, dtype = jnp.bool_)
            extras["subtask_start_index"] = jnp.zeros((1,), dtype = jnp.int32)
            extras["subtask_end_index"] = jnp.zeros((1,), dtype = jnp.int32)
            extras["critic_subtask_id"] = jnp.full((1,), -1, dtype = jnp.int32)
            if self._subtask_decoder is not None:
                extras["decode_critic_tokens"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.int32)
                extras["decode_critic_token_mask"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.bool_)
        return batched, extras

    def _broadcast_inputs(
        self, batched: dict[str, Any], extras: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Broadcast the inference inputs from JAX rank 0 to all other hosts.
        No-op on a single-host setup. Source defaults to rank 0 (matches
        where the websocket binds in serve_policy.py).

        `broadcast_one_to_all` up-promotes narrow dtypes (bool → int32,
        uint8 → uint32); we cast back so Pi0.embed_prefix's jaxtyping
        annotation accepts the prefix mask.
        """
        if not self._is_multi_host:
            return batched, extras
        pkg = {"batched": batched, "extras": extras}
        pkg = _multihost.broadcast_one_to_all(pkg)
        batched = _restore_inference_dtypes(pkg["batched"])
        extras = _restore_inference_dtypes(pkg["extras"])
        return batched, extras

    def _next_sample_rng(self):
        """Build (rng_replicated, sample_rngs_sharded) for one inference call.

        `rng_replicated` is a single key, identical on every host, used by
        downstream ops that should see the same rng on every host.

        `sample_rngs_sharded` is an `(num_samples_padded, ...)` key array
        sharded along `PartitionSpec(DATA_AXIS, ...)`. Every host computes the
        full split deterministically from `rng_replicated` (so the per-chip
        slices agree across hosts), then `device_put` distributes one unique
        rng per chip. The vmap'd policy.sample_actions inside the JIT then
        sees one different rng per chip → unique noise per chip → diverse
        candidates after `process_allgather`.

        Why two arrays: the caller's previous design folded `process_index`
        into a single replicated-typed rng, which under cross-host FSDP
        all-gather concatenated the disagreeing per-host slices into a
        "Frankenstein" tensor (a bug seen on multi-host serving). Declaring
        the divergent rng as DATA_AXIS-sharded lets JAX route each chip's
        unique slice through SPMD without that collapse.
        """
        self._inference_iter += 1
        fold_value = self._num_processes * self._inference_iter
        base_rng = jax.random.fold_in(self._rng, fold_value)

        total_rngs = jax.random.split(base_rng, self._num_samples_padded)
        # In sample_parallel mode shard rngs along BATCH_AXIS only (size =
        # num_samples) so each batch position owns one unique rng, replicated
        # across the per-sample FSDP group. Otherwise use the original
        # DATA_AXIS (combined batch+fsdp = device_count) sharding.
        leading_spec = (
            self._sharding_mod.BATCH_AXIS if self._sample_parallel
            else self._sharding_mod.DATA_AXIS
        )
        # Determine PartitionSpec based on key array layout: typed PRNG keys
        # have ndim=1, untyped uint32 keys have ndim=2 (trailing key-data axis).
        if total_rngs.ndim == 1:
            spec = jax.sharding.PartitionSpec(leading_spec)
        else:
            # ndim >= 2: shard leading axis only, replicate the trailing axes.
            spec = jax.sharding.PartitionSpec(
                leading_spec, *([None] * (total_rngs.ndim - 1))
            )
        sharding_spec = jax.sharding.NamedSharding(self._mesh, spec)
        # `jax.device_put` won't accept a multi-host sharding because each host
        # can only address its 4 local chips, not the global 32. `make_array_from_callback`
        # invokes `cb(index)` once per *addressable* shard on this host, asking
        # for the slice this shard owns. Since `total_rngs` is computed
        # deterministically from `base_rng` on every host, all hosts agree on
        # `total_rngs[index]` — so the resulting global array assembles the
        # right per-chip slices without any cross-host transfer.
        def cb(index):
            return total_rngs[index]
        sample_rngs = jax.make_array_from_callback(total_rngs.shape, sharding_spec, cb)

        # Diagnostic: dump THIS host's addressable slice of the sharded rng so we
        # can grep across worker logs and verify per-host divergence (the goal of
        # this fix). For 8 hosts × 4 chips, every host should see 4 unique key
        # values, and across hosts the 8×4 = 32 keys should all differ. If two
        # hosts log the same first-key, the FSDP all-gather is silently
        # collapsing per-host divergence again — same failure mode as the
        # original Frankenstein bug, just at the rng input.
        try:
            local_shards = sample_rngs.addressable_shards
            local_keys_repr = []
            for s in local_shards:
                arr = np.asarray(jax.random.key_data(s.data))
                local_keys_repr.append(arr.flatten().tolist())
            logger.info(
                "[rng] iter=%d rank=%d/%d local_n_shards=%d local_keys=%s",
                self._inference_iter, self._process_index, self._num_processes,
                len(local_shards), local_keys_repr,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"[rng] addressable_shards introspection failed: {exc!r}")
        return base_rng, sample_rngs

    def _gather_local_candidates(
        self, local_actions, local_q_values,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop the always-1 batch dim, gather across workers, return (actions[N, ah, ad], q_values[N]).

        With the new sharded-rng path, the JIT outputs are multi-host sharded
        along the candidate axis (axis 1 of (B=1, N, ah, ad)) — `jax.device_get`
        on such an array raises "Fetching value for `jax.Array` that spans
        non-addressable devices". Instead, walk the array's `addressable_shards`
        (each shard is a single-chip slice of size 1 along the candidate axis),
        materialize them with `np.asarray(s.data)`, and concat along axis 1 to
        get a host-local numpy view shaped (B=1, n_per_host, ah, ad). Then
        `process_allgather(tiled=True)` collects across hosts: the leading axis
        (= n_per_host after dropping B=1) is multiplied by num_processes, giving
        (num_samples_padded, ah, ad) on every host.
        """
        if self._sample_parallel:
            # JIT outputs are with_sharding_constraint'd to fully replicated, so
            # every chip on every host holds the complete (B, N, ...) array.
            # Read the first addressable shard — all shards are identical replicas,
            # and no cross-host gather is needed.
            actions = np.asarray(local_actions.addressable_shards[0].data)
            q_values = np.asarray(local_q_values.addressable_shards[0].data, dtype = np.float32)
            return actions[0], q_values[0]
        a_shards = [np.asarray(s.data) for s in local_actions.addressable_shards]
        q_shards = [np.asarray(s.data) for s in local_q_values.addressable_shards]
        # Each shard is one chip's slice along the sharded axis (axis 1). Concat axis=1
        # → host-local view (B=1, n_per_host, ...). For our setup n_per_host = 4.
        local_actions_np = np.concatenate(a_shards, axis = 1) if len(a_shards) > 1 else a_shards[0]
        local_q_np = np.concatenate(q_shards, axis = 1) if len(q_shards) > 1 else q_shards[0]
        # Drop B=1 leading dim.
        local_actions_np = local_actions_np[0]  # (n_per_host, ah, ad)
        local_q_np = local_q_np[0].astype(np.float32)  # (n_per_host,)
        if not self._is_multi_host:
            return local_actions_np, local_q_np
        # tiled=True concatenates host-local arrays along their leading axis,
        # giving (n_per_host * num_processes, ...) = (num_samples_padded, ...)
        # in process-index order. Order matches the rng layout from
        # _next_sample_rng (process p owns rng[p*n_per_host : (p+1)*n_per_host]).
        actions = np.asarray(_multihost.process_allgather(local_actions_np, tiled = True))
        q_values = np.asarray(_multihost.process_allgather(local_q_np, tiled = True), dtype = np.float32)
        return actions, q_values

    def _run_jit_inference(
        self, batched: dict[str, Any], extras: dict[str, Any], rng, sample_rngs,
    ) -> tuple[Any, Any | None]:
        """Build Observation/Transition and call the JIT'd sampling closure.
        Returns (actions_out, q_values_or_None). Workers 1..N-1 discard the
        outputs but still have to make the call so the collective ops complete.

        `rng` is a single replicated key; `sample_rngs` is the DATA_AXIS-sharded
        per-sample rng array (only consumed by the BestOfN-with-critic path).
        """
        observation = _model.Observation.from_dict(batched)
        if self._bestofn is not None:
            transition = _base_vf.Transition(observation = observation)
            actions_out, q_values = self._bestofn_sample(
                self._bestofn, self._critic_model, rng, sample_rngs, transition,
                extras["critic_tokens"], extras["critic_token_mask"],
                extras["critic_is_null_prompt"],
                extras["subtask_start_index"], extras["subtask_end_index"],
                extras["critic_subtask_id"],
            )
            return actions_out, q_values
        transition = _model.wrap_observation_as_transition(observation)
        sample_kwargs = dict(self._sample_kwargs)
        if "noise" in extras:
            sample_kwargs["noise"] = extras["noise"]
        if _DEBUG:
            _log_model_inputs("policy", observation)
        actions_out = self._sample_actions_jit(rng, transition, **sample_kwargs)
        # Note: the raw_actions debug print runs INSIDE the JIT graph (inside
        # `_sample_with_debug` wrapped by `_sample_actions_jit`), not here.
        return actions_out, None

    def participate_loop(self) -> None:
        """Workers 1..N-1: block on `broadcast_one_to_all` for each rank-0
        inference, run the JIT'd closure in lockstep, discard the outputs.

        Runs forever; killed when the tmux session is killed (or worker is
        preempted). No-op on single-host.
        """
        if not self._is_multi_host:
            return
        rank = self._process_index
        logger.info(
            f"Worker (JAX rank {rank}): participate_loop started; "
            f"waiting for rank-0 broadcasts..."
        )
        local_iter = 0
        while True:
            try:
                batched, extras = self._make_dummy_inputs()
                batched, extras = self._broadcast_inputs(batched, extras)
                # Drive the subtask decode in lockstep with rank 0 (no-op unless
                # this is a decode step for a predict_subtask_ar critic). Runs
                # BEFORE inference, mirroring rank 0's `infer` order.
                self._maybe_decode_subtask(batched, extras, on_rank0 = False)
                sample_rng, sample_rngs = self._next_sample_rng()
                actions_out, q_values = self._run_jit_inference(
                    batched, extras, sample_rng, sample_rngs,
                )
                jax.block_until_ready(actions_out)
                if q_values is not None:
                    q_values = jax.block_until_ready(q_values)
                    # Participate in rank 0's process_allgather; discard result.
                    self._gather_local_candidates(actions_out, q_values)
                local_iter += 1
                if local_iter % 100 == 0:
                    logger.info(f"Worker (JAX rank {rank}): participated in {local_iter} inferences")
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"Worker (JAX rank {rank}) participate_loop iteration error: {exc!r}")
                # Don't break; keep participating so a transient failure
                # doesn't permanently desync the collective.

    def _prewarm_jit(self) -> None:
        """Trigger one JIT compile pass before serve_forever opens the port.

        On multi-host, every rank participates: rank 0 runs `infer` with a
        dummy obs (which broadcasts inputs to all hosts), and ranks 1..N-1
        block on the matching `broadcast_one_to_all` from `participate_loop`
        — running exactly one iteration here so the JIT graph compiles and
        peer collectives are warm before the websocket starts. After prewarm
        ranks 1..N-1 enter the long-lived `participate_loop` from
        scripts/serve_policy.py main().

        Without prewarm the first websocket request takes 30-60s while JAX
        compiles, which blows past the client's default read timeout.
        """
        try:
            if self._is_multi_host and self._process_index != 0:
                logger.info(
                    f"Worker (JAX rank {self._process_index}): prewarm = one participate iteration"
                )
                batched, extras = self._make_dummy_inputs()
                batched, extras = self._broadcast_inputs(batched, extras)
                # Compile the decode JIT in lockstep with rank 0's prewarm infer
                # (decode runs BEFORE inference, mirroring `infer`'s order).
                self._maybe_decode_subtask(batched, extras, on_rank0 = False)
                sample_rng, sample_rngs = self._next_sample_rng()
                actions_out, q_values = self._run_jit_inference(
                    batched, extras, sample_rng, sample_rngs,
                )
                jax.block_until_ready(actions_out)
                if q_values is not None:
                    self._gather_local_candidates(actions_out, jax.block_until_ready(q_values))
                # Reset cadence + cache so the first real inference re-decodes
                # (prewarm decoded from zero-image dummies → a garbage subtask).
                self._subtask_iter = 0
                self._cached_subtask_str = None
                logger.info(f"Worker (JAX rank {self._process_index}): prewarm done")
                return

            logger.info("Prewarming BestOfNPolicy JIT (this may take 30-60s)...")
            zero_image = np.zeros((self._image_size, self._image_size, 3), dtype = np.uint8)
            # The input transform expects an `image` dict + flat `state` + `prompt`,
            # matching what the client sends. State norm is quantile so a zeros
            # vector is fine.
            dummy_obs = {
                "image": dict.fromkeys(self._image_keys, zero_image),
                "state": np.zeros(self._state_dim, dtype = np.float32),
                "prompt": "prewarm",
                # Keeps the prewarm broadcast package structurally identical
                # to real client calls: without a task description a
                # subtask-decoding critic's extras drop the decode_critic_*
                # leaves, so rank 0 compiles a different broadcast program
                # than the workers' dummies -> "unexpected peer in the
                # launch group" desync on multi-host.
                "task_description": "prewarm",
            }
            _ = self.infer(dummy_obs)
            logger.info("Prewarm complete.")
            # Reset cadence + cache so the first real inference re-decodes
            # (prewarm decoded from zero-image dummies → a garbage subtask).
            self._subtask_iter = 0
            self._cached_subtask_str = None
        except Exception as exc:  # pylint: disable=broad-except
            # Prewarm is best-effort; if the dummy obs doesn't fit the loaded
            # config (e.g. an input transform expecting other keys), fall back to lazy compile
            # at first real request.
            logger.warning(
                f"Prewarm failed ({exc!r}); will compile on first request instead."
            )


def create_bestofn_policy(
    *,
    policy_config_name: str,
    policy_checkpoint_dir: str,
    policy_step: int | None = None,
    policy_fine_tune_config: str | None = None,
    policy_task_description: str | None = None,
    critic_config_name: str | None = None,
    critic_checkpoint_dir: str | None = None,
    critic_step: int | None = None,
    critic_fine_tune_config: str | None = None,
    num_samples: int = 8,
    default_prompt: str | None = None,
    prewarm: bool = True,
    sample_parallel: bool = False,
    fsdp_devices: int | None = None,
    subtask_decode_every: int = 20,
    num_steps: int | None = None,
    inference_iter_path: str | None = None,
    start_inference_iter: int | None = None,
) -> BestOfNPolicy:
    """Convenience factory; matches the kwargs the serve_policy CLI exposes."""
    return BestOfNPolicy(
        policy_config_name = policy_config_name,
        policy_checkpoint_dir = policy_checkpoint_dir,
        policy_step = policy_step,
        policy_fine_tune_config = policy_fine_tune_config,
        policy_task_description = policy_task_description,
        critic_config_name = critic_config_name,
        critic_checkpoint_dir = critic_checkpoint_dir,
        critic_step = critic_step,
        critic_fine_tune_config = critic_fine_tune_config,
        num_samples = num_samples,
        default_prompt = default_prompt,
        prewarm = prewarm,
        sample_parallel = sample_parallel,
        fsdp_devices = fsdp_devices,
        subtask_decode_every = subtask_decode_every,
        num_steps = num_steps,
        inference_iter_path = inference_iter_path,
        start_inference_iter = start_inference_iter,
    )
