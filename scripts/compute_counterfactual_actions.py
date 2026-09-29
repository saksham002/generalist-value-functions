"""Distributed GPU computation of counterfactual action stores for RLDS datasets.

Pre-computes N action samples from a BC-trained policy for every state in the dataset,
using the same prompt construction as policy training. Follows the same shard-mirroring
TFDS pattern as the latent store.

Three tyro subcommands:
  launch  - Submit N sbatch worker jobs from the cluster head node.
  worker  - Process a contiguous chunk of episodes on a single GPU.
  merge   - Combine per-worker outputs into a single counterfactual action store.

Usage:
    # 1. Launch N workers
    uv run scripts/compute_counterfactual_actions.py launch \
        --config-name robocoin_bimanual_pi05_rlds \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16

    # 2. Worker mode (invoked by sbatch, not called directly)
    uv run scripts/compute_counterfactual_actions.py worker \
        --config-name robocoin_bimanual_pi05_rlds \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16 --worker-id 3

    # 3. Merge (after all workers finish)
    uv run scripts/compute_counterfactual_actions.py merge \
        --config-name robocoin_bimanual_pi05_rlds \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16
"""

from collections import defaultdict
from collections import deque
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import shlex
import subprocess
import time
from typing import Annotated, Any

from etils import epath
import numpy as np
from rlds_build_utils import get_rlds_episode_index
from rlds_build_utils import get_total_episodes
from rlds_build_utils import parse_partition_split
import tqdm
import tyro

import openpi.training.yam_eef as yam_eef

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_config_with_fine_tune(config_name: str, fine_tune: str | None, rlds_data_dir: str | None = None):
    """Resolve openpi config and optionally apply a FineTuneConfig's overrides.

    Mirrors the fine-tune handling in `scripts/train_value_function.py` (with
    `pretrained_step = None`, since we are not training): only data/model/interval
    overrides are applied, no schedule offsetting.

    ``rlds_data_dir`` optionally overrides the config's data dir (e.g. to point at a
    local mirror / a mini dataset instead of the GCS default baked into the config).

    Returns (config, data_config, dataset_cfg, model_config).
    """
    import openpi.training.config as _config

    config = _config.get_config(config_name)
    if fine_tune is not None:
        ft_config = _config.get_fine_tune_config(fine_tune)
        config = ft_config.apply_overrides(config)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data = dataclasses.replace(config.data, rlds_data_dir = rlds_data_dir))

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is None:
        raise ValueError("Config must have rlds_data_dir set.")

    datasets = data_config.datasets
    if not datasets:
        raise ValueError("Config must have datasets configured.")
    if len(datasets) > 1:
        logger.warning("Multiple datasets configured, only processing first one.")

    return config, data_config, datasets[0], config.model


# =============================================================================
# Argument dataclasses
# =============================================================================


@dataclasses.dataclass
class CommonArgs:
    """Arguments shared across all subcommands."""

    config_name: str = "robocoin_bimanual_pi05_rlds"
    """Config name to resolve RLDS data source and policy."""

    fine_tune: str | None = None
    """Optional FineTuneConfig name to apply on top of the base config (data/model overrides)."""

    checkpoint_dir: str = ""
    """Path to the trained policy checkpoint (required)."""

    output_dir: str = ""
    """Output counterfactual action store directory (local or GCS path)."""

    split: str = "train"
    """Split to process (e.g., 'train', 'val')."""

    num_workers: int = 8
    """Number of worker jobs to distribute across."""

    num_samples: int = 32
    """Number of action samples to generate per state."""

    stride: int = 1
    """Stride for computing counterfactual actions. stride=5 means compute at steps 0, 5, 10, etc."""

    debug_metrics: bool = False
    """If True, compute extra normalized-space diagnostics such as sampling loss."""

    convert_joint_actions_to_eef: bool = False
    """Run YAM forward kinematics on the policy's absolute joint actions so the store
    holds the 14D EEF layout (pos3+euler3+gripper per arm) the critic was pretrained on.
    Only valid for joint-space YAM policies; the conversion is not invertible here."""

    yam_fk_dir: str = yam_eef.DEFAULT_YAM_FK_DIR
    """Directory holding yam_fk.py + vendor/yam_vendor_kin.xml (see --convert-joint-actions-to-eef)."""

    reverse: bool = False
    """If True, reverse shard processing order and skip metadata writes. Intended for running
    a second set of workers that converge from the opposite end."""

    rlds_data_dir: str | None = None
    """Optional override of the config's data dir (e.g. a local mini dataset). Forwarded to workers."""

    sampling_num_steps: int | None = None
    """Flow-matching integration steps for action sampling (model.sample_actions's num_steps).
    None leaves the model default (10)."""


@dataclasses.dataclass
class LaunchArgs(CommonArgs):
    """Launch N sbatch worker jobs from the cluster head node."""

    partition: str = "preempt"
    """SLURM partition."""

    qos: str | None = None
    """Optional SLURM QoS."""

    partition_split: str | None = None
    """Split workers across partitions. Format: 'partition1:count1,partition2:count2,...'."""

    time_limit: str = "24:00:00"
    """SLURM time limit."""

    mem: str = "64GB"
    """SLURM memory limit."""

    cpus_per_task: int | None = None
    """SLURM --cpus-per-task for each worker (and the merge job). None = scheduler default."""

    gres: str = "gpu:L40S:1"
    """SLURM GPU resource spec."""

    extra_args: str | None = None
    """Extra sbatch arguments appended verbatim to the worker and merge submissions
    (e.g. '--exclude <node>')."""

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    samples_per_batch: int = 32
    """Maximum total sampled actions per GPU forward pass across all observations."""

    auto_merge: bool = False
    """Submit a dependent merge job after all workers."""

    dry_run: bool = False
    """Print sbatch commands without submitting."""

    uv_bin: str = "uv"
    """Full path to the uv binary on the worker nodes."""

    total_episodes: int | None = None
    """Total source episodes. If provided, skips querying the dataset."""

    ssh_host: str | None = None
    """SSH host to route sbatch through (a cluster login node)."""

    log_dir_base: str = "~/slurm_logs"
    """Base directory for SLURM log files."""


@dataclasses.dataclass
class WorkerArgs(CommonArgs):
    """Process a contiguous chunk of episodes on a single GPU."""

    worker_id: int = 0
    """This worker's index (0-based)."""

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    samples_per_batch: int = 32
    """Maximum total sampled actions per GPU forward pass across all observations."""

    profile_log_dir: str | None = None
    """If set, write a JAX trace for the first few sampling batches to this directory."""

    profile_num_batches: int = 1
    """Number of sampling batches to include in the trace when profile_log_dir is set."""

    profile_skip_batches: int = 0
    """Number of initial sampling batches to skip before starting JAX tracing."""

    only_shard: int | None = None
    """If set, process only this shard index and skip all others."""


@dataclasses.dataclass
class MergeArgs(CommonArgs):
    """Combine per-worker outputs into a single counterfactual action store."""

    source_dir: str | None = None
    """Directory containing worker outputs. Defaults to output_dir if not specified."""

    overwrite: bool = False
    """Overwrite existing merged output."""

    cleanup_workers: bool = False
    """Delete each worker's directory as soon as its shards have been copied
    into the merged output — reclaims disk incrementally during the merge."""


# =============================================================================
# Worker implementation
# =============================================================================


def run_worker(args: WorkerArgs) -> None:
    """Run a single worker that processes its assigned RLDS shards.

    For each episode, for each step:
    - Construct observation dict and apply input transforms
    - Generate num_samples action samples via the policy model
    - Apply output transforms to unnormalize actions
    - Store as [num_steps, num_samples, action_horizon, action_dim]
    """
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    import tensorflow_datasets as tfds

    import openpi.models.model as _model
    from openpi.rlds_utils.utils import RLDS_TO_STANDARD_CAMERA_MAP
    from openpi.rlds_utils.utils import extract_embodiment
    import openpi.training.counterfactual_action_store as ca_store
    from openpi.training.time_utils import Timer

    tf.config.set_visible_devices([], "GPU")

    worker_id = args.worker_id
    num_workers = args.num_workers

    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker_id={worker_id} out of range [0, {num_workers}).")

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")

    # Isolate this worker's download cache. maybe_download's filelock does not
    # serialize across NFS-mounted nodes, so workers sharing ~/.cache/openpi race
    # on the force_download=True norm-stats fetch. A per-job cache dir removes the
    # shared path entirely. Respect an externally-set OPENPI_DATA_HOME first so a
    # pre-staged local asset cache (e.g. the PaliGemma tokenizer) can be used to
    # avoid GCS fetches on DNS-flaky nodes.
    if "OPENPI_DATA_HOME" in os.environ:
        # Suffix the external cache root with the worker id so workers never share a
        # cache path (pre-stage per-worker copies of local assets, e.g. the PaliGemma
        # tokenizer, under <root>/<worker_id>/big_vision/).
        os.environ["OPENPI_DATA_HOME"] = os.path.join(os.environ["OPENPI_DATA_HOME"], str(worker_id))
    else:
        cache_tag = os.environ.get("SLURM_JOB_ID") or f"worker_{worker_id}"
        os.environ["OPENPI_DATA_HOME"] = os.path.expanduser(f"~/.cache/openpi_ca/{cache_tag}")

    config, data_config, dataset_cfg, _ = _resolve_config_with_fine_tune(
        args.config_name, args.fine_tune, args.rlds_data_dir,
    )

    # This worker has no subsample support: it samples on the raw (un-subsampled)
    # steps, so a subsample=True config would misalign the cache with training.
    assert not data_config.rlds_kwargs.get("subsample", False), (
        "compute_counterfactual_actions.py does not support subsample=True configs."
    )

    # Create builder once and reuse for metadata queries and dataset loading
    source_builder = tfds.builder(dataset_cfg.name, data_dir=data_config.rlds_data_dir, version=dataset_cfg.version)
    logger.info(
        f"Source dataset resolved to {source_builder.data_dir} "
        f"(rlds_data_dir={data_config.rlds_data_dir}, name={dataset_cfg.name}, version={dataset_cfg.version})"
    )
    split_info = source_builder.info.splits[args.split]
    total_episodes = split_info.num_examples
    shard_lengths = split_info.shard_lengths
    shard_info = []
    offset = 0
    for length in shard_lengths:
        shard_info.append((offset, length))
        offset += length
    num_shards = len(shard_info)

    # Distribute shards across workers (round-robin)
    my_shards = [i for i in range(num_shards) if i % num_workers == worker_id]
    if args.only_shard is not None:
        if args.only_shard < 0 or args.only_shard >= num_shards:
            raise ValueError(f"--only-shard={args.only_shard} out of range [0, {num_shards}).")
        my_shards = [args.only_shard]
    if args.reverse:
        my_shards = my_shards[::-1]
    my_episode_count = sum(shard_info[i][1] for i in my_shards)

    logger.info(
        f"Worker {worker_id}/{num_workers}: assigned {len(my_shards)} shards "
        f"({my_episode_count} episodes total)"
        + (" [REVERSE order]" if args.reverse else "")
    )
    if args.reverse:
        logger.warning("--reverse mode: metadata files, manifest, and done marker will NOT be written.")

    output_dir = epath.Path(args.output_dir)
    worker_dir = output_dir / "_workers" / f"worker_{worker_id}"
    done_marker = worker_dir / "_DONE"

    if done_marker.exists():
        logger.info(f"Worker {worker_id}: already completed, skipping.")
        return

    # Load policy. For value function training configs, the checkpoint stores params
    # as {"critic": ..., "policy": ...}. We need to extract the "policy" subtree and
    # load it into the policy model config (config.policy), not the value function
    # model (config.model). We replicate the relevant parts of create_trained_policy
    # here to handle this param extraction.
    policy_model_config = config.policy if config.policy is not None else config.model

    logger.info(f"Loading policy from {args.checkpoint_dir}")

    from openpi.training import checkpoints as _checkpoints

    checkpoint_dir_path = epath.Path(args.checkpoint_dir)
    all_params = _model.restore_params(checkpoint_dir_path / "params", dtype=jnp.bfloat16)

    # Extract BC policy params from the checkpoint. For actor-critic training configs,
    # checkpoint params are structured as {"policy": {"params": {<model keys>}}, "critic": ...}.
    # Unwrap to get the raw model params for BaseModelConfig.load().
    if "policy" in all_params:
        policy_params = all_params["policy"]
        if "params" in policy_params and len(policy_params) == 1:
            policy_params = policy_params["params"]
        logger.info(f"Extracted policy params with keys: {list(policy_params.keys())[:10]}")
    else:
        policy_params = all_params

    # Load model, bypassing BaseModelConfig.load()'s strict pytree equality check
    # which fails on Orbax's string-serialized integer keys. We use
    # replace_state_from_pure_dict_numeric_key_compat which handles the mismatch.
    from flax import nnx
    import orbax.checkpoint as ocp

    import openpi.shared.nnx_utils as _nnx_utils

    _model_instance = nnx.eval_shape(policy_model_config.create, jax.random.key(0))
    _graphdef, _state = nnx.split(_model_instance)
    policy_params = ocp.transform_utils.intersect_trees(_state.to_pure_dict(), policy_params)
    _nnx_utils.replace_state_from_pure_dict_numeric_key_compat(_state, policy_params)
    model = nnx.merge(_graphdef, _state)
    data_config_for_policy = config.data.create(config.assets_dirs, policy_model_config)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir_path / "assets", data_config_for_policy.asset_id)
    # Keep only the norm-stat keys the Normalize / Unnormalize transforms reference at
    # inference. Checkpoints also store "action_diff" (chunk-wise-delta stats); leaving
    # it in trips Unnormalize's strict key check since the output dict has no such key.
    # Mirrors best_of_n_policy._INFERENCE_NORM_KEYS.
    if norm_stats is not None:
        inference_norm_keys = {"state", "actions", "next_state", "next_actions"}
        norm_stats = {k: v for k, v in norm_stats.items() if k in inference_norm_keys}

    import openpi.policies.policy as _policy
    import openpi.transforms as _transforms

    policy = _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(None),
            *data_config_for_policy.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config_for_policy.use_quantile_norm),
            *(
                [_transforms.Clip(data_config_for_policy.clip_normalized_bounds)]
                if data_config_for_policy.clip_normalized_bounds is not None
                else []
            ),
            *data_config_for_policy.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config_for_policy.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles = data_config_for_policy.use_quantile_norm),
            *data_config_for_policy.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )

    # Extract model and transforms for batched inference.
    # We access private members because the Policy.infer() API only handles single
    # observations, and we need batched inference for efficiency.
    model = policy._model  # noqa: SLF001
    input_transform = policy._input_transform  # noqa: SLF001
    output_transform = policy._output_transform  # noqa: SLF001
    sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
    if args.sampling_num_steps is not None:
        sample_kwargs["num_steps"] = args.sampling_num_steps
        logger.info(f"Overriding flow-matching integration steps: num_steps={args.sampling_num_steps}")
    rng = policy._rng  # noqa: SLF001

    # Disable classifier-free guidance for counterfactual action generation.
    # Set it on the model config so the branch stays static under JIT.
    if hasattr(model, "config") and hasattr(model.config, "guidance"):
        object.__setattr__(model.config, "guidance", 0.0)
        logger.info("Disabled classifier-free guidance for sampling (guidance=0.0).")

    # JIT-compile sample_actions for the model
    from openpi.shared import nnx_utils

    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)
    has_prefix_cache = hasattr(model, "compute_prefix_cache")
    compute_prefix_cache_jit = nnx_utils.module_jit(model.compute_prefix_cache) if has_prefix_cache else None
    encode_images_jit = nnx_utils.module_jit(model.encode_images) if hasattr(model, "encode_images") else None
    profiled_batches = 0
    seen_sampling_batches = 0

    @dataclasses.dataclass
    class PendingRequest:
        step_idx: int
        strided_idx: int
        state: np.ndarray
        transformed: dict[str, Any]
        gt_actions: np.ndarray | None
        action_mask: np.ndarray | None
        encoded_images: jax.Array | None
        written_samples: int = 0

    def _build_policy_prompt(subtask_texts: list[str], first_null_index: int) -> str | None:
        valid_subtask_texts = []
        for text in subtask_texts[:first_null_index]:
            stripped = text.rstrip(". ").strip()
            lowered = stripped.lower()
            if lowered in {"static", "abnormal"}:
                continue
            valid_subtask_texts.append(stripped)
        if not valid_subtask_texts:
            return None
        return ", ".join(valid_subtask_texts)

    def _repeat_tree(tree: Any, repeats: int) -> Any:
        return jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (repeats, *np.shape(x))),
            tree,
        )

    def _concat_tree(trees: list[Any]) -> Any:
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)

    def _pad_batch_to_size(tree: Any, current_size: int, target_size: int) -> Any:
        """Pad arrays in tree along axis 0 to target_size to avoid JIT recompilation."""
        if current_size >= target_size:
            return tree
        pad_size = target_size - current_size

        def pad_array(x: jax.Array | None) -> jax.Array | None:
            if x is None:
                return None
            pad_width = [(0, pad_size)] + [(0, 0)] * (len(x.shape) - 1)
            return jnp.pad(x, pad_width)

        return jax.tree.map(pad_array, tree)

    def _construct_eef_repr_np(action: np.ndarray, eef_action: np.ndarray) -> np.ndarray:
        # Gripper slots within `action` depend on its dim: left at dim//2 - 1, right at
        # dim - 1, matching the 14D (6, 13) and 16D (7, 15) raw layouts. Mirrors
        # Hdf5RldsDataset._construct_eef_repr so 16D joint states map grippers correctly.
        total_dim = action.shape[-1]
        left_gripper_index = total_dim // 2 - 1
        right_gripper_index = total_dim - 1
        return np.concatenate(
            [
                eef_action[..., :6],
                action[..., left_gripper_index : left_gripper_index + 1],
                eef_action[..., 6:12],
                action[..., right_gripper_index : right_gripper_index + 1],
            ],
            axis = -1,
        ).astype(np.float32)

    def _read_existing_shard_metadata(shard_path: epath.Path) -> dict[str, int]:
        import tensorflow as tf

        episode_count = 0
        for _ in tf.data.TFRecordDataset([str(shard_path)]):
            episode_count += 1
        return {
            "episode_count": episode_count,
            "num_bytes": int(shard_path.stat().length),
        }

    def _select_policy_subtask(
        step: dict[str, Any], subtask_texts: list[str], fps: int
    ) -> tuple[str | None, int, np.ndarray]:
        first_null_index_val = int(step["first_null_index"])
        if first_null_index_val == 0:
            return None, 0, np.zeros(action_horizon, dtype = np.bool_)

        steps_all = np.asarray(step["steps_to_subtask_end"], dtype = np.int32)
        include_subtasks = np.zeros(max_subtasks, dtype = np.bool_)
        for idx, text in enumerate(subtask_texts[:first_null_index_val]):
            lowered = text.rstrip(". ").strip().lower()
            include_subtasks[idx] = lowered not in {"dummy", "static", "abnormal"}

        policy_prompt = _build_policy_prompt(subtask_texts, first_null_index_val)
        if policy_prompt is None:
            return None, 0, np.zeros(action_horizon, dtype = np.bool_)

        masked_steps = np.where(include_subtasks, steps_all, np.iinfo(np.int32).max)
        sampled_idx = int(np.argmin(masked_steps))
        selected_steps = int(steps_all[sampled_idx])
        if data_config.rlds_kwargs["mask_boundary_actions"]:
            action_mask = np.arange(action_horizon, dtype = np.int32) <= selected_steps
        else:
            action_mask = np.ones(action_horizon, dtype = np.bool_)
        if fps == 30:
            valid_30fps_actions = 3 * action_horizon // 5
            action_mask &= np.arange(action_horizon, dtype = np.int32) < valid_30fps_actions

        return policy_prompt, sampled_idx, action_mask

    # Get action dimensions from policy model config
    action_horizon = policy_model_config.action_horizon
    action_dim_mask = getattr(policy_model_config, "action_dim_mask", None)
    if action_dim_mask is not None:
        action_dim_mask = np.asarray(action_dim_mask, dtype = np.bool_)
        action_dim = int(np.sum(action_dim_mask))
        # Mirrors PadStatesAndActions._resolve_offset: contiguous True block.
        true_indices = np.where(action_dim_mask)[0]
        assert (true_indices == np.arange(true_indices[0], true_indices[-1] + 1)).all(), (
            f"action_dim_mask must be a contiguous True block, got {action_dim_mask.tolist()}"
        )
        action_dim_offset = int(true_indices[0])
    else:
        action_dim = policy_model_config.action_dim
        action_dim_offset = getattr(policy_model_config, "action_dim_offset", 0)
    max_subtasks = 5
    prompt_mode = data_config.rlds_kwargs.get("prompt_mode", "subtask")
    # LeRobot datasets (rlds_dataset_class="lerobot") carry a single per-step `subtask`
    # field — no subtask_1..5 / first_null_index — and the subtask prompt is that text
    # verbatim (LeRobotRldsDataset.frame_transforms: frame["prompt"] = frame["subtask"]).
    # RoboCOIN/HDF5 datasets instead carry subtask_1..5 and build a composite prompt.
    is_lerobot = getattr(data_config, "rlds_dataset_class", None) == "lerobot"

    def _select_task_description_prompt(
        step: dict[str, Any], fps: int, task_description: str,
    ) -> tuple[str | None, int, np.ndarray]:
        """Constant-per-episode prompt; mirrors hdf5_rlds_dataset.py:569-572."""
        if not task_description:
            return None, 0, np.zeros(action_horizon, dtype = np.bool_)
        if data_config.rlds_kwargs["mask_boundary_actions"]:
            steps_value = int(np.asarray(step["steps_to_subtask_end"]).flatten()[0])
            action_mask = np.arange(action_horizon, dtype = np.int32) <= steps_value
        else:
            action_mask = np.ones(action_horizon, dtype = np.bool_)
        if fps == 30:
            valid_30fps_actions = 3 * action_horizon // 5
            action_mask &= np.arange(action_horizon, dtype = np.int32) < valid_30fps_actions
        return task_description, 0, action_mask

    def _select_lerobot_subtask_prompt(
        step: dict[str, Any],
    ) -> tuple[str | None, int, np.ndarray]:
        """LeRobot subtask-conditioned prompt: the per-step `subtask` text verbatim.

        Mirrors LeRobotRldsDataset under prompt_mode="subtask"
        (frame["prompt"] = frame["subtask"]). The LeRobot schema exposes a single
        `subtask` field (no subtask_1..5 / first_null_index), so the prompt is that
        string as-is — training feeds it unmodified and the tokenizer only strips
        whitespace.

        action_mask is always all-ones: this config runs with critic_mode=False, where
        the training data path never sets an action_mask, so the model defaults it to
        all-ones (pi0.py).
        """
        subtask_text = step["subtask"]
        if hasattr(subtask_text, "numpy"):
            subtask_text = subtask_text.numpy()
        if isinstance(subtask_text, bytes):
            subtask_text = subtask_text.decode("utf-8")
        if not subtask_text:
            return None, 0, np.zeros(action_horizon, dtype = np.bool_)
        action_mask = np.ones(action_horizon, dtype = np.bool_)
        return subtask_text, 0, action_mask

    logger.info(
        f"Action horizon={action_horizon}, action_dim={action_dim}, num_samples={args.num_samples}, "
        f"action_dim_mask={action_dim_mask}, prompt_mode={prompt_mode}, "
        f"mask_boundary_actions={data_config.rlds_kwargs.get('mask_boundary_actions')}, "
        f"use_chunk_wise_delta={data_config.rlds_kwargs.get('use_chunk_wise_delta')}, "
        f"rng_seed=0 (hardcoded in Policy)"
    )

    # Create manifest
    manifest = ca_store.CounterfactualActionStoreManifest(
        version="1.0",
        source_rlds_data_dir=data_config.rlds_data_dir,
        source_dataset_name=dataset_cfg.name,
        source_dataset_version=dataset_cfg.version,
        source_num_episodes=total_episodes,
        num_samples=args.num_samples,
        action_dim=action_dim,
        action_horizon=action_horizon,
        stride=args.stride,
        policy_config_name=args.config_name,
        policy_checkpoint_dir=args.checkpoint_dir,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    total_written = 0
    shard_metadata: dict[int, dict[str, int]] = {}
    worker_times = defaultdict(float)
    timed_episode_count = 0

    def format_timing(times: dict[str, float]) -> str:
        total_time = sum(times.values())
        if total_time == 0:
            return "no timed work"
        parts = []
        for key, value in sorted(times.items(), key=lambda item: item[1], reverse=True):
            parts.append(f"{key}={value:.2f}s ({100.0 * value / total_time:.1f}%)")
        return ", ".join(parts)

    for shard_list_pos, shard_idx in enumerate(my_shards):
        # In reverse mode, stop early if the forward worker has caught up: if the 10th
        # upcoming shard (in our reversed order) already exists, both ends have converged.
        if args.reverse:
            lookahead = 10
            lookahead_pos = shard_list_pos + lookahead
            if lookahead_pos < len(my_shards):
                lookahead_shard_idx = my_shards[lookahead_pos]
                lookahead_path = (
                    worker_dir
                    / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME
                    / ca_store.VERSION
                    / ca_store.get_shard_filename(args.split, lookahead_shard_idx)
                )
                if lookahead_path.exists():
                    # On resume, the current shard already existing means this is
                    # our own prior reverse output (not forward convergence), so
                    # skip it and keep going; only stop if it is missing.
                    current_shard_path = (
                        worker_dir
                        / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME
                        / ca_store.VERSION
                        / ca_store.get_shard_filename(args.split, shard_idx)
                    )
                    if current_shard_path.exists():
                        logger.info(
                            "Worker %d: reverse lookahead shard %d (position +%d) exists but "
                            "current shard %d also exists (prior reverse output); skipping.",
                            worker_id, lookahead_shard_idx, lookahead, shard_idx,
                        )
                        continue
                    logger.info(
                        "Worker %d: reverse lookahead shard %d (position +%d) already exists, "
                        "forward worker has caught up. Stopping early.",
                        worker_id, lookahead_shard_idx, lookahead,
                    )
                    break

        start_pos, num_episodes = shard_info[shard_idx]

        # Apply max_episodes limit
        if args.max_episodes is not None:
            global_end = min(args.max_episodes, total_episodes)
            shard_end = start_pos + num_episodes
            if start_pos >= global_end:
                logger.info(f"Shard {shard_idx}: skipping (past max_episodes limit)")
                continue
            if shard_end > global_end:
                num_episodes = global_end - start_pos

        split_spec = f"{args.split}[{start_pos}:{start_pos + num_episodes}]"
        dataset = source_builder.as_dataset(split=split_spec)

        shard_path = (
            worker_dir
            / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME
            / ca_store.VERSION
            / ca_store.get_shard_filename(args.split, shard_idx)
        )
        if shard_path.exists():
            existing_metadata = _read_existing_shard_metadata(shard_path)
            if existing_metadata["episode_count"] == num_episodes:
                shard_metadata[shard_idx] = existing_metadata
                total_written += existing_metadata["episode_count"]
                logger.info(
                    "Worker %d shard %d: skipping existing shard %s (%d/%d episodes, %d bytes)",
                    worker_id,
                    shard_idx,
                    shard_path,
                    existing_metadata["episode_count"],
                    num_episodes,
                    existing_metadata["num_bytes"],
                )
                continue
            logger.warning(
                "Worker %d shard %d: existing shard %s has %d episodes, expected %d; recomputing shard.",
                worker_id,
                shard_idx,
                shard_path,
                existing_metadata["episode_count"],
                num_episodes,
            )


        # Initialize the shard_writer once per shard
        shard_writer = ca_store.CounterfactualActionStoreTFDSShardWriter(
            output_dir = str(worker_dir),
            shard_idx = shard_idx,
            manifest = manifest,
            split = args.split,
        )

        for raw_episode in tqdm.tqdm(dataset, total=num_episodes, desc=f"Worker {worker_id} Shard {shard_idx}"):
            episode_start_time = time.perf_counter()
            episode_timer = Timer()

            with episode_timer.context("materialize_episode"):
                episode = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in raw_episode.items()}
                episode["steps"] = list(episode["steps"])

            rlds_episode_index = get_rlds_episode_index(episode)
            num_steps = len(episode["steps"])

            repo_id = episode["episode_metadata"]["repo_id"]
            if hasattr(repo_id, "numpy"):
                repo_id = repo_id.numpy()
            embodiment = extract_embodiment(repo_id)
            fps = int(episode["episode_metadata"]["fps"])
            # Optional (dexterous/hdf5 configs only): whether this episode carries subtask
            # boundary annotations. Logged alongside the debug metrics below.
            has_subtask_annotations = episode["episode_metadata"].get("has_subtask_annotations")
            if hasattr(has_subtask_annotations, "numpy"):
                has_subtask_annotations = has_subtask_annotations.numpy()
            if has_subtask_annotations is not None:
                has_subtask_annotations = bool(has_subtask_annotations)
            episode_task_description = ""
            if prompt_mode == "task_description":
                td = episode["episode_metadata"]["task_description"]
                if hasattr(td, "numpy"):
                    td = td.numpy()
                if isinstance(td, bytes):
                    td = td.decode("utf-8")
                # Pass the raw task_description verbatim; training feeds it unmodified
                # (hdf5_rlds_dataset.py: frame["prompt"] = frame["task_description"]) and
                # the tokenizer only strips whitespace. Stripping a trailing period here
                # would diverge the tokenized prompt from training.
                episode_task_description = td or ""
            episode_actions = None
            if args.debug_metrics:
                episode_actions = []
                for step in episode["steps"]:
                    step_action = step["action"]
                    if hasattr(step_action, "numpy"):
                        step_action = step_action.numpy()
                    step_action = np.asarray(step_action, dtype = np.float32)
                    if data_config_for_policy.use_eef:
                        step_eef_action = step["eef_sim_pose_action"]
                        if hasattr(step_eef_action, "numpy"):
                            step_eef_action = step_eef_action.numpy()
                        step_eef_action = np.asarray(step_eef_action, dtype = np.float32)
                        step_action = _construct_eef_repr_np(step_action, step_eef_action)
                    episode_actions.append(step_action)
                episode_actions = np.asarray(episode_actions, dtype = np.float32)

            # Allocate output array (strided dimensions if stride > 1)
            num_strided_positions = (num_steps + args.stride - 1) // args.stride
            all_actions = np.zeros(
                (num_strided_positions, args.num_samples, action_horizon, action_dim),
                dtype=np.float32,
            )
            episode_valid_frames = 0
            episode_sampling_l1_sum = 0.0
            episode_sampling_mse_sum = 0.0
            episode_sampling_num_valid = 0.0
            episode_cov_trace_sum = 0.0
            episode_cov_trace_num_valid = 0.0

            if prompt_mode == "task_description":
                skip_whole_episode = not episode_task_description
            elif is_lerobot:
                # LeRobot subtask schema has no first_null_index and every step normally
                # carries a `subtask`; process the episode and let the per-step path skip
                # any empty-subtask step (leaving all-zeros, same as a whole-episode skip).
                skip_whole_episode = False
            else:
                skip_whole_episode = not any(int(step["first_null_index"]) > 0 for step in episode["steps"])
            if skip_whole_episode:
                # No valid prompt for this episode, leave all zeros
                pass
            else:
                pending_requests: deque[PendingRequest] = deque()
                step_requests: dict[int, list[PendingRequest]] = {}
                step_encoding_inputs: list[tuple[int, dict[str, Any]]] = []
                encode_batch_size = min(args.samples_per_batch, 32)

                def flush_pending_step_encodes(
                    step_encoding_inputs: list = step_encoding_inputs,
                    step_requests: dict = step_requests,
                    episode_timer: Timer = episode_timer,
                    encode_batch_size: int = encode_batch_size,
                ) -> None:
                    if not step_encoding_inputs:
                        return

                    actual_batch_size = len(step_encoding_inputs)
                    batched_inputs = _concat_tree(
                        [_repeat_tree(transformed, 1) for _, transformed in step_encoding_inputs]
                    )

                    # Pad to fixed size to avoid JIT recompilation for partial batches
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, encode_batch_size)

                    encoded_observation = _model.Observation.from_dict(batched_inputs)
                    with episode_timer.context("encode_images"):
                        batch_encoded_images = encode_images_jit(encoded_observation.images)
                        batch_encoded_images = jax.block_until_ready(batch_encoded_images)

                    # Slice back to actual size (discard padding)
                    batch_encoded_images = batch_encoded_images[:actual_batch_size]

                    for encoded_idx, (encoded_step_idx, _) in enumerate(step_encoding_inputs):
                        step_encoded_images = batch_encoded_images[encoded_idx : encoded_idx + 1]
                        for request in step_requests[encoded_step_idx]:
                            request.encoded_images = step_encoded_images

                    step_encoding_inputs.clear()

                for strided_idx, step_idx in enumerate(range(0, num_steps, args.stride)):
                    step = episode["steps"][step_idx]

                    # subtask_1..subtask_5 exist only in the RoboCOIN/HDF5 subtask schema
                    # and feed the composite subtask prompt path; task_description mode and
                    # the LeRobot single-`subtask` mode don't use them, so skip the read.
                    subtask_texts = []
                    if prompt_mode != "task_description" and not is_lerobot:
                        for si in range(1, max_subtasks + 1):
                            text = step[f"subtask_{si}"]
                            if hasattr(text, "numpy"):
                                text = text.numpy()
                            if isinstance(text, bytes):
                                text = text.decode("utf-8")
                            subtask_texts.append(text)


                    if prompt_mode == "task_description":
                        policy_prompt, sampled_idx, action_mask = _select_task_description_prompt(
                            step, fps, episode_task_description,
                        )
                    elif is_lerobot:
                        # LeRobot subtask-conditioned caching only supports prompt_mode="subtask"
                        # (prompt = per-step subtask verbatim). Fail loudly on anything else.
                        if prompt_mode != "subtask":
                            raise ValueError(
                                f"LeRobot counterfactual caching supports prompt_mode='subtask', got {prompt_mode!r}."
                            )
                        policy_prompt, sampled_idx, action_mask = _select_lerobot_subtask_prompt(step)
                    else:
                        policy_prompt, sampled_idx, action_mask = _select_policy_subtask(
                            step, subtask_texts, fps,
                        )
                    if policy_prompt is None:
                        continue
                    episode_valid_frames += 1

                    first_transformed_for_step = None

                    # Extract observation data once per step (shared across subtasks)
                    step_state = step["observation/state"]
                    if hasattr(step_state, "numpy"):
                        step_state = step_state.numpy()
                    step_state = np.array(step_state, dtype=np.float32)
                    if data_config_for_policy.use_eef:
                        # Mirrors RoboCoinRldsDataset._construct_eef_state: 14-D EEF layout
                        # (left xyz/rpy + left gripper + right xyz/rpy + right gripper)
                        # built from 12-D eef_sim_pose_state and the joint state's gripper slots.
                        step_eef_state = step["eef_sim_pose_state"]
                        if hasattr(step_eef_state, "numpy"):
                            step_eef_state = step_eef_state.numpy()
                        step_eef_state = np.asarray(step_eef_state, dtype = np.float32)
                        step_state = _construct_eef_repr_np(step_state, step_eef_state)
                    elif data_config.rlds_kwargs.get("state_dim") == 16 and step_state.shape[-1] == 14:
                        step_state = np.concatenate([step_state[:6], [0.0], step_state[6:13], [0.0], step_state[13:]], axis = 0).astype(np.float32)

                    # Decode images once per step (not per subtask). Resize to the model's
                    # 224x224 input happens later in the ResizeImages model transform.
                    decoded_images = {}
                    with episode_timer.context("image_decode"):
                        for cam_key in (
                            "observation/image/cam_0",
                            "observation/image/cam_1",
                            "observation/image/cam_2",
                        ):
                            img_data = step[cam_key]
                            if hasattr(img_data, "numpy"):
                                img_data = img_data.numpy()
                            cam_name = cam_key.split("/")[-1]
                            standard_cam_name = RLDS_TO_STANDARD_CAMERA_MAP[cam_name]
                            decoded_images[standard_cam_name] = np.asarray(
                                tf.io.decode_image(img_data, expand_animations = False, dtype = tf.uint8).numpy()
                            )

                    obs_dict: dict[str, Any] = {
                        "image": dict(decoded_images),
                        "state": step_state,
                        "prompt": policy_prompt,
                        "embodiment": embodiment,
                    }

                    transform_with_actions = dict(obs_dict)
                    transform_with_actions["action_mask"] = action_mask
                    if args.debug_metrics:
                        gt_action_indices = np.minimum(
                            step_idx + np.arange(action_horizon, dtype = np.int32),
                            num_steps - 1,
                        )
                        # Pass absolute actions; the input_transform's DeltaActions step will
                        # convert to state-relative delta when use_chunk_wise_delta is enabled.
                        gt_actions = episode_actions[gt_action_indices].copy()
                        transform_with_actions["actions"] = gt_actions

                    with episode_timer.context("input_transform"):
                        transformed = input_transform(transform_with_actions)

                    gt_actions_transformed = None
                    action_mask_transformed = action_mask.astype(np.bool_)
                    if args.debug_metrics:
                        gt_actions_transformed = np.asarray(transformed["actions"], dtype = np.float32)

                    transformed = {
                        k: v
                        for k, v in transformed.items()
                        if not isinstance(v, str) and k not in ("embodiment", "actions")
                    }
                    first_transformed_for_step = transformed
                    if encode_images_jit is not None:
                        transformed = {k: v for k, v in transformed.items() if k != "image"}

                    request = PendingRequest(
                        step_idx=step_idx,
                        strided_idx=strided_idx,
                        state=step_state,
                        transformed=transformed,
                        gt_actions=gt_actions_transformed,
                        action_mask=action_mask_transformed,
                        encoded_images=None,
                    )
                    pending_requests.append(request)
                    step_requests.setdefault(step_idx, []).append(request)

                    if first_transformed_for_step is not None and encode_images_jit is not None:
                        step_encoding_inputs.append((step_idx, first_transformed_for_step))
                        if len(step_encoding_inputs) >= encode_batch_size:
                            flush_pending_step_encodes()

                if encode_images_jit is not None:
                    flush_pending_step_encodes()

                while pending_requests:
                    batch_requests: list[tuple[PendingRequest, int]] = []
                    remaining_capacity = args.samples_per_batch

                    while pending_requests and remaining_capacity > 0:
                        request = pending_requests.popleft()
                        num_to_take = min(args.num_samples - request.written_samples, remaining_capacity)
                        batch_requests.append((request, num_to_take))
                        request.written_samples += num_to_take
                        remaining_capacity -= num_to_take
                        if request.written_samples < args.num_samples:
                            pending_requests.appendleft(request)

                    actual_batch_size = sum(n for _, n in batch_requests)
                    batched_inputs = _concat_tree(
                        [_repeat_tree(request.transformed, n) for request, n in batch_requests]
                    )

                    # Pad to fixed size to avoid JIT recompilation for partial batches
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, args.samples_per_batch)

                    observation = _model.Observation.from_dict(batched_inputs)
                    transition = _model.wrap_observation_as_transition(observation)

                    # Compute prefix KV cache from unique observations, then repeat to match sample counts.
                    prefix_cache = None
                    if compute_prefix_cache_jit is not None:
                        unique_batch_size = args.samples_per_batch // args.num_samples
                        unique_inputs = _concat_tree(
                            [_repeat_tree(request.transformed, 1) for request, _ in batch_requests]
                        )
                        unique_inputs = _pad_batch_to_size(unique_inputs, len(batch_requests), unique_batch_size)
                        unique_obs = _model.Observation.from_dict(unique_inputs)
                        raw_cache = compute_prefix_cache_jit(unique_obs)
                        raw_kv_cache, raw_prefix_mask = raw_cache
                        repeated_kv_cache = jax.tree.map(
                            lambda x: jnp.repeat(x, args.num_samples, axis = 1),
                            raw_kv_cache,
                        )
                        repeated_prefix_mask = jnp.repeat(raw_prefix_mask, args.num_samples, axis = 0)
                        prefix_cache = (repeated_kv_cache, repeated_prefix_mask)

                    rng, sample_rng = jax.random.split(rng)
                    call_kwargs = sample_kwargs
                    if prefix_cache is not None:
                        call_kwargs = {**call_kwargs, "prefix_cache": prefix_cache}
                    if batch_requests[0][0].encoded_images is not None:
                        batch_encoded_images = jnp.concatenate(
                            [
                                jnp.broadcast_to(
                                    request.encoded_images,
                                    (n, *request.encoded_images.shape[1:]),
                                )
                                for request, n in batch_requests
                            ],
                            axis=0,
                        )
                        # Pad encoded images to fixed size as well
                        batch_encoded_images = _pad_batch_to_size(
                            batch_encoded_images, actual_batch_size, args.samples_per_batch
                        )
                        call_kwargs = {**call_kwargs, "encoded_images": batch_encoded_images}

                    should_profile_batch = (
                        args.profile_log_dir is not None
                        and seen_sampling_batches >= args.profile_skip_batches
                        and profiled_batches < args.profile_num_batches
                    )
                    if should_profile_batch:
                        with (
                            jax.profiler.trace(args.profile_log_dir, create_perfetto_link=False),
                            jax.profiler.StepTraceAnnotation("counterfactual_sample_batch", step_num=profiled_batches),
                            episode_timer.context("sample_actions"),
                        ):
                            actions_out = sample_actions_jit(sample_rng, transition, **call_kwargs)
                            actions_out = jax.block_until_ready(actions_out)
                        logger.info(
                            "Wrote JAX trace for profiled batch %d/%d (sampling batch index %d) to %s",
                            profiled_batches + 1,
                            args.profile_num_batches,
                            seen_sampling_batches,
                            args.profile_log_dir,
                        )
                        profiled_batches += 1
                    else:
                        with episode_timer.context("sample_actions"):
                            actions_out = sample_actions_jit(sample_rng, transition, **call_kwargs)
                            actions_out = jax.block_until_ready(actions_out)
                    seen_sampling_batches += 1

                    # Slice back to actual size (discard padding)
                    actions_out = actions_out[:actual_batch_size]

                    with episode_timer.context("sample_actions_to_host"):
                        actions_np = np.asarray(actions_out)
                    # Slice the padded model output (action_dim) down to the meaningful
                    # contiguous block before output_transform. norm_stats["actions"] is
                    # stored at the unpadded dim, and Unnormalize is strict on shape;
                    # without this slice it would assert. Mirrors BestOfNPolicy.infer
                    # (src/openpi/policies/best_of_n_policy.py:621-637).
                    actions_np = actions_np[..., action_dim_offset : action_dim_offset + action_dim]
                    batch_states = np.concatenate(
                        [
                            np.broadcast_to(request.transformed["state"][None], (n, *request.transformed["state"].shape))
                            for request, n in batch_requests
                        ],
                        axis=0,
                    )
                    with episode_timer.context("output_transform"):
                        transformed_outputs = output_transform(
                            {
                                "embodiment": embodiment,
                                "state": batch_states,
                                "actions": actions_np,
                                "next_state": batch_states,
                                "next_actions": actions_np,
                            }
                        )
                    output_actions = transformed_outputs["actions"]
                    if args.convert_joint_actions_to_eef:
                        # Must run here: output_transform has already applied
                        # Unnormalize + AbsoluteActions, so these are absolute joint
                        # positions. FK on a delta or on normalized values is meaningless.
                        output_actions = yam_eef.joint_actions_to_eef(
                            output_actions, yam_fk_dir = args.yam_fk_dir,
                        )

                    offset = 0
                    for request, n in batch_requests:
                        request_actions = actions_np[offset : offset + n]
                        if args.debug_metrics:
                            gt_actions = request.gt_actions
                            if action_dim_mask is not None:
                                gt_actions = gt_actions[..., action_dim_mask]
                            gt_actions_broadcast = np.broadcast_to(
                                gt_actions[None, ...],
                                request_actions.shape,
                            )
                            abs_diff = np.abs(request_actions - gt_actions_broadcast)
                            sq_diff = np.square(request_actions - gt_actions_broadcast)

                            l1_per_step = np.mean(abs_diff, axis = -1)
                            mse_per_step = np.mean(sq_diff, axis = -1)

                            step_mask = request.action_mask.astype(np.float32)[None, :]
                            episode_sampling_l1_sum += float(np.sum(l1_per_step * step_mask))
                            episode_sampling_mse_sum += float(np.sum(mse_per_step * step_mask))
                            episode_sampling_num_valid += float(np.sum(step_mask) * n)

                        centered = request_actions - np.mean(request_actions, axis = 0, keepdims = True)
                        cov_trace_per_timestep = np.sum(np.mean(np.square(centered), axis = 0), axis = -1)
                        episode_cov_trace_sum += float(np.mean(cov_trace_per_timestep))
                        episode_cov_trace_num_valid += 1.0

                        start = request.written_samples - n
                        end = request.written_samples
                        all_actions[request.strided_idx, start:end] = output_actions[
                            offset : offset + n
                        ]
                        offset += n

            with episode_timer.context("write_episode"):
                shard_writer.write_episode(
                    episode_index=rlds_episode_index,
                    num_steps=num_steps,
                    actions=all_actions,
                )
            episode_elapsed = time.perf_counter() - episode_start_time
            episode_times = episode_timer.get_total_times(reset=True)
            for key, value in episode_times.items():
                worker_times[key] += value
            timed_episode_count += 1
            logger.info(
                "Worker %d shard %d episode_index=%d num_steps=%d valid_frames=%d "
                "has_subtask_annotations=%s "
                "sampling_l1=%.6f sampling_mse=%.6f cov_trace_per_timestep=%.6f "
                "debug_metrics=%s elapsed=%.2fs",
                worker_id,
                shard_idx,
                rlds_episode_index,
                num_steps,
                episode_valid_frames,
                has_subtask_annotations,
                episode_sampling_l1_sum / max(episode_sampling_num_valid, 1.0),
                episode_sampling_mse_sum / max(episode_sampling_num_valid, 1.0),
                episode_cov_trace_sum / max(episode_cov_trace_num_valid, 1.0),
                args.debug_metrics,
                episode_elapsed,
            )
            logger.info(
                "Worker %d shard %d episode_index=%d timing: %s",
                worker_id,
                shard_idx,
                rlds_episode_index,
                format_timing(episode_times),
            )

        shard_writer.finalize()
        total_written += shard_writer.episode_count

        shard_metadata[shard_idx] = {
            "episode_count": shard_writer.episode_count,
            "num_bytes": shard_writer.total_bytes,
        }

    if not args.reverse:
        # Save manifest
        ca_store.save_manifest(manifest, str(worker_dir))

        # Save shard metadata
        shard_metadata_path = worker_dir / "shard_metadata.json"
        with shard_metadata_path.open("w") as f:
            json.dump(shard_metadata, f, indent=2)

        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if timed_episode_count > 0:
        average_times = {key: value / timed_episode_count for key, value in worker_times.items()}
        logger.info(
            "Worker %d timing totals across %d episodes: %s",
            worker_id,
            timed_episode_count,
            format_timing(dict(worker_times)),
        )
        logger.info(
            "Worker %d timing averages per episode: %s",
            worker_id,
            format_timing(average_times),
        )

    logger.info(f"Worker {worker_id}: done. Wrote {total_written} episodes across {len(shard_metadata)} shards.")


# =============================================================================
# Launch implementation
# =============================================================================


def run_launch(args: LaunchArgs) -> None:
    """Submit N sbatch worker jobs from the cluster head node."""
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")

    if args.partition_split is not None:
        worker_partitions = parse_partition_split(args.partition_split, args.num_workers)
    else:
        worker_partitions = [args.partition] * args.num_workers

    if args.total_episodes is not None:
        total_episodes = args.total_episodes
    else:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        _, data_config, dataset_cfg, _ = _resolve_config_with_fine_tune(
            args.config_name, args.fine_tune, args.rlds_data_dir,
        )
        total_episodes = get_total_episodes(data_config.rlds_data_dir, dataset_cfg, args.split)

    logger.info(f"Total source episodes: {total_episodes}")
    for worker_id in range(args.num_workers):
        logger.info(f"  Worker {worker_id} -> {worker_partitions[worker_id]}")

    log_dir = epath.Path(args.log_dir_base).expanduser() / args.config_name
    if args.fine_tune is not None:
        log_dir = log_dir / args.fine_tune
    if args.ssh_host:
        subprocess.run(["ssh", args.ssh_host, f"mkdir -p {shlex.quote(str(log_dir))}"], check=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    def _submit_sbatch(cmd: list[str]) -> str:
        if args.ssh_host:
            remote_cmd = " ".join(shlex.quote(arg) for arg in cmd)
            result = subprocess.run(["ssh", args.ssh_host, remote_cmd], capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split(";")[0]

    # Workers run from this checkout; the launcher only needs the repo root to be reachable
    # under the same path on the compute nodes.
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    def _make_wrap_cmd(inner_cmd: str) -> str:
        # Propagate OPENPI_DATA_HOME from the launcher env to each worker so a pre-staged
        # local asset cache (e.g. the PaliGemma tokenizer) is used instead of refetching
        # from GCS on DNS-flaky nodes. No-op when unset (workers fall back to their default).
        openpi_data_home = os.environ.get("OPENPI_DATA_HOME")
        data_home_export = (
            f"export OPENPI_DATA_HOME={shlex.quote(openpi_data_home)} && " if openpi_data_home else ""
        )
        return (
            f'bash -lc "'
            f"source ~/.bashrc && "
            # Force the CUDA backend: sbatch --export=ALL propagates the submit shell's env, so a stray
            # JAX_PLATFORMS=cpu (e.g. left over from CPU-only debugging) would otherwise silently push
            # the policy onto CPU (~1000x slower). Exporting here overrides any inherited value. Use
            # "cuda" not "gpu": "gpu" also tries the rocm backend, which fails fatally on these nodes.
            f"export JAX_PLATFORMS=cuda && "
            f"{data_home_export}"
            f"export CURL_CA_BUNDLE=\\$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || echo /etc/ssl/certs/ca-bundle.crt) && "
            f"export REQUESTS_CA_BUNDLE=\\$CURL_CA_BUNDLE && "
            f"export SSL_CERT_FILE=\\$CURL_CA_BUNDLE && "
            f"cd {shlex.quote(str(repo_root))} && {inner_cmd}"
            f'"'
        )

    extra_worker_args = [
        "--split",
        args.split,
        "--samples-per-batch",
        str(args.samples_per_batch),
        "--num-samples",
        str(args.num_samples),
        "--stride",
        str(args.stride),
    ]
    if args.max_episodes is not None:
        extra_worker_args += ["--max-episodes", str(args.max_episodes)]
    if args.debug_metrics:
        extra_worker_args += ["--debug-metrics"]
    if args.reverse:
        extra_worker_args += ["--reverse"]
    if args.fine_tune is not None:
        extra_worker_args += ["--fine-tune", args.fine_tune]
    if args.rlds_data_dir is not None:
        extra_worker_args += ["--rlds-data-dir", args.rlds_data_dir]
    if args.sampling_num_steps is not None:
        extra_worker_args += ["--sampling-num-steps", str(args.sampling_num_steps)]
    if args.convert_joint_actions_to_eef:
        extra_worker_args += ["--convert-joint-actions-to-eef", "--yam-fk-dir", args.yam_fk_dir]

    extra_args_str = " ".join(extra_worker_args)

    job_ids = []
    for worker_id in range(args.num_workers):
        worker_partition = worker_partitions[worker_id]
        worker_cmd = (
            f"{args.uv_bin} run --no-sync scripts/compute_counterfactual_actions.py worker"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --worker-id {worker_id}"
        )
        if extra_args_str:
            worker_cmd += f" {extra_args_str}"

        wrap_cmd = _make_wrap_cmd(worker_cmd)
        log_suffix = "_reverse" if args.reverse else ""
        log_pattern = str(log_dir / f"ca_store_worker_{worker_id}{log_suffix}.log")

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p",
            worker_partition,
            "--mem",
            args.mem,
            "--gres",
            args.gres,
            "--time",
            args.time_limit,
            "--output",
            log_pattern,
            f"--job-name=ca_w{worker_id}",
            "--wrap",
            wrap_cmd,
        ]
        if args.cpus_per_task is not None:
            sbatch_cmd.extend(["--cpus-per-task", str(args.cpus_per_task)])
        if args.qos is not None:
            sbatch_cmd.extend(["--qos", args.qos])
        if args.extra_args is not None:
            sbatch_cmd.extend(shlex.split(args.extra_args))

        if args.dry_run:
            logger.info(f"[DRY RUN] Worker {worker_id}: {' '.join(sbatch_cmd)}")
            job_ids.append(f"DRY_{worker_id}")
        else:
            job_id = _submit_sbatch(sbatch_cmd)
            job_ids.append(job_id)
            logger.info(f"Worker {worker_id}: submitted job {job_id} (log: {log_pattern})")

    if args.auto_merge and not args.dry_run:
        merge_cmd = (
            f"{args.uv_bin} run --no-sync scripts/compute_counterfactual_actions.py merge"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --split {args.split}"
        )

        merge_wrap = _make_wrap_cmd(merge_cmd)
        dep_str = ":".join(job_ids)
        merge_log_pattern = str(log_dir / "ca_store_merge.log")

        merge_partition = worker_partitions[0]
        merge_sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p",
            merge_partition,
            "--mem",
            "32GB",
            "--gres",
            args.gres,
            "--time",
            "04:00:00",
            "--output",
            merge_log_pattern,
            "--job-name=ca_merge",
            f"--dependency=afterok:{dep_str}",
            "--wrap",
            merge_wrap,
        ]
        if args.cpus_per_task is not None:
            merge_sbatch_cmd.extend(["--cpus-per-task", str(args.cpus_per_task)])
        if args.qos is not None:
            merge_sbatch_cmd.extend(["--qos", args.qos])
        if args.extra_args is not None:
            merge_sbatch_cmd.extend(shlex.split(args.extra_args))
        merge_job_id = _submit_sbatch(merge_sbatch_cmd)
        logger.info(f"Merge job submitted: {merge_job_id} (depends on workers: {dep_str})")

    logger.info("All jobs submitted.")
    if not args.auto_merge:
        logger.info(
            "To merge after all workers finish, run:\n"
            f"  uv run scripts/compute_counterfactual_actions.py merge"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
        )


# =============================================================================
# Merge implementation
# =============================================================================


def run_merge(args: MergeArgs) -> None:
    """Merge per-worker shard outputs into a single counterfactual action store TFDS dataset."""
    import concurrent.futures

    import tensorflow_datasets as tfds

    import openpi.training.counterfactual_action_store as ca_store

    output_dir = epath.Path(args.output_dir)
    source_dir = epath.Path(args.source_dir) if args.source_dir else output_dir
    merged_dataset_dir = output_dir / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / ca_store.VERSION

    if source_dir != output_dir:
        logger.info(f"Reading workers from: {source_dir}")
        logger.info(f"Writing merged output to: {output_dir}")

    if merged_dataset_dir.exists():
        if args.overwrite:
            logger.info(f"Removing existing merged output: {merged_dataset_dir}")
            merged_dataset_dir.rmtree()
        else:
            raise FileExistsError(f"Merged output already exists: {merged_dataset_dir}. Use --overwrite.")

    # Verify all workers are done
    missing_workers = []
    for worker_id in range(args.num_workers):
        done_marker = source_dir / "_workers" / f"worker_{worker_id}" / "_DONE"
        if not done_marker.exists():
            missing_workers.append(worker_id)

    if missing_workers:
        raise RuntimeError(f"Missing _DONE markers for workers: {missing_workers}. Not all workers have completed.")

    # Collect shard metadata from all workers
    shard_data: dict[int, dict] = {}
    worker_shards: dict[int, list[int]] = {}
    reference_manifest: ca_store.CounterfactualActionStoreManifest | None = None

    for worker_id in range(args.num_workers):
        worker_dir = source_dir / "_workers" / f"worker_{worker_id}"
        worker_dataset_dir = worker_dir / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / ca_store.VERSION

        if reference_manifest is None:
            manifest_path = worker_dir / "manifest.json"
            if manifest_path.exists():
                reference_manifest = ca_store.load_manifest(str(worker_dir))

        metadata_path = worker_dir / "shard_metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Worker {worker_id}: shard_metadata.json not found, skipping.")
            continue

        with metadata_path.open("r") as f:
            worker_metadata = json.load(f)

        for shard_idx_str, info in worker_metadata.items():
            shard_idx = int(shard_idx_str)
            if shard_idx in shard_data:
                raise RuntimeError(f"Duplicate shard {shard_idx} from multiple workers")

            shard_path = worker_dataset_dir / ca_store.get_shard_filename(args.split, shard_idx)
            if not shard_path.exists():
                raise FileNotFoundError(f"Expected shard file not found: {shard_path}")

            shard_data[shard_idx] = {
                "episode_count": info["episode_count"],
                "num_bytes": info["num_bytes"],
                "source_path": shard_path,
            }
            worker_shards.setdefault(worker_id, []).append(shard_idx)

        logger.info(f"Worker {worker_id}: {len(worker_metadata)} shards")

    if not shard_data:
        raise RuntimeError("No shard data found across any workers.")

    if reference_manifest is None:
        raise RuntimeError("Could not load manifest from any worker.")

    sorted_shard_indices = sorted(shard_data.keys())
    max_shard_idx = max(sorted_shard_indices)

    if sorted_shard_indices != list(range(max_shard_idx + 1)):
        missing = set(range(max_shard_idx + 1)) - set(sorted_shard_indices)
        raise RuntimeError(f"Missing shards: {sorted(missing)}. Cannot create valid TFDS dataset.")

    shard_lengths = [shard_data[i]["episode_count"] for i in sorted_shard_indices]
    total_num_bytes = sum(shard_data[i]["num_bytes"] for i in sorted_shard_indices)
    total_episodes = sum(shard_lengths)
    num_shards = len(sorted_shard_indices)

    logger.info(f"Merge plan: {total_episodes} episodes in {num_shards} shards.")

    merged_dataset_dir.mkdir(parents=True, exist_ok=True)

    def _copy_shard(shard_idx: int, source_path: epath.Path) -> None:
        dest_path = merged_dataset_dir / ca_store.get_shard_filename(args.split, shard_idx)
        logger.info(f"Copying shard {shard_idx}: {source_path} -> {dest_path}")
        source_path.copy(dest_path)

    # Copy + (optionally) delete one worker at a time so disk is reclaimed
    # incrementally instead of holding every worker's shards through merge end.
    for worker_id in range(args.num_workers):
        these_shards = worker_shards.get(worker_id, [])
        if not these_shards:
            continue
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_copy_shard, sid, shard_data[sid]["source_path"])
                for sid in these_shards
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        if args.cleanup_workers:
            worker_root = source_dir / "_workers" / f"worker_{worker_id}"
            logger.info(f"Cleaning up worker {worker_id}: {worker_root}")
            worker_root.rmtree()

    features = ca_store.get_counterfactual_action_store_tfds_feature_spec(reference_manifest)

    dataset_name = ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME

    filename_template = tfds.core.ShardedFileTemplate(
        dataset_name=dataset_name,
        split=args.split,
        filetype_suffix="tfrecord",
        data_dir=str(merged_dataset_dir),
        template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
    )

    split_info = tfds.core.SplitInfo(
        name=args.split,
        shard_lengths=shard_lengths,
        num_bytes=total_num_bytes,
        filename_template=filename_template,
    )

    identity = tfds.core.DatasetIdentity(
        name=dataset_name,
        version=tfds.core.Version(ca_store.VERSION),
        data_dir=str(merged_dataset_dir),
        module_name=__name__,
    )
    merged_info = tfds.core.DatasetInfo(
        builder=identity,
        description=f"Counterfactual action store for {reference_manifest.source_dataset_name}",
        features=features,
    )
    merged_info.set_splits(tfds.core.SplitDict([split_info]))
    merged_info.write_to_directory(merged_dataset_dir)

    ca_store.save_manifest(reference_manifest, str(output_dir))

    # Verify
    verify_builder = tfds.builder_from_directory(str(merged_dataset_dir))
    verify_count = verify_builder.info.splits[args.split].num_examples
    if verify_count != total_episodes:
        raise RuntimeError(
            f"Verification failed: expected {total_episodes} episodes, got {verify_count} in merged dataset."
        )
    logger.info(f"Merge complete: {verify_count} episodes in {num_shards} shards written to {merged_dataset_dir}")

    if args.cleanup_workers:
        workers_dir = source_dir / "_workers"
        try:
            if not list(workers_dir.iterdir()):
                workers_dir.rmtree()
        except Exception:
            pass


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    args = tyro.cli(
        Annotated[
            Annotated[LaunchArgs, tyro.conf.subcommand("launch")]
            | Annotated[WorkerArgs, tyro.conf.subcommand("worker")]
            | Annotated[MergeArgs, tyro.conf.subcommand("merge")],
            tyro.conf.OmitSubcommandPrefixes,
        ],
    )

    if isinstance(args, LaunchArgs):
        run_launch(args)
    elif isinstance(args, WorkerArgs):
        run_worker(args)
    elif isinstance(args, MergeArgs):
        run_merge(args)
    else:
        raise ValueError(f"Unknown args type: {type(args)}")


if __name__ == "__main__":
    main()
