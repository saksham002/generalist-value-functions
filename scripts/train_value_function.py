"""Training script for value functions (critics).

Trains MC, SARSA or TD best-of-N critics. The optional policy in the config is a frozen
best-of-N sampler over cached counterfactual actions used only for the TD backup.
"""

import dataclasses
import functools
import gc
import logging
import os
import pickle
import platform as _platform
import threading
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.rlds_utils.load_model_utils as _load_model_utils
from openpi.rlds_utils.utils import SnapshotConfig
from openpi.rlds_utils.utils import apply_override_prompt
from openpi.rlds_utils.utils import cache_val_episodes
from openpi.rlds_utils.utils import count_subtask_segments
from openpi.rlds_utils.utils import decode_episode_images
from openpi.rlds_utils.utils import decode_text
from openpi.rlds_utils.utils import inject_shuffled_actions
from openpi.rlds_utils.utils import predict_values
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
from openpi.training.time_utils import Timer
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _value_fn
import openpi.value_functions.value_function as _value_fn_impl


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    ft_config: _config.FineTuneConfig | None = None,
    start_new: bool = False,
):
    # Only worker 0 should initialize wandb to avoid file conflicts and duplicate runs
    if not enabled or jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    # Fine-tune runs own a separate wandb run + wandb_id.txt under the FT subdir
    # (config.checkpoint_dir / ft_config.name), decoupled from the base run so a
    # fine-tune launched with --resume (needed to load the pretrained critic) does
    # not reattach to the base run's wandb. The subdir may not exist yet for a fresh
    # FT (ft_config.initialize creates it later), so create it here before writing
    # the run id; whether to resume the FT run is keyed off that subdir's
    # wandb_id.txt rather than the base checkpoint's `resuming` flag.
    if ft_config is not None:
        wandb_id_dir = ckpt_dir / ft_config.name
        wandb_id_dir.mkdir(parents = True, exist_ok = True)
        resume_run = (wandb_id_dir / "wandb_id.txt").exists()
    else:
        wandb_id_dir = ckpt_dir
        resume_run = resuming
    if resume_run and not start_new:
        run_id = (wandb_id_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="allow", project=config.project_name)
    else:
        base_name = config.exp_name if config.exp_name else config.name
        run_name = f"{base_name}/{ft_config.name}" if ft_config is not None else base_name
        wandb.init(
            name=run_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group=config.wandb_group,
        )
        (wandb_id_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _cast_param(param, dtype):
    """Cast a param's value; nnx layers created with use_bias=False hold Param(None) biases, left as is."""
    return param if param.value is None else param.replace(param.value.astype(dtype))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.ActorCriticTrainState, Any]:
    """Initialize training state for a value function model (and optional policy)."""
    critic_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(f"Expected BaseValueFunctionConfig, got {type(config.model)}")
    if isinstance(config.model, _value_fn_impl.ValueFunctionConfig):
        model_config: _value_fn.BaseValueFunctionConfig = dataclasses.replace(
            config.model, action_horizon = config.action_horizon,
        )
    else:
        model_config = config.model

    def init_critic(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: _cast_param(p, jnp.bfloat16),
        )
        weight_dtype = jnp.dtype(model_config.weight_dtype) if hasattr(model_config, "weight_dtype") else jnp.float32
        target_dtype = jnp.dtype(model_config.target_dtype) if hasattr(model_config, "target_dtype") else jnp.float32
        params = nnx_utils.state_map(
            params,
            config.trainable_filter,
            lambda p: _cast_param(p, weight_dtype),
        )
        params = nnx_utils.state_map(
            params,
            nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*"),
            lambda p: _cast_param(p, target_dtype),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=critic_tx,
            opt_state=critic_tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else jax.tree.map(jnp.copy, params),
        )

    policy_tx = None
    if config.policy is not None:
        policy_schedule = config.policy_lr_schedule if config.policy_lr_schedule is not None else config.lr_schedule
        policy_tx = _optimizer.create_optimizer(config.optimizer, policy_schedule, weight_decay_mask=None)

    def init_policy(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        if config.policy is None or policy_tx is None:
            raise ValueError("Config does not specify a policy")
        policy_model = config.policy.create(rng)

        if partial_params is not None:
            graphdef, state = nnx.split(policy_model)
            nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, partial_params)
            policy_model = nnx.merge(graphdef, state)

        policy_params = nnx.state(policy_model)
        policy_params = nnx_utils.state_map(
            policy_params,
            config.freeze_filter,
            lambda p: _cast_param(p, jnp.bfloat16),
        )
        return training_utils.TrainState(
            step=0,
            params=policy_params,
            model_def=nnx.graphdef(policy_model),
            tx=policy_tx,
            opt_state=policy_tx.init(policy_params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    def init_actor_critic(
        rng: at.KeyArrayLike,
        critic_partial_params: at.Params | None = None,
        policy_partial_params: at.Params | None = None,
    ) -> training_utils.ActorCriticTrainState:
        rng, critic_rng, policy_rng = jax.random.split(rng, 3)
        critic_state = init_critic(critic_rng, critic_partial_params)

        policy_state = None
        if config.policy is not None:
            policy_state = init_policy(policy_rng, policy_partial_params)

        return training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)

    train_state_shape = jax.eval_shape(init_actor_critic, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    critic_partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.critic.params.to_pure_dict()
    )
    policy_partial_params = None
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Pre-shard partial_params to match the FSDP output sharding so each chip
    # only receives its shard (~x/N GB) instead of the full replicated copy
    # (~x GB). This avoids OOM during init on memory-constrained devices.
    critic_params_sharding = sharding.fsdp_sharding(critic_partial_params, mesh)
    policy_params_sharding = None

    # `jax.device_put(tree, sharding)` calls `multihost_utils.assert_equal`
    # under the hood, which `broadcast_one_to_all`s each leaf with
    # `out_shardings=PartitionSpec()` (REPLICATED). For very large param trees
    # like Gemma 4 e2b (~27 GB total, largest leaf ~9 GB), this replicated
    # output overflows TPU HBM (16 GiB per chip on v5litepod) — the chip can't
    # hold a full unsharded copy of the largest leaf. PaliGemma (224x224)
    # works because its largest leaf is small.
    #
    # Workaround: each host already loaded the *same* deterministic .npz, so
    # the assert_equal step is redundant. Construct each leaf's sharded
    # `jax.Array` directly from local numpy via `make_array_from_callback` —
    # each chip materialises only its own shard, no global broadcast.
    def _make_sharded_array(arr, sharding):
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            return arr
        shape = arr.shape
        def _cb(idx):
            return np.asarray(arr[idx])
        return jax.make_array_from_callback(shape, sharding, _cb)
    critic_partial_params = jax.tree.map(_make_sharded_array, critic_partial_params, critic_params_sharding)

    donate_argnums = (1,) if policy_partial_params is None else (1, 2)
    in_shardings = (
        replicated_sharding,
        critic_params_sharding,
        policy_params_sharding,
    )

    train_state = jax.jit(
        init_actor_critic,
        donate_argnums = donate_argnums,
        in_shardings = in_shardings,
        out_shardings = state_sharding,
    )(init_rng, critic_partial_params, policy_partial_params)

    return train_state, state_sharding


@at.typecheck
def value_function_train_step(
    config: _config.TrainConfig,
    lr_schedule: optax.Schedule,
    state: training_utils.TrainState,
    policy_state: training_utils.TrainState | None,
    batch: dict[str, Any],
    rng: at.KeyArrayLike,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step for value function.

    Args:
        config: Training configuration.
        state: Current training state.
        batch: Batch of data.
        rng: Random key for algorithms that need stochasticity (e.g., SAC).

    Returns:
        Tuple of (new_state, info_dict).
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()
    policy = None if policy_state is None else nnx.merge(policy_state.model_def, policy_state.params)

    transition = _value_fn.Transition.from_batch(batch)

    def loss_fn(model: _value_fn.BaseValueFunction):
        # compute_loss returns (per_sample_loss, info_dict)
        per_sample_loss, value_info = model.compute_loss(transition, train=True, rng=rng, policy=policy)
        mean_loss = jnp.mean(per_sample_loss)
        return mean_loss, value_info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, value_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)

    # Call post_step_update for algorithms that need it (e.g., SAC target network update)
    model.post_step_update()

    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # The trailing-name regex uses `[/_]` so it also matches the stacked-layer-params
    # layout, where per-layer scales/biases live under names like
    # `g0__pre_attention_norm__scale` (separator `__` instead of `/`).
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*[/_](bias|scale|pos_embedding|input_embedding)")),
            nnx.Not(nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    target_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*"),
            nnx.Not(nnx_utils.PathRegex(".*[/_](bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Batch statistics
    obs_state = transition.observation.state
    action = transition.action
    # When subsample=True (Hdf5 with subsample) the chunk has H slots but only
    # the first H//2 are valid (fps prefix mask zeros the rest); restrict action
    # stats to the valid prefix so the padded slots don't pollute mean/std.
    if getattr(config.data, "subsample", False):
        action = action[:, : config.action_horizon // 2, :]
    mc_return_mask = transition.mc_return_mask
    valid_mc_return = (
        transition.mc_return
        if mc_return_mask is None
        else jnp.where(mc_return_mask, transition.mc_return, jnp.nan)
    )
    batch_stats = {
        # Observation stats
        "batch/obs_mean": jnp.mean(obs_state),
        "batch/obs_std": jnp.std(obs_state),
        "batch/obs_min": jnp.min(obs_state),
        "batch/obs_max": jnp.max(obs_state),
        "batch/obs_out_of_range_frac": jnp.mean((jnp.abs(obs_state) >= 1.001).astype(jnp.float32)),
        # Action stats
        "batch/action_mean": jnp.mean(action),
        "batch/action_std": jnp.std(action),
        "batch/action_min": jnp.min(action),
        "batch/action_max": jnp.max(action),
        "batch/action_out_of_range_frac": jnp.mean((jnp.abs(action) >= 1.001).astype(jnp.float32)),
        # Reward stats
        "batch/reward_mean": jnp.mean(transition.reward),
        "batch/reward_std": jnp.std(transition.reward),
        "batch/reward_min": jnp.min(transition.reward),
        "batch/reward_max": jnp.max(transition.reward),
        # MC return stats, over frames whose return is trustworthy (see Transition.mc_return_mask).
        "batch/mc_return_mean": jnp.nanmean(valid_mc_return),
        "batch/mc_return_std": jnp.nanstd(valid_mc_return),
        "batch/mc_return_min": jnp.nanmin(valid_mc_return),
        "batch/mc_return_max": jnp.nanmax(valid_mc_return),
        # Termination stats
        "batch/termination_mean": jnp.mean(transition.termination.astype(jnp.float32)),
        "batch/termination_std": jnp.std(transition.termination.astype(jnp.float32)),
        "batch/termination_min": jnp.min(transition.termination.astype(jnp.float32)),
        "batch/termination_max": jnp.max(transition.termination.astype(jnp.float32)),
        # Truncation stats
        "batch/truncation_mean": jnp.mean(transition.truncation.astype(jnp.float32)),
        "batch/truncation_std": jnp.std(transition.truncation.astype(jnp.float32)),
        "batch/truncation_min": jnp.min(transition.truncation.astype(jnp.float32)),
        "batch/truncation_max": jnp.max(transition.truncation.astype(jnp.float32)),
    }
    if transition.counterfactual_next_actions is not None:
        # Same subsample slice as `action` above: chunk has H slots but only the first
        # H//2 are real actions when subsample=True; the trailing slots are zero-pad
        # from _subsample_trajectory and are masked off at the critic forward.
        cf_next = transition.counterfactual_next_actions
        if getattr(config.data, "subsample", False):
            cf_next = cf_next[:, :, : config.action_horizon // 2, :]
        batch_stats["batch/counterfactual_next_actions_mean"] = jnp.mean(cf_next)
        batch_stats["batch/counterfactual_next_actions_std"] = jnp.std(cf_next)
        batch_stats["batch/counterfactual_next_actions_min"] = jnp.min(cf_next)
        batch_stats["batch/counterfactual_next_actions_max"] = jnp.max(cf_next)
        batch_stats["batch/counterfactual_next_actions_out_of_range_frac"] = jnp.mean(
            (jnp.abs(cf_next) >= 1.001).astype(jnp.float32)
        )

    batch_size = transition.reward.shape[0]

    value_stats = {}
    non_terminal = ~transition.termination
    num_non_terminal = jnp.maximum(jnp.sum(non_terminal.astype(jnp.float32)), 1.0)

    for key in ("predicted_value", "target_value", "td_error"):
        if key in value_info:
            arr = value_info.pop(key)
            value_stats[f"{key}_mean"] = jnp.sum(arr * non_terminal.astype(arr.dtype)) / num_non_terminal
            value_stats[f"{key}_std"] = jnp.std(arr)

    if "next_value" in value_info:
        next_val = value_info.pop("next_value")
        value_stats["next_value_mean"] = jnp.sum(next_val * non_terminal.astype(next_val.dtype)) / num_non_terminal

    if "mc_loss" in value_info:
        mc_loss_arr = value_info.pop("mc_loss")
        value_stats["mc_loss"] = _value_fn.masked_mean(mc_loss_arr, mc_return_mask)
    if mc_return_mask is not None:
        value_stats["mc_return_valid_frac"] = jnp.mean(mc_return_mask.astype(jnp.float32))

    if "next_token_loss" in value_info:
        next_token_loss_arr = value_info.pop("next_token_loss")
        value_stats["next_token_loss"] = jnp.mean(next_token_loss_arr)

    # The value error on its own, with the weighted next-token auxiliary term excluded.
    # The headline loss mixes the two, so it is not comparable between runs that set
    # different next_token_loss_weight -- which is exactly the comparison these runs exist
    # to make. Popped rather than left in place because it is per-sample, and anything
    # still in value_info downstream is cast with float().
    if "value_loss" in value_info:
        value_stats["value_loss"] = jnp.mean(value_info.pop("value_loss"))

    # Variable-horizon only: MC loss is inherently high at short sampled horizons, which
    # does not imply inaccurate values at the longer horizons we ultimately use. Log MC
    # loss restricted to samples whose sampled horizon k is at least half the action chunk
    # (fps-scaled: 25 @50fps, 15 @30fps), plus the number of such samples.
    if "q_pred_per_sample" in value_info:
        q_pred_per_sample = value_info.pop("q_pred_per_sample")
        mc_return_per_sample = value_info.pop("mc_return_per_sample")
        if "variable_k_native" in batch:
            per_sample_mc_loss = jnp.square(q_pred_per_sample - mc_return_per_sample)
            half_action_chunk = config.action_horizon // 2
            k_threshold = jnp.where(jnp.asarray(batch["fps"]) == 30, 3 * half_action_chunk // 5, half_action_chunk)
            long_horizon_mask = jnp.asarray(batch["variable_k_native"]) >= k_threshold
            if mc_return_mask is not None:
                long_horizon_mask = jnp.logical_and(long_horizon_mask, mc_return_mask)
            long_horizon_mask = long_horizon_mask.astype(jnp.float32)
            num_long_horizon = jnp.sum(long_horizon_mask)
            value_stats["mc_loss_long_horizon"] = (
                jnp.sum(per_sample_mc_loss * long_horizon_mask) / jnp.maximum(num_long_horizon, 1.0)
            )
            value_stats["num_long_horizon"] = num_long_horizon

    grads_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), grads)
    kernel_params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), kernel_params)
    target_params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), target_params)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads_f32),
        "param_norm": optax.global_norm(kernel_params_f32),
        "target_param_norm": optax.global_norm(target_params_f32),
        "learning_rate": lr_schedule(state.step),
        **value_info,
        **value_stats,
        **batch_stats,
    }

    if "steps_to_subtask_end" in batch:
        steps = jnp.asarray(batch["steps_to_subtask_end"]).astype(jnp.float32)
        info["batch/steps_to_subtask_end_mean"] = jnp.mean(steps)
        info["batch/steps_to_subtask_end_std"] = jnp.std(steps)
        info["batch/steps_to_subtask_end_min"] = jnp.min(steps)
        info["batch/steps_to_subtask_end_max"] = jnp.max(steps)
    return new_state, info


def get_trajectory_frames(dataset, episode_idx: int) -> list[dict]:
    """Extract all frames from a specific episode in the dataset.

    Supports both NumpyDataset and LeRobotDataset.
    """
    # Check if this is a NumpyDataset (has get_episode_frames method)
    if hasattr(dataset, "get_episode_frames"):
        return dataset.get_episode_frames(episode_idx)

    # LeRobotDataset path: use episode_data_index
    episode_data_index = dataset.episode_data_index
    start_idx = episode_data_index["from"][episode_idx].item()
    end_idx = episode_data_index["to"][episode_idx].item()

    frames = []
    for idx in range(start_idx, end_idx):
        frame = dataset[idx]
        frames.append(frame)
    return frames


def _create_value_video(
    mc_returns: list,
    predicted_values: list,
    ep_idx: int,
    step: int,
    suffix: str,
    oracle_values: list | None,
    frame_images: list[np.ndarray],
    fps: int,
    subtask_texts: list[str] | None = None,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Video | str":
    """Create a 2x2 layout video: left wrist (top-left), right wrist (bottom-left),
    value plot (top-right), base camera (bottom-right). Subtask list shown below the plot.

    When output_dir is set, saves an MP4 to disk via imageio and returns the file path.
    When output_dir is None, returns a wandb.Video (GIF)."""
    T = len(mc_returns)
    timesteps = np.arange(T)
    video_frames = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_left_wrist, ax_val = axes[0, 0], axes[0, 1]
    ax_right_wrist, ax_base = axes[1, 0], axes[1, 1]

    # Pre-compute subtask caption and adjust layout once
    subtask_caption = None
    if subtask_texts:
        numbered_subtasks = [f"{i+1}. {text}" for i, text in enumerate(subtask_texts)]
        lines = []
        for i in range(0, len(numbered_subtasks), 3):
            lines.append("  ".join(numbered_subtasks[i:i+3]))
        subtask_caption = "Subtasks:\n" + "\n".join(lines)
        num_lines = len(lines) + 1
        bottom_margin = 0.10 + 0.03 * num_lines
        fig.subplots_adjust(bottom = bottom_margin)

    for t in range(T):
        for ax in axes.flat:
            ax.cla()

        ax_left_wrist.imshow(frame_images[t][0])
        ax_left_wrist.axis("off")
        ax_left_wrist.set_title("Left Wrist", fontsize=12)

        ax_right_wrist.imshow(frame_images[t][1])
        ax_right_wrist.axis("off")
        ax_right_wrist.set_title("Right Wrist", fontsize=12)

        ax_base.imshow(frame_images[t][2])
        ax_base.axis("off")
        ax_base.set_title(f"Timestep {t}", fontsize=12)

        ax_val.plot(timesteps, mc_returns, label="MC Returns", color="blue", linewidth=2)
        ax_val.plot(timesteps, predicted_values, label="Predicted Value", color="orange", linewidth=2, linestyle="--")
        if oracle_values is not None:
            ax_val.plot(timesteps, oracle_values, label="Oracle Q-Value", color="green", linewidth=2, linestyle="-.", alpha=0.7)
        ax_val.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_val.set_xlabel("Timestep", fontsize=12)
        ax_val.set_ylabel("Value", fontsize=12)
        ax_val.set_title(f"Episode {ep_idx} - Step {step}{suffix}", fontsize=14)
        ax_val.legend(fontsize=11)
        ax_val.grid(visible=True, alpha=0.3)

        if subtask_caption is None:
            plt.tight_layout()
        else:
            for txt in fig.texts:
                txt.remove()
            fig.text(0.5, 0.01, subtask_caption, ha="center", va="bottom", fontsize=9,
                     family="monospace", linespacing=1.5)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        video_frames.append(buf.copy())

    plt.close(fig)

    if output_dir is not None:
        import imageio

        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.mp4")
        # Ensure dimensions are divisible by 16 (imageio macro_block_size)
        h, w = video_frames[0].shape[:2]
        h_crop = h - (h % 16)
        w_crop = w - (w % 16)
        cropped_frames = [frame[:h_crop, :w_crop] for frame in video_frames]
        imageio.mimsave(out_path, cropped_frames, format = "mp4", fps = fps, codec = "libx264", quality = 8)
        logging.info(f"Saved video to {out_path}")
        return out_path

    video_array = np.stack(video_frames).transpose(0, 3, 1, 2)
    # GIF is used because wandb renders it inline. mp4 shows as "File type unknown" in the wandb UI.
    # GIF's 256-color palette quantization causes a visible quality drop (color jitter appearance),
    # but the video remains interpretable.
    return wandb.Video(video_array, fps = fps, format = "gif")


def _create_value_plot(
    mc_returns,
    predicted_values,
    ep_idx,
    step,
    suffix,
    oracle_values=None,
    subtask_texts: list[str] | None = None,
    plot_video: bool = False,
    frame_images: list[np.ndarray] | None = None,
    fps: int = 10,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Image | wandb.Video | str":
    """Create a matplotlib plot comparing MC returns vs predicted values.

    Args:
        mc_returns: List of MC return values.
        predicted_values: List of predicted values.
        ep_idx: Episode index (from dataset's episode_index, used as title).
        step: Training step.
        suffix: Suffix for the title.
        oracle_values: Optional list of oracle Q-values.
        subtask_texts: Optional ordered list of subtask segment texts for caption.
        plot_video: If True and frame_images are provided, returns a wandb.Video instead of wandb.Image.
        frame_images: Optional list of (3, 224, 224, 3) uint8 arrays (left wrist, right wrist, base_0) per timestep.
        fps: Frame rate of the episode, used when encoding the output video.
        output_dir: If set, save to disk instead of returning a wandb object.
        plot_key: Key used to derive the filename when saving to disk.
    """
    if plot_video and frame_images is not None and len(frame_images) == len(mc_returns):
        return _create_value_video(
            mc_returns, predicted_values, ep_idx, step, suffix, oracle_values, frame_images, fps, subtask_texts,
            output_dir = output_dir, plot_key = plot_key,
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    timesteps = np.arange(len(mc_returns))

    ax.plot(timesteps, mc_returns, label="MC Returns", color="blue", linewidth=2)
    ax.plot(
        timesteps,
        predicted_values,
        label="Predicted Value",
        color="orange",
        linewidth=2,
        linestyle="--",
    )

    if oracle_values is not None:
        ax.plot(
            timesteps,
            oracle_values,
            label="Oracle Q-Value",
            color="green",
            linewidth=2,
            linestyle="-.",
            alpha=0.7,
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Episode {ep_idx} - Step {step}{suffix}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(visible=True, alpha=0.3)

    # Add numbered subtask list below x-axis label if provided
    if subtask_texts:
        # Number the subtask segments and group 3 per line
        numbered_subtasks = [f"{i+1}. {text}" for i, text in enumerate(subtask_texts)]
        lines = []
        for i in range(0, len(numbered_subtasks), 3):
            line = "  ".join(numbered_subtasks[i:i+3])
            lines.append(line)
        caption = "Subtasks:\n" + "\n".join(lines)

        # Calculate bottom margin based on number of lines
        num_lines = len(lines) + 1  # +1 for "Subtasks:" header
        bottom_margin = 0.10 + 0.03 * num_lines
        plt.subplots_adjust(bottom=bottom_margin)

        # Place text below the x-axis label
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9,
                 family="monospace", linespacing=1.5)
    else:
        plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.png")
        fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
        logging.info(f"Saved plot to {out_path}")
        return out_path

    img = wandb.Image(fig)
    plt.close(fig)
    return img


_render_thread: threading.Thread | None = None


def attn_modality_labels(num_modalities: int, *, action_conditioned: bool) -> list[str]:
    """Names for the per-modality CLS attention groups, in the network's grouping order.

    Mirrors ``_group_attn_scores``: three cameras, the text prompt, the state token when
    the network embeds one, and the action chunk for a Q-function. The state slot is
    inferred from the count because the plotting side does not see the network config.
    """
    labels = [f"img{i + 1}" for i in range(3)] + ["text"]
    if num_modalities > 4 + int(action_conditioned):
        labels.append("state")
    if action_conditioned:
        labels.append("action")
    if len(labels) != num_modalities:
        raise ValueError(f"Cannot name {num_modalities} attention groups (action_conditioned={action_conditioned})")
    return labels


def _create_attn_plot(
    attn_scores: list[np.ndarray],
    ep_idx: int,
    step: int,
    repo_id: str,
    action_conditioned: bool,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Image | str":
    """Create a line plot of per-modality CLS attention scores over time."""
    scores = np.stack(attn_scores, axis=0)  # [T, n_modalities]
    timesteps = np.arange(len(scores))
    labels = attn_modality_labels(scores.shape[1], action_conditioned = action_conditioned)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        ax.plot(timesteps, scores[:, i], label=label)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Mean Attention Scores", fontsize=12)
    ax.set_title(f"Episode {ep_idx} - Step {step} - CLS Attention ({repo_id})", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"attn_ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.png")
        fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
        logging.info(f"Saved attention plot to {out_path}")
        return out_path

    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _render_and_log_plots(
    all_predictions: dict,
    all_predictions_neg: dict,
    all_predictions_random: dict,
    all_predictions_shuffled: dict,
    all_attn_scores: dict,
    ep_mc_returns: dict,
    ep_frame_images: dict,
    ep_fps: dict,
    ep_include_masks: dict,
    ep_subtasks: dict,
    ep_negative_subtasks: dict,
    traj_to_repo_ep: dict,
    action_conditioned: bool,
    step: int,
    *,
    all_predictions_counterfactual: dict | None = None,
    output_dir: str | None = None,
) -> None:
    images = {}
    for traj_idx in ep_mc_returns:
        repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]
        plot_key = f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}"
        mc_returns = ep_mc_returns[traj_idx]
        include_masks = ep_include_masks[traj_idx]
        predicted_values = all_predictions[traj_idx]

        if len(predicted_values) != len(mc_returns):
            logging.warning(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): mismatch between predictions ({len(predicted_values)}) and mc_returns ({len(mc_returns)})")
            continue

        filtered_mc_returns = []
        filtered_predictions = []
        for mc, pred, mask in zip(mc_returns, predicted_values, include_masks):
            if mask:
                filtered_mc_returns.append(mc)
                filtered_predictions.append(pred)

        if len(filtered_mc_returns) == 0:
            logging.warning(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): no frames with include_subtask=True")
            continue

        subtasks = ep_subtasks.get(traj_idx, [])
        logging.info(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): {len(filtered_predictions)}/{len(predicted_values)} frames after include_subtask filter, subtasks={subtasks}")

        ep_frame_images[traj_idx] = [img for img, mask in zip(ep_frame_images[traj_idx], include_masks) if mask]

        images[plot_key] = _create_value_plot(
            filtered_mc_returns, filtered_predictions, ep_idx, step, " (RoboCOIN)",
            oracle_values = None, subtask_texts = subtasks,
            plot_video = True, frame_images = ep_frame_images[traj_idx],
            fps = ep_fps[traj_idx],
            output_dir = output_dir, plot_key = plot_key,
        )
        logging.info(f"Repo {repo_id}, episode {ep_idx} plot created")

        attn_scores = all_attn_scores.get(traj_idx, [])
        filtered_attn = [s for s, mask in zip(attn_scores, include_masks) if mask]
        if len(filtered_attn) == len(filtered_mc_returns) and len(filtered_attn) > 0:
            images[f"{plot_key}_attn"] = _create_attn_plot(
                filtered_attn, ep_idx, step, repo_id.removeprefix("RoboCOIN/"), action_conditioned,
                output_dir = output_dir, plot_key = f"{plot_key}_attn",
            )

        negative_subtasks = ep_negative_subtasks.get(traj_idx)
        predicted_values_neg = all_predictions_neg.get(traj_idx, [])
        if negative_subtasks and len(predicted_values_neg) == len(predicted_values):
            filtered_predictions_neg = [pred for pred, mask in zip(predicted_values_neg, include_masks) if mask]
            if len(filtered_predictions_neg) == len(filtered_mc_returns) and len(filtered_predictions_neg) > 0:
                images[f"{plot_key}_counterfactual_text"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_neg, ep_idx, step, " (Counterfactual Text)",
                    oracle_values = None, subtask_texts = negative_subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_counterfactual_text",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} counterfactual text plot created")

        predicted_values_random = all_predictions_random.get(traj_idx, [])
        if len(predicted_values_random) == len(predicted_values):
            filtered_predictions_random = [pred for pred, mask in zip(predicted_values_random, include_masks) if mask]
            if len(filtered_predictions_random) == len(filtered_mc_returns) and len(filtered_predictions_random) > 0:
                images[f"{plot_key}_random_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_random, ep_idx, step, " (Random Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_random_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} random action plot created")

        predicted_values_shuffled = all_predictions_shuffled.get(traj_idx, [])
        if len(predicted_values_shuffled) == len(predicted_values):
            filtered_predictions_shuffled = [pred for pred, mask in zip(predicted_values_shuffled, include_masks) if mask]
            if len(filtered_predictions_shuffled) == len(filtered_mc_returns) and len(filtered_predictions_shuffled) > 0:
                images[f"{plot_key}_shuffled_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_shuffled, ep_idx, step, " (Shuffled Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_shuffled_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} shuffled action plot created")

        predicted_values_counterfactual = all_predictions_counterfactual.get(traj_idx, []) if all_predictions_counterfactual is not None else []
        if len(predicted_values_counterfactual) == len(predicted_values):
            filtered_predictions_counterfactual = [
                pred for pred, mask in zip(predicted_values_counterfactual, include_masks) if mask
            ]
            if len(filtered_predictions_counterfactual) == len(filtered_mc_returns) and len(filtered_predictions_counterfactual) > 0:
                images[f"{plot_key}_counterfactual_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_counterfactual, ep_idx, step, " (Counterfactual Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_counterfactual_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} counterfactual action plot created")

    if output_dir is None and images:
        # Log without an explicit step, avoiding the "step must be monotonically
        # increasing" warning that occurs because this thread may run after further
        # training steps have been logged.
        wandb.log(images)
    logging.info(f"Render thread finished: {'saved' if output_dir else 'logged'} {len(images)} plots for step {step}")
    del images, ep_frame_images, all_predictions, all_predictions_neg, all_predictions_random, all_predictions_shuffled, all_predictions_counterfactual, all_attn_scores


def _render_snapshot_plots(
    frames: list[dict],
    predicted_values: list[float],
    snapshot: SnapshotConfig,
    episode_name: str,
    output_dir: str | None = None,
) -> dict:
    """Render the extra per-interval value snapshots for one validation episode.

    Reuses the already-decoded ``frames`` and the already-computed
    ``predicted_values`` (one per frame, in order) — no re-loading or
    re-prediction. For each interval ``[a, b]`` (inclusive) in
    ``snapshot.shade_intervals`` the interval's subtask is read from
    ``frames[i]["subtask_1_text"]`` and asserted constant across the interval
    (no subtask-boundary crossing). Intervals are grouped by subtask:

    - one value plot per subtask ``snapshot/<episode_name>_<subtask>``: the
      predicted-value blue line over that subtask's frame span, each interval
      shaded light red/green;
    - one camera image per interval
      ``snapshot/<episode_name>_<subtask>_<camera>_f<midpoint>``.

    Returns a dict of plot_key -> wandb object (output_dir None) or saved path.
    """
    import re

    # Low-alpha fills keep the blue value line clearly visible while red vs green
    # stay easily distinguishable.
    shade_style = {"r": ((1.0, 0.0, 0.0), 0.18), "g": ((0.0, 0.7, 0.0), 0.18)}

    def _sanitize(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")

    per_frame_subtask = [decode_text(f["subtask_1_text"]) for f in frames]

    # Resolve each interval's subtask, asserting it does not cross a boundary.
    subtask_of_interval: list[str] = []
    for interval_start, interval_end in snapshot.shade_intervals:
        interval_subtasks = set(per_frame_subtask[interval_start : interval_end + 1])
        if len(interval_subtasks) != 1:
            raise ValueError(
                f"Snapshot interval [{interval_start}, {interval_end}] crosses a subtask "
                f"boundary: {sorted(interval_subtasks)}."
            )
        subtask_of_interval.append(next(iter(interval_subtasks)))

    images: dict = {}

    # One value plot per distinct subtask that has intervals.
    for subtask_text in dict.fromkeys(subtask_of_interval):
        member_indices = [i for i, st in enumerate(subtask_of_interval) if st == subtask_text]
        frame_indices = [i for i, st in enumerate(per_frame_subtask) if st == subtask_text]
        frame_start = min(frame_indices)
        frame_end = max(frame_indices)
        timesteps = np.arange(frame_start, frame_end + 1)
        values = predicted_values[frame_start : frame_end + 1]

        fig, ax = plt.subplots(figsize = (10, 6))
        ax.plot(timesteps, values, label = "Predicted Value", color = "blue", linewidth = 2)
        for i in member_indices:
            interval_start, interval_end = snapshot.shade_intervals[i]
            facecolor, alpha = shade_style[snapshot.shade_colours[i]]
            ax.axvspan(interval_start, interval_end, facecolor = facecolor, alpha = alpha)
        ax.set_xlabel("Timestep", fontsize = 12)
        ax.set_ylabel("Value", fontsize = 12)
        ax.set_title(subtask_text, fontsize = 13)
        ax.legend(fontsize = 11)
        ax.grid(visible = True, alpha = 0.3)
        plt.tight_layout()

        plot_key = f"snapshot/{episode_name}_{_sanitize(subtask_text)}"
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok = True)
            out_path = os.path.join(output_dir, f"{plot_key.replace('/', '_')}.png")
            fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
            plt.close(fig)
            logging.info(f"Saved snapshot plot to {out_path}")
            images[plot_key] = out_path
        else:
            images[plot_key] = wandb.Image(fig)
            plt.close(fig)

    # One camera image per interval, captured at the interval midpoint.
    for i, (interval_start, interval_end) in enumerate(snapshot.shade_intervals):
        midpoint = (interval_start + interval_end) // 2
        camera = snapshot.snapshot_camera[i]
        image = np.asarray(frames[midpoint]["image"][camera])
        img_key = f"snapshot/{episode_name}_{_sanitize(subtask_of_interval[i])}_{camera}_f{midpoint}"
        if output_dir is not None:
            import imageio

            os.makedirs(output_dir, exist_ok = True)
            out_path = os.path.join(output_dir, f"{img_key.replace('/', '_')}.png")
            imageio.imwrite(out_path, image)
            logging.info(f"Saved snapshot image to {out_path}")
            images[img_key] = out_path
        else:
            images[img_key] = wandb.Image(image)

    return images


def generate_validation_plots_rlds(
    model: _value_fn.BaseValueFunction,
    val_episode_indices: list[int],
    step: int,
    *,
    action_conditioned: bool,
    data_config: _config.DataConfig,
    cache_dir: str,
    output_dir: str | None = None,
    batch_size: int = 64,
    override_prompt: tuple[np.ndarray, np.ndarray] | None = None,
    snapshot: SnapshotConfig | None = None,
) -> dict:
    """Generate validation plots from cached RLDS validation episodes.

    Loads pre-cached validation episodes from ``cache_dir`` (populated at
    init time via ``cache_val_episodes``), runs value-function inference,
    and logs plots to W&B (async) or saves to ``output_dir`` on disk.

    Args:
        model: The value function model.
        val_episode_indices: List of episode indices to plot (used for count only).
        step: Current training step.
        action_conditioned: Whether the model is action-conditioned (Q vs V).
        data_config: Data configuration.
        cache_dir: Directory containing cached validation episodes.
        output_dir: If set, save plots/videos to this directory instead of logging to wandb.

    Returns:
        Empty dict (plots are logged/saved asynchronously by a background thread).
    """
    # Per-episode load → decode + resize → predict → free pipeline so peak host
    # RAM is bounded to one episode's decoded frames, not the entire val cache.
    # Cache pkls hold raw image bytes (decode_images=False at the dataset level) and
    # `decode_episode_images` materializes them lazily.
    image_size = data_config.rlds_kwargs.get("image_size")
    assert image_size is not None, (
        "image_size missing from data_config.rlds_kwargs — required to decode/resize "
        "compressed val cache images at predict time."
    )

    cache_files: list[tuple[str, str]] = []
    for filename in sorted(os.listdir(cache_dir)):
        if filename.endswith(".pkl"):
            traj_idx = filename[: -len(".pkl")]
            cache_files.append((traj_idx, os.path.join(cache_dir, filename)))
    logging.info(f"Found {len(cache_files)} cached validation episodes in {cache_dir}")

    MAX_SUBTASK_SEGMENTS = 16

    # Accumulators populated one episode at a time. Heavy per-episode arrays
    # (decoded images, frame dicts) are dropped after predict; we keep only
    # what the renderer needs.
    traj_to_repo_ep: dict[str, tuple[str, int, str]] = {}
    ep_subtasks: dict[str, list[str]] = {}
    ep_mc_returns: dict[str, list] = {}
    ep_frame_images: dict[str, list[np.ndarray]] = {}
    ep_fps: dict[str, int] = {}
    ep_include_masks: dict[str, list[bool]] = {}
    ep_negative_subtasks: dict[str, list[str]] = {}
    all_predictions: dict[str, list[float]] = {}
    all_predictions_neg: dict[str, list[float]] = {}
    all_predictions_random: dict[str, list[float]] = {}
    all_predictions_counterfactual: dict[str, list[float]] = {}
    all_predictions_shuffled: dict[str, list[float]] = {}
    all_attn_scores: dict[str, list[np.ndarray]] = {}

    for traj_idx, cache_file in cache_files:
        with open(cache_file, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            logging.warning(f"Traj {traj_idx} cache empty, skipping")
            continue

        if override_prompt is not None:
            apply_override_prompt(frames, override_prompt)

        decode_episode_images(frames, image_size)

        repo_id_raw = frames[0]["repo_id"]
        if isinstance(repo_id_raw, np.ndarray):
            repo_id_raw = repo_id_raw.item()
        if isinstance(repo_id_raw, bytes):
            repo_id_raw = repo_id_raw.decode("utf-8")
        ep_idx_raw = frames[0]["episode_index"]
        if isinstance(ep_idx_raw, np.ndarray):
            ep_idx_raw = ep_idx_raw.item()
        ep_idx = int(ep_idx_raw)

        n_segments, split_frame_idx, segments = count_subtask_segments(frames)
        if n_segments > MAX_SUBTASK_SEGMENTS:
            split_seg_idx = n_segments // 2
            segment_specs = [
                (f"{traj_idx}_p0", "_part0", frames[:split_frame_idx], segments[:split_seg_idx]),
                (f"{traj_idx}_p1", "_part1", frames[split_frame_idx:], segments[split_seg_idx:]),
            ]
            logging.info(
                f"Traj {traj_idx} (repo {repo_id_raw}, episode {ep_idx}): "
                f"{n_segments} subtask segments > {MAX_SUBTASK_SEGMENTS}, splitting at "
                f"segment {split_seg_idx}, frame {split_frame_idx} "
                f"({split_frame_idx} + {len(frames) - split_frame_idx} frames)"
            )
        else:
            segment_specs = [(str(traj_idx), "", frames, segments)]

        for seg_key, part_suffix, seg_frames, seg_subtasks in segment_specs:
            if len(seg_frames) == 0:
                continue

            inject_shuffled_actions(seg_frames, action_conditioned = action_conditioned)

            traj_to_repo_ep[seg_key] = (repo_id_raw, ep_idx, part_suffix)
            ep_subtasks[seg_key] = seg_subtasks
            ep_mc_returns[seg_key] = [f["mc_return"] for f in seg_frames]
            ep_frame_images[seg_key] = [
                np.stack([
                    np.asarray(f["image"]["left_wrist_0_rgb"]),
                    np.asarray(f["image"]["right_wrist_0_rgb"]),
                    np.asarray(f["image"]["base_0_rgb"]),
                ])
                for f in seg_frames
            ]
            ep_fps[seg_key] = int(seg_frames[0]["fps"])
            ep_include_masks[seg_key] = [bool(f.get("include_subtask", True)) for f in seg_frames]
            # Negative-text variant is RoboCOIN-only; skip when the cached frames
            # don't carry "negative_subtask_1_text".
            if seg_frames and "negative_subtask_1_text" in seg_frames[0]:
                _, _, negative_segments = count_subtask_segments(seg_frames, prefix = "negative_")
                ep_negative_subtasks[seg_key] = negative_segments
            else:
                ep_negative_subtasks[seg_key] = []

            seg_all_frames = [(seg_key, idx, frame) for idx, frame in enumerate(seg_frames)]
            seg_ep_mc_returns = {seg_key: ep_mc_returns[seg_key]}
            logging.info(
                f"Traj {seg_key} (repo {repo_id_raw}, episode {ep_idx}{part_suffix}): "
                f"running predictions on {len(seg_all_frames)} frames"
            )
            preds, preds_neg, preds_random, preds_cf, preds_shuffled, attn, _ = predict_values(
                model, seg_all_frames, seg_ep_mc_returns, action_conditioned,
                batch_size = batch_size,
            )
            all_predictions[seg_key] = preds[seg_key]
            all_predictions_neg[seg_key] = preds_neg[seg_key]
            all_predictions_random[seg_key] = preds_random[seg_key]
            all_predictions_counterfactual[seg_key] = preds_cf[seg_key]
            all_predictions_shuffled[seg_key] = preds_shuffled[seg_key]
            all_attn_scores[seg_key] = attn[seg_key]

            del seg_all_frames

        # Extra snapshot plots reuse this episode's already-decoded frames and
        # already-computed predictions (no re-load / re-predict). Rank-0 only:
        # pure host-side matplotlib over replicated arrays, no SPMD collectives.
        if (
            snapshot is not None
            and jax.process_index() == 0
            and traj_idx == snapshot.episode_file.removesuffix(".pkl")
        ):
            # segment_specs partitions `frames` in order, so concatenating the
            # per-segment predictions realigns them to absolute frame indices.
            full_predictions: list[float] = []
            for snapshot_seg_key, _, _, _ in segment_specs:
                full_predictions.extend(all_predictions[snapshot_seg_key])
            snapshot_images = _render_snapshot_plots(
                frames, full_predictions, snapshot, episode_name = traj_idx, output_dir = output_dir,
            )
            if output_dir is None and snapshot_images:
                wandb.log(snapshot_images)
            logging.info(f"Rendered {len(snapshot_images)} snapshot plots for episode {traj_idx}")

        # Drop the decoded frames for this trajectory before loading the next.
        del frames
        gc.collect()

    if not all_predictions:
        logging.warning("No valid frames found across all episodes")
        return {}

    logging.info(
        f"Computed predictions for {len(all_predictions)} trajectory segments"
    )

    if jax.process_index() == 0:
        start_render_thread(
            all_predictions, all_predictions_neg, all_predictions_random,
            all_predictions_shuffled, all_attn_scores,
            ep_mc_returns, ep_frame_images, ep_fps, ep_include_masks,
            ep_subtasks, ep_negative_subtasks,
            traj_to_repo_ep, action_conditioned, step,
            all_predictions_counterfactual = all_predictions_counterfactual,
            output_dir = output_dir,
        )

    return {}


def start_render_thread(*args, **kwargs) -> None:
    """Render the standard validation plots on a background thread.

    Takes ``_render_and_log_plots``'s arguments. One render runs at a time: a still-running
    previous render is joined first. Callers must join ``_render_thread`` before exiting.
    """
    global _render_thread
    if _render_thread is not None:
        if _render_thread.is_alive():
            logging.warning("Previous render thread still running, waiting for it to finish...")
        _render_thread.join()
        _render_thread = None
    _render_thread = threading.Thread(target = _render_and_log_plots, args = args, kwargs = kwargs, daemon = True)
    _render_thread.start()


def main(config: _config.TrainConfig):
    """Train a value function."""
    init_logging()
    logger = logging.getLogger(__name__)

    # Initialize distributed training for TPU pods
    # Set PLATFORM=tpu environment variable to enable
    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        logger.info("Calling jax.distributed.initialize()")
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    logger.info(f"Running on: {_platform.node()}, platform: {platform}")

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(
            f"train_value_function.py requires a value function config, got {type(config.model).__name__}. "
            "Use a critic config such as 'robocoin_bimanual_paligemma_cql_rlds_subtask_ar'."
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # The CLI already handed us the FineTuneConfig, with any --fine-tune.* overrides applied.
    ft_config = config.fine_tune

    if ft_config is not None:
        ft_config = dataclasses.replace(ft_config, overwrite = config.overwrite, resume = config.resume)
        config = ft_config.apply_overrides(config)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        ft_config=ft_config,
        start_new=config.wandb_new,
    )
    logging.info(f"Initialized checkpoint manager with resuming={resuming}, config.resume={config.resume}")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)

    raw_batch = next(data_iter)
    if isinstance(raw_batch, tuple):
        obs, _ = raw_batch
        batch = {"state": obs.state}
    else:
        batch = raw_batch

    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(raw_batch)}")

    # Create validation data_config (same overridden config.data)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Initialize variables for all branches
    num_episodes = None
    val_dataloader = None
    val_episode_indices: list[int] = []
    val_episodes_cache_dir = None

    if data_config.rlds_dataset_class in ("robocoin", "hdf5", "lerobot"):
        action_horizon = config.action_horizon or config.model.action_horizon
        val_tokenizer = config.data._get_critic_tokenizer(config.model)
        # Build a val-only data_config that (a) optionally points at val_dataset_dir
        # (a smaller variant of the same dataset) and (b) leaves images compressed.
        # `decode_images=False` keeps cam_X as the raw JPEG/PNG bytes coming out
        # of the TFRecord rather than decoded uint8 arrays, so the trajectory
        # iterator does not blow up host RAM during caching. The downstream
        # consumer (utils.cache_val_episodes → utils.load_episode_for_predict)
        # decodes + resizes lazily, one episode at a time, at prediction time.
        # These overrides only affect val trajectory caching — the training
        # data_config and loader are untouched.
        import dataclasses as _dc
        _val_rlds_kwargs = {**data_config.rlds_kwargs, "decode_images": False}
        _val_data_config = _dc.replace(
            data_config,
            rlds_data_dir = data_config.val_dataset_dir or data_config.rlds_data_dir,
            rlds_kwargs = _val_rlds_kwargs,
        )
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            _val_data_config,
            action_horizon,
            config.batch_size,
            split = data_config.val_split,
            shuffle = False,
            return_trajectories = True,
        )
        val_input_transform = _transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
            *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
            *data_config.model_transforms.inputs,
            _config.AddValidationVariants(
                val_tokenizer,
                use_quantile_norm = data_config.use_quantile_norm,
            ),
        ])
        val_episode_indices = list(range(config.num_val_trajectories))
        # Cached validation episodes live on the local filesystem; NFS or another shared
        # path can be given with --validation-cache-dir for multi-host runs.
        val_episodes_cache_dir = config.validation_cache_dir or os.path.join(
            os.path.expanduser("~/.cache/openpi/val_episodes"),
            config.name,
            ft_config.name if ft_config is not None else "base",
        )
        val_dataset = None

        # All workers participate in caching: worker 0 covers all repos up to
        # num_val_trajectories; non-zero workers only cache include_repos. Each
        # worker checks file existence before claiming/writing, so concurrent
        # writes to NFS are safe.
        allow_duplicate_repos = ft_config is not None
        cache_val_episodes(
            val_trajectory_dataset, config.num_val_trajectories, val_episodes_cache_dir,
            include_repos = config.include_repos, save_only = True,
            input_transform = val_input_transform,
            allow_duplicate_repos = allow_duplicate_repos,
        )
        if jax.process_count() > 1:
            jax.experimental.multihost_utils.sync_global_devices("val_cache_write")
        del val_trajectory_dataset, val_input_transform, _val_data_config, _val_rlds_kwargs
    else:
        raise ValueError(f"Unsupported rlds_dataset_class {data_config.rlds_dataset_class!r} for validation.")

    if val_dataset is not None:
        num_episodes = val_dataset.num_episodes
        val_rng = np.random.default_rng(config.seed)

        # Filter to episodes with at least 10 frames for meaningful validation plots
        min_episode_length = 10
        if hasattr(val_dataset, "episode_starts") and hasattr(val_dataset, "episode_ends"):
            episode_lengths = val_dataset.episode_ends - val_dataset.episode_starts
            valid_episode_indices = np.where(episode_lengths >= min_episode_length)[0]
            if len(valid_episode_indices) < config.num_val_trajectories:
                logging.warning(
                    f"Only {len(valid_episode_indices)} episodes with >= {min_episode_length} frames, "
                    f"using all of them for validation"
                )
                val_episode_indices = valid_episode_indices.tolist()
            else:
                val_episode_indices = val_rng.choice(
                    valid_episode_indices, size=config.num_val_trajectories, replace=False
                ).tolist()
        else:
            val_episode_indices = val_rng.choice(
                num_episodes, size=min(config.num_val_trajectories, num_episodes), replace=False
            ).tolist()
        logging.info(f"Selected validation episodes: {val_episode_indices}")

        val_episodes_cache_dir = None

    action_conditioned = config.action_horizon is not None
    logging.info(f"Validation plots: action_conditioned={action_conditioned}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if resuming:
        logging.info("Resuming training from checkpoint")
        # train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        train_state = _load_model_utils.restore_state_with_shardings(
            checkpoint_manager, train_state, train_state_sharding,
        )

    # Unpack state and sharding
    if not isinstance(train_state, training_utils.ActorCriticTrainState):
        raise TypeError(f"Expected ActorCriticTrainState, got {type(train_state)}")

    critic_state = train_state.critic
    policy_state = train_state.policy
    critic_sharding = train_state_sharding.critic
    policy_sharding = train_state_sharding.policy
    # Release the outer wrapper: it aliases critic_state's param/optimizer
    # buffers. Holding it resident leaks the pre-fine-tune optimizer state
    # and blocks ptrain_step's buffer donation.
    del train_state, train_state_sharding
    logging.info(f"Initialized combined state:\nCritic: {training_utils.array_tree_to_info(critic_state.params)}")
    if policy_state:
        logging.info(f"Policy: {training_utils.array_tree_to_info(policy_state.params)}")

    jax.block_until_ready(critic_state)

    # === val_only mode: run one validation pass and exit ===
    if ft_config is not None and ft_config.val_only:
        logging.info("val_only mode: running validation plotting")
        step = int(critic_state.step)
        model = nnx.merge(critic_state.model_def, critic_state.params)

        if data_config.rlds_dataset_class in ("robocoin", "hdf5", "lerobot"):
            generate_validation_plots_rlds(
                model = model,
                val_episode_indices = val_episode_indices,
                step = step,
                action_conditioned = action_conditioned,
                data_config = data_config,
                cache_dir = val_episodes_cache_dir,
                batch_size = 32,
            )

        # Wait for background render thread if any
        if jax.process_index() == 0 and _render_thread is not None and _render_thread.is_alive():
            logging.info("Waiting for render thread to finish")
            _render_thread.join()

        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("val_only_done")

        logging.info("val_only mode complete")
        return

    # Determine effective training parameters based on fine-tune config
    pretrained_step = int(critic_state.step)
    is_fine_tuning = ft_config is not None and not ft_config.val_only

    if is_fine_tuning:
        config, critic_state, critic_sharding, checkpoint_manager, ft_resuming = ft_config.initialize(
            config, pretrained_step, critic_state, mesh,
        )
        if ft_resuming:
            # The FT save (the ActorCriticTrainState(...) call inside the training loop) wraps
            # critic + policy as AC. Mirror that wrapping for the restore so the tree
            # structures match.
            ac_state = training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)
            ac_sharding = training_utils.ActorCriticTrainState(critic=critic_sharding, policy=policy_sharding)
            ac_state = _load_model_utils.restore_state_with_shardings(
                checkpoint_manager, ac_state, ac_sharding,
            )
            critic_state = ac_state.critic
            policy_state = ac_state.policy
            logging.info("Resuming fine-tuning from FT checkpoint")

    lr_schedule = config.lr_schedule.create()

    if policy_state is not None:
        ptrain_step = jax.jit(
            functools.partial(value_function_train_step, config, lr_schedule),
            in_shardings=(critic_sharding, policy_sharding, data_sharding, replicated_sharding),
            out_shardings=(critic_sharding, replicated_sharding),
            donate_argnums=(0,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(value_function_train_step, config, lr_schedule),
            in_shardings=(critic_sharding, None, data_sharding, replicated_sharding),
            out_shardings=(critic_sharding, replicated_sharding),
            donate_argnums=(0,),
        )

    start_step = int(critic_state.step)
    # Fine-tune mode loads pretrained weights from a different dataset, so
    # advancing the new data iterator by ``start_step`` batches has no resume
    # semantics — skip the fast-forward and iterate only the fine-tune range.
    #
    # TEMPORARY: unconditionally skip the fast-forward for pre-training resumes too.
    # Every preemption of a preemptible run otherwise pays start_step full data
    # loads (JPEG decode, transforms, CF-store join) before the first real step —
    # ~1 h at 33k steps, growing linearly — for a batch order that is not actually
    # reproducible across restarts (sharded/prefetched shuffle buffer, per-frame
    # random subtask sampling). Revert to ``config.fine_tune is not None`` once
    # spot preemption stops dominating wall-clock.
    skip_fast_forward = True
    if start_step > 0 and not skip_fast_forward:
        logging.info(f"Resuming with data-loader fast-forward through step {start_step}")
    loop_range = (
        range(start_step, config.num_train_steps)
        if skip_fast_forward
        else range(config.num_train_steps)
    )
    pbar = tqdm.tqdm(
        loop_range,
        total=config.num_train_steps,
        initial=start_step if skip_fast_forward else 0,
        dynamic_ncols=True,
    )

    timer = Timer()

    # Log initial timing info
    logging.info(f"Starting training with batch_size={config.batch_size}, num_workers={config.num_workers}")

    for step in pbar:
        # Split rng for this step
        rng, step_rng = jax.random.split(rng)
        if step < start_step:
            with timer.context("data_fetch"):
                raw_batch = next(data_iter)

            with timer.context("data_postprocess"):
                if isinstance(raw_batch, tuple):
                    obs, _ = raw_batch
                    batch = {"state": obs.state}
                else:
                    batch = raw_batch
            continue

        with timer.context("train_step_compute"), sharding.set_mesh(mesh):
            critic_state, info = ptrain_step(critic_state, policy_state, batch, step_rng)

        with timer.context("train_step_sync"):
            jax.block_until_ready(critic_state)
            jax.block_until_ready(info)

        if step % config.log_interval == 0:
            info = jax.device_get(info)
            # Add timing info to logged metrics (average and total)
            total_times = timer.get_total_times(reset=False)
            avg_times = timer.get_average_times(reset=True)
            timing_info = {f"average_times/{k}": v for k, v in avg_times.items()}
            timing_info.update({f"total_times/{k}": v for k, v in total_times.items()})
            info.update(timing_info)

            info = {k: float(v) for k, v in info.items()}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(info, step=step)

        # Break down data loading into components
        with timer.context("data_fetch"):
            raw_batch = next(data_iter)

        with timer.context("data_postprocess"):
            if isinstance(raw_batch, tuple):
                obs, _ = raw_batch
                batch = {"state": obs.state}
            else:
                batch = raw_batch

        # Orbax surfaces the SIGTERM that JAX's coordination service broadcasts to every
        # host, so all processes agree on the same bail-out step. Returns False when
        # jax.distributed was never initialized, so single-host configs are unaffected.
        preempted = checkpoint_manager.reached_preemption(step + 1)
        if preempted or (step + 1) % config.save_interval == 0 or step + 1 == config.num_train_steps:
            with timer.context("checkpoint_save"):
                state_to_save = training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)
                _checkpoints.save_state(checkpoint_manager, state_to_save, data_loader, step + 1)
        if preempted:
            # The save above is async; without this the process dies mid-write.
            checkpoint_manager.wait_until_finished()
            logging.info("Preemption signal at step %d: checkpoint committed, exiting.", step + 1)
            raise SystemExit(0)

        # Generate validation plots (all workers participate for FSDP, only worker 0 creates plots/logs)
        # The third disjunct fires once at the resumed step so we can inspect the
        # restored model state before further training shifts it.
        if (
            (step + 1) % config.plot_interval == 0
            or step + 1 == config.num_train_steps
            # or (step % config.plot_interval == 0 and step == start_step and start_step > 0)
        ):
            with timer.context("validation_plot"):
                model = nnx.merge(critic_state.model_def, critic_state.params)

                if data_config.rlds_dataset_class in ("robocoin", "hdf5", "lerobot"):
                    generate_validation_plots_rlds(
                        model = model,
                        val_episode_indices = val_episode_indices,
                        step = step,
                        action_conditioned = action_conditioned,
                        data_config = data_config,
                        cache_dir = val_episodes_cache_dir,
                        batch_size = 8,
                    )
                del model

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    # Keep all hosts alive until rank 0 has fully completed async rendering/logging.
    if jax.process_index() == 0 and _render_thread is not None and _render_thread.is_alive():
        logging.info("Waiting for render thread to finish")
        _render_thread.join()
    if jax.process_count() > 1:
        logging.info("Waiting at post-render multihost barrier")
        multihost_utils.sync_global_devices("train_value_function_post_render_join")


if __name__ == "__main__":
    main(_config.cli())
