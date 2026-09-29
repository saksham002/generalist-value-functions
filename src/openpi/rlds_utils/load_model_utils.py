"""Utilities for loading policy and value function models from checkpoints."""

from collections.abc import Callable
import dataclasses
import importlib
import logging
import os
from typing import Any

from etils import epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))


def _load_script_module(script_name: str):
    """Dynamically import a script from scripts/ as a module."""
    path = os.path.join(_SCRIPTS_DIR, script_name)
    module_name = script_name.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_module():
    """Dynamically import train_value_function.py as a module."""
    return _load_script_module("train_value_function.py")


def _state_from_params_only(checkpoint_manager, state_shape, state_sharding, step: int):
    """Materialize a training state from a checkpoint that only holds the ``params`` item.

    The released pretrained critics ship without optimizer state. The optimizer state is
    zero-initialized (what ``tx.init`` produces for AdamW) and the step is read from the
    checkpoint, so a fine-tune resumes from it exactly as from a full checkpoint;
    ``FineTuneConfig.initialize`` replaces the optimizer anyway.
    """
    params = _checkpoints.copy_restored_arrays(
        restore_params_with_shardings(checkpoint_manager, state_shape, state_sharding, step = step)
    )
    # The abstract state carries ShapeDtypeStruct leaves, which the TrainState type check rejects.
    with at.disable_typechecking():
        train_state_shape, _ = _checkpoints._split_params(state_shape)
        train_state_sharding, _ = _checkpoints._split_params(state_sharding)
        zeros = jax.jit(
            lambda: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), train_state_shape),
            out_shardings = train_state_sharding,
        )()
    state = _checkpoints._merge_params(zeros, params)

    def _with_step(component):
        if component is None:
            return None
        return dataclasses.replace(component, step = jnp.asarray(step, dtype = component.step.dtype))

    if isinstance(state, training_utils.ActorCriticTrainState):
        return training_utils.ActorCriticTrainState(critic = _with_step(state.critic), policy = _with_step(state.policy))
    return _with_step(state)


def restore_state_with_shardings(checkpoint_manager, state_shape, state_sharding, *, step: int | None = None):
    """Restore checkpoint with explicit target shardings for cross-device loading.

    Use this instead of checkpoints.restore_state when the checkpoint was saved
    on a different device topology (e.g. loading a TPU pod checkpoint on a
    single GPU). ``step=None`` restores the latest step (training-resume
    behavior); pass an explicit step to restore a specific one. A checkpoint that
    carries only the ``params`` item (the released pretrained critics) is
    materialized with a fresh optimizer state.
    """
    if step is None:
        step = checkpoint_manager.latest_step()
    if not (epath.Path(checkpoint_manager.directory) / str(step) / "train_state").exists():
        logger.info("Checkpoint step %s has no train_state item; restoring params only.", step)
        return _state_from_params_only(checkpoint_manager, state_shape, state_sharding, step)
    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state_shape)
        train_state_sharding, params_sharding = _checkpoints._split_params(state_sharding)

        def _to_restore_args(sharding_tree):
            return jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding = s), sharding_tree)

        restored = checkpoint_manager.restore(
            step = step,
            args = ocp.args.Composite(
                train_state = ocp.args.PyTreeRestore(
                    item = train_state,
                    restore_args = _to_restore_args(train_state_sharding),
                ),
                params = ocp.args.PyTreeRestore(
                    item = {"params": params},
                    restore_args = {"params": _to_restore_args(params_sharding)},
                ),
            ),
        )
    return _checkpoints.copy_restored_arrays(_checkpoints._merge_params(restored["train_state"], restored["params"]))


def restore_params_with_shardings(checkpoint_manager, state_shape, state_sharding, *, step: int | None = None):
    """Restore only inference params with explicit target shardings.

    This avoids materializing optimizer state when loading a checkpoint for
    evaluation, which substantially reduces memory pressure compared to
    restoring the full training state.
    """
    with at.disable_typechecking():
        _, params = _checkpoints._split_params(state_shape)
        _, params_sharding = _checkpoints._split_params(state_sharding)

        def _to_restore_args(sharding_tree):
            return jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding = s), sharding_tree)

        restored = checkpoint_manager.restore(
            step = step,
            args = ocp.args.Composite(
                params = ocp.args.PyTreeRestore(
                    item = {"params": params},
                    restore_args = {"params": _to_restore_args(params_sharding)},
                ),
            ),
        )
    return restored["params"]


def load_critic(
    config_name: str,
    checkpoint_path: str,
    fine_tune: str | None = None,
    step: int | None = None,
    config_override: Callable[[Any], Any] | None = None,
    fsdp_devices: int = 16,
    use_full_state_restore: bool = False,
) -> tuple[nnx.Module, dict[str, _normalize.NormStats], Any, int]:
    """Load a value-function checkpoint plus norm stats.

    Args:
        config_name: Registered train config name.
        checkpoint_path: Critic checkpoint directory.
        fine_tune: Optional fine-tune config name.
        step: Optional explicit checkpoint step. If None, uses the latest numeric step dir.
        config_override: Optional callback to apply additional config overrides before
            train-state initialization. This keeps the loader reusable across scripts
            that need task-specific config edits.

    Returns:
        Tuple of (critic_model, critic_norm_stats, resolved_config, resolved_step).
    """
    config = _config.get_config(config_name)
    if fine_tune is not None:
        ft_config = _config.get_fine_tune_config(fine_tune)
        config = ft_config.apply_overrides(config, pretrained_step = None)
    if config_override is not None:
        config = config_override(config)
    config = dataclasses.replace(config, fsdp_devices = fsdp_devices)

    train_module = load_train_module()
    rng = jax.random.PRNGKey(86)
    mesh = _sharding.make_mesh(config.fsdp_devices)
    train_state_shape, state_sharding = train_module.init_train_state(
        config, rng, mesh, resume = True,
    )

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_path,
        keep_period = None,
        overwrite = False,
        resume = True,
    )
    if use_full_state_restore:
        # Mirror train_value_function.py's resume path: restore the full
        # train_state (train_state + params Composite items) and pull the critic
        # out of it, instead of the params-only Composite restore. The
        # params-only path can deadlock on some load topologies.
        restored_train_state = restore_state_with_shardings(
            checkpoint_manager, train_state_shape, state_sharding, step = step,
        )
        critic_state = restored_train_state.critic
        critic_model = nnx.merge(critic_state.model_def, critic_state.params)
    else:
        restored = restore_params_with_shardings(
            checkpoint_manager, train_state_shape, state_sharding, step = step,
        )
        critic_model = nnx.merge(
            train_state_shape.critic.model_def,
            restored["params"]["critic"]["params"],
        )

    # Value-function critic configs leave model.action_horizon=None and take the
    # action-chunk length from TrainConfig.action_horizon (which the FineTuneConfig
    # overrides). data.create() slices the action_diff norm stats by
    # model.action_horizon, so resolve it from the TrainConfig value when the model
    # leaves it unset (mirrors how training resolves the critic action horizon).
    # Only the norm-stats path is affected; the model used for checkpoint restore is
    # left untouched.
    data_model_config = config.model
    if getattr(config.model, "action_horizon", None) is None and config.action_horizon is not None:
        data_model_config = dataclasses.replace(config.model, action_horizon = config.action_horizon)
    data_config = config.data.create(config.assets_dirs, data_model_config)
    if step is not None:
        resolved_step = step
    else:
        step_dirs = sorted(
            (d for d in os.listdir(checkpoint_path) if d.isdigit()),
            key = int,
        )
        resolved_step = int(step_dirs[-1])
    norm_stats_dir = os.path.join(checkpoint_path, str(resolved_step), "assets", data_config.asset_id)
    logger.info(f"Loading critic norm stats from {norm_stats_dir}")
    critic_norm_stats = _normalize.load(norm_stats_dir)
    logger.info(f"Loaded critic from {checkpoint_path}")
    return critic_model, critic_norm_stats, config, resolved_step


# =============================================================================
# Policy loading
# =============================================================================


@dataclasses.dataclass(frozen = True)
class LoadPolicyConfig:
    """Configuration for loading a policy checkpoint."""

    config_name: str
    checkpoint_path: str
    fine_tune: str | None = None
    step: int | None = None
    # Optional override of TrainConfig.fsdp_devices for the inference mesh. None
    # keeps the previous default (16) and the same-topology restore_state path
    # for v5e-32. Setting a different value (e.g. device_count // num_samples
    # for the BestOfN sample-parallel path) forces re-sharding via
    # restore_params_with_shardings.
    fsdp_devices: int | None = None
    # If True, the re-shard branch restores the full train_state via
    # restore_state_with_shardings instead of the params-only
    # restore_params_with_shardings. The params-only path can deadlock on some
    # GPU load topologies (multi-L40S); the full-state path is the one
    # train.py resumes with. Requires the checkpoint to contain train_state.
    use_full_state_restore: bool = False


def load_policy(load_config: LoadPolicyConfig):
    """Load a pi0 policy model from a checkpoint.

    Returns:
        model: The loaded policy model (BaseModel).
        config: The resolved TrainConfig.
    """
    config = _config.get_config(load_config.config_name)
    if load_config.fine_tune is not None:
        ft_config = _config.get_fine_tune_config(load_config.fine_tune)
        config = ft_config.apply_overrides(config, pretrained_step = None)

    # Default fsdp_devices=16 reproduces the saved (2, 16) sharding on v5e-32.
    # Override (e.g. BestOfN sample-parallel) forces re-sharding because the
    # mesh shape no longer matches the checkpoint's metadata.
    fsdp_override = load_config.fsdp_devices
    fsdp_devices = fsdp_override if fsdp_override is not None else 16
    config = dataclasses.replace(config, fsdp_devices = fsdp_devices)

    train_module = _load_script_module("train.py")
    rng = jax.random.PRNGKey(86)
    mesh = _sharding.make_mesh(config.fsdp_devices)
    train_state_shape, state_sharding = train_module.init_train_state(config, rng, mesh, resume = True)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        load_config.checkpoint_path,
        keep_period = None,
        overwrite = False,
        resume = True,
    )
    # Same-topology path (v5e-32, fsdp_devices=16, mesh (2, 16) = 32 chips):
    # restore_state uses the saved sharding metadata directly. Different-topology
    # path (override-set, or device_count != 32): the saved sharding may not cover
    # the current mesh, so force a re-shard via restore_params_with_shardings.
    saved_chip_count = 32
    if fsdp_override is None and jax.device_count() == saved_chip_count:
        class _DummyLoader:
            def state_dict(self): return {}
            def load_state_dict(self, *_args, **_kw): pass

        restored_state = _checkpoints.restore_state(
            checkpoint_manager, train_state_shape, _DummyLoader(), step = load_config.step,
        )
        params = restored_state.params
        model_def = restored_state.model_def
    elif load_config.use_full_state_restore:
        logger.info(
            "load_policy: fsdp_devices=%d device_count=%d mesh=%s; using restore_state_with_shardings (full train_state) to re-shard.",
            fsdp_devices, jax.device_count(), mesh.devices.shape,
        )
        restored_state = restore_state_with_shardings(
            checkpoint_manager, train_state_shape, state_sharding, step = load_config.step,
        )
        params = restored_state.params
        model_def = restored_state.model_def
    else:
        logger.info(
            "load_policy: fsdp_devices=%d device_count=%d mesh=%s; using restore_params_with_shardings to re-shard.",
            fsdp_devices, jax.device_count(), mesh.devices.shape,
        )
        restored = restore_params_with_shardings(
            checkpoint_manager, train_state_shape, state_sharding, step = load_config.step,
        )
        params = restored["params"]
        model_def = train_state_shape.model_def

    model = nnx.merge(model_def, params)
    logger.info(f"Loaded policy from {load_config.checkpoint_path}")

    return model, config
