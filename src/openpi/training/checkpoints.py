
from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir)
    # Only resolve local paths - GCS paths (gs://) should not be resolved
    if "gs://" not in str(checkpoint_dir):
        checkpoint_dir = checkpoint_dir.resolve()

    resuming = False

    # Only worker 0 should perform directory operations to avoid races
    if jax.process_index() == 0:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize all workers - wait for worker 0 to complete directory setup
    # This is a lightweight barrier using JAX's collective operations
    sync_array = jax.numpy.ones(())
    sync_array = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(
        jax.numpy.ones(jax.local_device_count())
    )
    sync_array.block_until_ready()

    # After sync, check if we're resuming (worker 0's decision needs to be broadcast)
    # For now, all workers re-check the directory state
    if jax.process_index() != 0:
        if checkpoint_dir.exists() and resume:
            resuming = True

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState | training_utils.ActorCriticTrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState | training_utils.ActorCriticTrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState | training_utils.ActorCriticTrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return copy_restored_arrays(_merge_params(restored["train_state"], restored["params"]))


def copy_restored_arrays(tree: at.PyTree) -> at.PyTree:
    """Re-materialize every restored array in a freshly allocated device buffer.

    Orbax assembles a restored array from per-shard ``device_put`` results. On the CPU backend,
    handing such an array to a jitted step that donates it (the train step) crashes inside XLA
    once the step completes (``AsyncValue::NotifyAvailable`` on a released buffer), so a run
    resumed from a checkpoint died at its first step. The copy keeps sharding and dtype and
    detaches the state from Orbax's buffers; the extra memory is transient.
    """
    return jax.tree.map(lambda x: jnp.array(x, copy = True) if isinstance(x, jax.Array) else x, tree)


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(
    state: training_utils.TrainState | training_utils.ActorCriticTrainState,
) -> tuple[training_utils.TrainState | training_utils.ActorCriticTrainState, at.Params | dict[str, at.Params]]:
    if isinstance(state, training_utils.ActorCriticTrainState):
        critic_state, critic_params = _split_params(state.critic)
        params = {"critic": critic_params if isinstance(critic_params, dict) else {"params": critic_params}}

        new_policy = None
        if state.policy is not None:
            policy_state, policy_params = _split_params(state.policy)
            new_policy = policy_state
            params["policy"] = policy_params if isinstance(policy_params, dict) else {"params": policy_params}

        return training_utils.ActorCriticTrainState(critic=critic_state, policy=new_policy), params

    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(
    train_state: training_utils.TrainState | training_utils.ActorCriticTrainState,
    params: dict[str, at.Params],
) -> training_utils.TrainState | training_utils.ActorCriticTrainState:
    if isinstance(train_state, training_utils.ActorCriticTrainState):
        # Unwrap if needed
        if "params" in params and "critic" not in params:
            params = params["params"]

        new_critic = _merge_params(train_state.critic, params["critic"])
        new_policy = (
            _merge_params(train_state.policy, params["policy"]) if train_state.policy and "policy" in params else None
        )
        return training_utils.ActorCriticTrainState(critic=new_critic, policy=new_policy)

    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
