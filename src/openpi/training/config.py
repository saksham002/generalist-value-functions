"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
import re
import sys
from typing import Any, ClassVar, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.best_of_n as _best_of_n
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.rlds_utils.utils as _rlds_utils
from openpi.shared.action_bounds import ActionBounds
import openpi.shared.download as _download
import openpi.shared.nnx_utils as _nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.hdf5_rlds_dataset as hdf5_rlds_dataset
import openpi.training.lerobot_rlds_dataset as lerobot_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.robocoin_rlds_dataset as robocoin_rlds_dataset
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _value_functions_base
import openpi.value_functions.heads as _heads
import openpi.value_functions.networks.paligemma as _paligemma_network
import openpi.value_functions.networks.resnet as _resnet_network
import openpi.value_functions.value_function as _value_function

# =============================================================================
# Storage root. Edit once after cloning: every registered config derives its
# dataset, norm-stats, counterfactual-store and pretrained-weight paths from
# DATA_ROOT using the layout described in README.md ("Data layout"). Importing
# this module with the placeholder in place is fine; get_config() and cli()
# raise for any config that still references it, unless the paths were
# overridden on the command line.
# =============================================================================
_UNSET = "CHANGE_ME"
DATA_ROOT = f"gs://{_UNSET}"  # e.g. "gs://my-bucket/seeq" or "/mnt/data/seeq"

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for the RLDS data loader.
    rlds_data_dir: str | None = None
    # Optional override RLDS dir for validation trajectory caching only. When set,
    # `val_trajectory_dataset` reads from here instead of `rlds_data_dir` so we
    # can use a lighter (e.g. resized) variant for val without affecting training.
    val_dataset_dir: str | None = None
    rlds_dataset_class: str | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[rlds_dataset.RLDSDataset] = ()
    use_eef: bool = False
    val_split: str = "test"
    clip_normalized_bounds: dict[str, tuple[float, float]] | None = None
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    rlds_kwargs: dict[str, Any] = dataclasses.field(default_factory = dict)

    # RL training mode options (for value function training)
    critic_mode: bool = False  # If True, use value function training pipeline
    discount: float = 0.99  # Discount factor (used if MC returns not in dataset)

    # Reward transformation: r' = reward_scale * r + reward_bias (applied before MC return computation)
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # Keys to skip during normalization/unnormalization
    skip_normalize_keys: tuple[str, ...] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim,
                            action_dim_offset = model_config.action_dim_offset,
                            pad_state = model_config.pad_state_to_action_dim,
                            action_dim_mask = model_config.action_dim_mask,
                        ),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DecodePromptBytes:
    """Decode RoboCOIN prompt bytes before generic tokenization."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        for key in ("prompt", "subtask_text"):
            if key not in data:
                continue
            value = data[key]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            data[key] = value
        return data


def _critic_subtask_id_transforms(network_config) -> list[_transforms.DataTransformFn]:
    """Decode the prompt strings and map the subtask to a categorical id for ResNet critics with a vocab."""
    if isinstance(network_config, _resnet_network.ResNetNetworkConfig) and network_config.subtask_vocab is not None:
        return [DecodePromptBytes(), _transforms.SubtaskTextToId(vocab = network_config.subtask_vocab)]
    return []


@dataclasses.dataclass(frozen=True)
class AddValidationVariants:
    """Add validation variants (random actions + optional negative-text prompt)."""

    # None for critics without a text pathway (e.g. the ResNet network); the negative-prompt
    # variant is skipped in that case.
    tokenizer: _tokenizer.PaligemmaTokenizer | None
    use_quantile_norm: bool = False
    # When True (default), produce a negative-prompt variant via the RoboCOIN-specific
    # `generate_negative_subtask_text` heuristic; requires "subtask_1" to be present.
    # Set False for non-RoboCOIN datasets that only need random_actions.
    include_negative: bool = True

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data

        traj_index = None
        frame_index = None
        if "_traj_index" in data:
            traj_index = int(np.asarray(data["_traj_index"]).item())
        if "_frame_index" in data:
            frame_index = int(np.asarray(data["_frame_index"]).item())

        if self.include_negative and self.tokenizer is not None and "subtask_1" in data:
            subtask_text = _rlds_utils.decode_text(data["subtask_1"])
            negative_subtask_text = _rlds_utils.generate_negative_subtask_text(subtask_text)
            negative_tokens, negative_mask = self.tokenizer.tokenize(negative_subtask_text, state = None)
            data["negative_subtask_1_text"] = negative_subtask_text
            data["tokenized_negative_prompt"] = negative_tokens
            data["tokenized_negative_prompt_mask"] = negative_mask

        data["random_actions"] = _rlds_utils.sample_random_actions(
            np.asarray(data["actions"]),
            use_quantile_norm = self.use_quantile_norm,
            _traj_index = traj_index,
            frame_index = frame_index,
        )
        return data


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            # Always re-download norm_stats: the files are tiny and stale per-pod caches
            # silently break runs when the on-disk schema changes (e.g. action_diff added).
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir, force_download = True))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class RoboCoinRldsDataConfig(DataConfigFactory):
    """Data config for RoboCOIN using the RLDS dataset pipeline."""

    repo_id: str = "robocoin"
    assets: AssetsConfig = dataclasses.field(
        default_factory = lambda: AssetsConfig(
            assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
            asset_id = "embodiment_wise",
        )
    )

    # RLDS dataset loading
    rlds_data_dir: str = f"{DATA_ROOT}/robocoin_bimanual"
    # Optional override RLDS dir used only for validation trajectory caching.
    # Useful when training reads from an _unresized variant whose 480x720 raw
    # images blow up host RAM during val caching; set this to a resized variant.
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "robocoin_bimanual", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    shuffle_buffer_size: int = 250_000
    num_parallel_reads: int = 8
    num_parallel_calls: int = 8

    # Image and model
    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    # RL training
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    critic_mode: bool = True

    # Data pipeline options
    td_n: int | None = None
    use_eef: bool = False
    use_quantile_norm: bool = False
    filter_n: int | None = None
    mask_50fps: bool = False
    mask_boundary_actions: bool = True
    replace_boundary_actions: bool = False
    variable_horizon: bool = False
    # Lower bound (50-fps frames, fps-scaled like the upper cap) on the sampled chunk length when variable_horizon=True.
    lower_action_horizon: int = 1
    use_chunk_wise_delta: bool = False
    state_dim: int = 14
    subtask_prompt_mode: robocoin_rlds_dataset.SubtaskPromptMode = "subtask_only"

    def __post_init__(self) -> None:
        if self.mask_boundary_actions and self.replace_boundary_actions:
            raise ValueError("At most one of mask_boundary_actions and replace_boundary_actions can be True.")

    @override
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict | None:
        """Load norm stats, handling both standard openpi format and RoboCOIN-specific format.

        Standard format: {"norm_stats": {"state": {"mean": [...], ...}, ...}}
        RoboCOIN format: {"embodiment": {"observation.state": {"mean": [...], ...}, "action": {...}, ...}, ...}

        For RoboCOIN format, converts key names (observation.state → state, action → actions) and applies
        EEF combining / chunk-wise delta selection based on config flags.
        """
        if asset_id is None:
            return None

        import json

        path = epath.Path(str(assets_dir / asset_id)) / "norm_stats.json"
        if not path.exists():
            logging.info(f"Norm stats not found at {path}, skipping.")
            return None

        data = json.loads(path.read_text())

        if "norm_stats" in data:
            return _normalize.deserialize_json(path.read_text())

        # Flat single-embodiment RoboCOIN format: "observation.state" is a top-level key
        if "observation.state" in data:
            result = self._convert_robocoin_stats(data)
            logging.info(f"Loaded flat single-embodiment RoboCOIN norm_stats from {path}")
            return result

        # RoboCOIN format: top-level keys are embodiment names
        result = {}
        for embodiment, emb_data in data.items():
            if not isinstance(emb_data, dict) or "observation.state" not in emb_data:
                continue
            result[embodiment] = self._convert_robocoin_stats(emb_data)

        if not result:
            return None

        logging.info(f"Loaded embodiment-keyed RoboCOIN norm_stats from {path}, embodiments: {list(result.keys())}")
        return result

    def _convert_robocoin_stats(self, data: dict) -> dict[str, _transforms.NormStats]:
        """Convert a single embodiment's RoboCOIN-format stats to NormStats dict."""
        import numpy as np

        norm_stats: dict[str, _transforms.NormStats] = {}

        state_stats = data["observation.state"]
        raw_state_stats = _transforms.NormStats(
            mean = np.array(state_stats["mean"]),
            std = np.array(state_stats["std"]),
            q01 = np.array(state_stats["q01"]) if self.use_quantile_norm else None,
            q99 = np.array(state_stats["q99"]) if self.use_quantile_norm else None,
        )
        state_norm_stats = raw_state_stats
        if self.use_eef:
            eef_state_stats = data["eef_sim_pose_state"]
            combined_state = {}
            stat_keys = ("mean", "std", "q01", "q99") if self.use_quantile_norm else ("mean", "std")
            for stat_key in stat_keys:
                eef_arr = np.array(eef_state_stats[stat_key])
                joint_arr = np.array(state_stats[stat_key])
                combined_state[stat_key] = self._combine_eef_and_gripper_stats(eef_arr, joint_arr)
            state_norm_stats = _transforms.NormStats(**combined_state)
        if self.use_eef:
            assert state_norm_stats.mean.shape[-1] == 14, (
                f"use_eef=True requires 14D state norm stats inside the data pipeline, "
                f"got {state_norm_stats.mean.shape[-1]}D"
            )
            norm_stats["state"] = state_norm_stats
        elif self.state_dim == 14:
            assert state_norm_stats.mean.shape[-1] == 14, (
                f"state_dim=14 but norm stats have {state_norm_stats.mean.shape[-1]}D state"
            )
            norm_stats["state"] = state_norm_stats
        elif self.state_dim == 16:
            if state_norm_stats.mean.shape[-1] == 16:
                norm_stats["state"] = state_norm_stats
            else:
                norm_stats["state"] = self._pad_state_norm_stats_14_to_16(state_norm_stats)
        else:
            raise ValueError(f"Unsupported state_dim={self.state_dim}, expected 14 or 16")

        eef_action_key = "eef_sim_pose_action_diff" if self.use_chunk_wise_delta else "eef_sim_pose_action"
        ax = -1 if self.use_chunk_wise_delta else 0

        if "action" in data:
            if self.use_eef or self.use_chunk_wise_delta:
                # Chunk-wise delta actions are 14D EEF-layout (6 pose + gripper + 6 pose + gripper);
                # non-gripper dims come from eef_sim_pose_action_diff and gripper dims (6, 13) stay
                # absolute from "action". `*_diff` stats can carry an extra leading chunk dim, so
                # broadcast the absolute gripper slice to match before concat.
                eef_action_stats = data[eef_action_key]
                gripper_action_stats = data["action"]
                combined = {}
                stat_keys = ("mean", "std", "q01", "q99") if self.use_quantile_norm else ("mean", "std")
                for stat_key in stat_keys:
                    eef_arr = np.array(eef_action_stats[stat_key])
                    abs_arr = np.array(gripper_action_stats[stat_key])
                    # Grippers sit at dim//2 - 1 (left) and dim - 1 (right) of the raw action,
                    # matching `_construct_eef_repr`. Works for both 14D (6, 13) and 16D (7, 15).
                    raw_action_dim = abs_arr.shape[-1]
                    left_gripper_idx = raw_action_dim // 2 - 1
                    right_gripper_idx = raw_action_dim - 1
                    left_grip = self._broadcast_gripper_stats(
                        abs_arr[..., left_gripper_idx:left_gripper_idx + 1], eef_arr[..., :1]
                    )
                    right_grip = self._broadcast_gripper_stats(
                        abs_arr[..., right_gripper_idx:right_gripper_idx + 1], eef_arr[..., :1]
                    )
                    combined[stat_key] = np.concatenate(
                        [
                            eef_arr[..., :6],
                            left_grip,
                            eef_arr[..., 6:12],
                            right_grip,
                        ],
                        axis = ax,
                    )
                norm_stats["actions"] = _transforms.NormStats(**combined)
            else:
                action_stats = data["action"]
                norm_stats["actions"] = _transforms.NormStats(
                    mean = np.array(action_stats["mean"]),
                    std = np.array(action_stats["std"]),
                    q01 = np.array(action_stats["q01"]) if self.use_quantile_norm else None,
                    q99 = np.array(action_stats["q99"]) if self.use_quantile_norm else None,
                )

        if "state" in norm_stats:
            norm_stats["next_state"] = norm_stats["state"]
        if "actions" in norm_stats:
            norm_stats["next_actions"] = norm_stats["actions"]
            # Cached counterfactual actions stay absolute+unnormalized on disk; DeltaActions now
            # converts them to chunk-wise-delta and these aliases quantile-normalize them with the
            # same per-embodiment action stats as next_actions (Normalize selects by embodiment).
            norm_stats["counterfactual_actions"] = norm_stats["actions"]
            norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        return norm_stats

    @staticmethod
    def _broadcast_gripper_stats(abs_slice, ref_slice):
        """Broadcast absolute gripper stats to the rank of reference (possibly chunked) stats."""
        import numpy as np

        if abs_slice.ndim < ref_slice.ndim:
            return np.broadcast_to(abs_slice, ref_slice.shape).copy()
        return abs_slice

    @staticmethod
    def _combine_eef_and_gripper_stats(eef_arr, joint_arr):
        """Combine EEF pose stats with the left/right gripper slots from the joint-state stats.

        EEF stats are always 12D (xyz+rpy per arm); the grippers come from the joint-state
        stats at dim//2-1 (left) and dim-1 (right), matching the layout produced by the
        RLDS dataset's `_construct_eef_repr` helper.
        """
        import numpy as np

        total_dim = joint_arr.shape[-1]
        assert eef_arr.shape[-1] == 12, (
            f"Expected EEF stats to have 12 dims (xyz+rpy per arm), got {eef_arr.shape}"
        )
        left_gripper_index = total_dim // 2 - 1
        right_gripper_index = total_dim - 1

        return np.concatenate(
            [
                eef_arr[..., :6],
                joint_arr[..., left_gripper_index:left_gripper_index + 1],
                eef_arr[..., 6:12],
                joint_arr[..., right_gripper_index:right_gripper_index + 1],
            ],
            axis = -1,
        )

    @staticmethod
    def _pad_14_to_16(x, fill_value):
        """Pad 14D state vector to 16D: x[:7], fill, x[7:], fill."""
        import numpy as np

        return np.concatenate([x[..., :6], np.full((*x.shape[:-1], 1), fill_value), x[..., 6:13], np.full((*x.shape[:-1], 1), fill_value), x[..., 13:]], axis = -1)

    def _pad_state_norm_stats_14_to_16(self, stats: _transforms.NormStats) -> _transforms.NormStats:
        """Pad 14D norm stats to 16D with identity-like fill values."""
        return _transforms.NormStats(
            mean = self._pad_14_to_16(stats.mean, 0.0),
            std = self._pad_14_to_16(stats.std, 1.0),
            q01 = self._pad_14_to_16(stats.q01, -1.0),
            q99 = self._pad_14_to_16(stats.q99, 1.0),
        )

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        return {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
            "next_state": (-clip_bound, clip_bound),
            "next_actions": (-clip_bound, clip_bound),
            "counterfactual_actions": (-clip_bound, clip_bound),
            "counterfactual_next_actions": (-clip_bound, clip_bound),
        }

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            # Real (pre-padding) action dim. When an explicit mask is provided it is the
            # source of truth; otherwise fall back to action_dim - action_dim_offset.
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodePromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if self.replace_boundary_actions:
            transforms.append(_transforms.ReplaceMaskedActions(use_quantile_norm = self.use_quantile_norm))
        if tokenizer is not None:
            if self.subtask_prompt_mode == "all_subtasks_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeSubtaskPrompt(
                    tokenizer = tokenizer,
                    prefix_text = robocoin_rlds_dataset.ALL_SUBTASKS_HANG_PROMPT,
                )
            elif self.subtask_prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform = _transforms.TokenizeSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            # Image resizing is handled inside BaseRldsDataset before TF batching.
            transforms.extend(
                [
                    DecodePromptBytes(),
                    tokenize_transform,
                ]
            )
        transforms.extend(_critic_subtask_id_transforms(self._get_critic_network_config(model_config)))
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("RoboCoinRldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert self.state_dim == action_dim, (
                f"chunk-wise delta requires state_dim == action_dim, "
                f"got state_dim={self.state_dim}, action_dim={action_dim}"
            )
            # 14D EEF layout: left xyz (0-2), left rpy (3-5), left gripper (6),
            # right xyz (7-9), right rpy (10-12), right gripper (13).
            # Mask is True for non-gripper dims; the rpy slots are additionally overridden
            # by rpy_index_start so they use relative-rotation composition.
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3, 10))
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "robocoin",
            datasets = self.datasets,
            use_eef = self.use_eef,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            max_num_demos = self.max_num_demos,
            rlds_kwargs = {
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "mask_50fps": self.mask_50fps,
                "mask_boundary_actions": self.mask_boundary_actions or self.replace_boundary_actions,
                "variable_horizon": self.variable_horizon,
                "lower_action_horizon": self.lower_action_horizon,
                "use_chunk_wise_delta": self.use_chunk_wise_delta,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
                "state_dim": self.state_dim,
                "subtask_prompt_mode": self.subtask_prompt_mode,
            },
        )


@dataclasses.dataclass(frozen=True)
class Hdf5RldsDataConfig(DataConfigFactory):
    """Data config for HDF5-sourced datasets (e.g. ``real_hang``).

    Standalone sibling of ``RoboCoinRldsDataConfig`` — routes to the corresponding dataset
    via ``rlds_dataset_class = "hdf5"``. The real_hang assets happen to ship in the
    RoboCOIN multi-embodiment norm-stat format, so ``_load_norm_stats`` delegates
    to the RoboCOIN implementation (and the helpers it transitively reads off
    ``self``) without otherwise pulling in the RoboCOIN class hierarchy.
    """

    repo_id: str = "real_shirt_hang"
    assets: AssetsConfig = dataclasses.field(default_factory = AssetsConfig)

    # RLDS dataset loading
    rlds_data_dir: str = f"{DATA_ROOT}/hdf5"
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    shuffle_buffer_size: int = 250_000
    num_parallel_reads: int = 8
    num_parallel_calls: int = 8

    # Image and model
    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    # RL training
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    critic_mode: bool = True

    # Data pipeline options
    td_n: int | None = None
    use_eef: bool = False
    use_quantile_norm: bool = False
    filter_n: int | None = None
    filter_intervention: bool = False
    # When not None, drop all frames whose int32 `repo_index` is not in this list.
    filter_repo_index: tuple[int, ...] | None = None
    mask_boundary_actions: bool = True
    replace_boundary_actions: bool = False
    variable_horizon: bool = False
    use_chunk_wise_delta: bool = False
    state_dim: int = 16
    prompt_mode: hdf5_rlds_dataset.PromptMode = "subtask"
    subsample: bool = False

    def __post_init__(self) -> None:
        if self.mask_boundary_actions and self.replace_boundary_actions:
            raise ValueError("At most one of mask_boundary_actions and replace_boundary_actions can be True.")
        if self.variable_horizon and self.mask_boundary_actions:
            raise ValueError("variable_horizon=True requires mask_boundary_actions=False")
        # State-dim invariant: (state_dim=16, use_eef=False) or (state_dim=14, use_eef=True).
        if not ((self.state_dim == 16 and not self.use_eef) or (self.state_dim == 14 and self.use_eef)):
            raise ValueError(
                "Hdf5RldsDataConfig requires (state_dim=16, use_eef=False) or "
                f"(state_dim=14, use_eef=True); got state_dim={self.state_dim}, use_eef={self.use_eef}"
            )

    # No _load_norm_stats override — the base DataConfigFactory loader handles
    # the standard ``{"norm_stats": {...}}`` files compute_norm_stats writes.

    @staticmethod
    def _broadcast_gripper_stats(abs_slice, ref_slice):
        return RoboCoinRldsDataConfig._broadcast_gripper_stats(abs_slice, ref_slice)

    @staticmethod
    def _combine_eef_and_gripper_stats(eef_arr, joint_arr):
        return RoboCoinRldsDataConfig._combine_eef_and_gripper_stats(eef_arr, joint_arr)

    @staticmethod
    def _pad_14_to_16(x, fill_value):
        return RoboCoinRldsDataConfig._pad_14_to_16(x, fill_value)

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        return {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
            "next_state": (-clip_bound, clip_bound),
            "next_actions": (-clip_bound, clip_bound),
            "counterfactual_actions": (-clip_bound, clip_bound),
            "counterfactual_next_actions": (-clip_bound, clip_bound),
        }

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # HDF5 only has a single subtask per episode (no all_subtasks variant), but the
        # `task_description_predict_current_subtask` mode routes through the same
        # TokenizeSubtaskPrompt path as RoboCOIN.
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodePromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if self.replace_boundary_actions:
            transforms.append(_transforms.ReplaceMaskedActions(use_quantile_norm = self.use_quantile_norm))
        if tokenizer is not None:
            if self.prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            transforms.extend(
                [
                    DecodePromptBytes(),
                    tokenize_transform,
                ]
            )
        transforms.extend(_critic_subtask_id_transforms(self._get_critic_network_config(model_config)))
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("Hdf5RldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        # When compute_norm_stats produced the standard
        # openpi format (no RoboCOIN-format conversion), route action_diff stats into
        # `actions` for use_chunk_wise_delta runtimes and replicate state/actions into
        # next_state/next_actions. The RoboCOIN loader already does this internally, so
        # the action_diff routing only fires when the absolute-action stats are still in
        # `actions` (i.e., for standard-format files) and an action_diff entry is present.
        if norm_stats is not None:
            if self.use_chunk_wise_delta and "action_diff" in norm_stats:
                norm_stats["actions"] = _slice_action_diff_norm_stats(
                    norm_stats["action_diff"], model_config.action_horizon,
                    subsample = self.subsample,
                )
            if "state" in norm_stats and "next_state" not in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            if "actions" in norm_stats and "next_actions" not in norm_stats:
                norm_stats["next_actions"] = norm_stats["actions"]
            # Cached counterfactual actions are absolute+unnormalized on disk; DeltaActions
            # converts them to chunk-wise-delta and these aliases route them through
            # Normalize+Clip alongside (next_)actions. Without them the cf_* fields skip
            # Normalize entirely (apply_tree strict=False is a no-op for missing keys),
            # leaving them in raw delta units while the critic was trained on normalized
            # ones. Mirrors RoboCoinRldsDataConfig._load_norm_stats:1068-1069.
            if "actions" in norm_stats and "counterfactual_actions" not in norm_stats:
                norm_stats["counterfactual_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert self.state_dim == action_dim, (
                f"chunk-wise delta requires state_dim == action_dim, "
                f"got state_dim={self.state_dim}, action_dim={action_dim}"
            )
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3, 10))
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "hdf5",
            datasets = self.datasets,
            use_eef = self.use_eef,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            max_num_demos = self.max_num_demos,
            rlds_kwargs = {
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "filter_intervention": self.filter_intervention,
                "filter_repo_index": self.filter_repo_index,
                "mask_boundary_actions": self.mask_boundary_actions or self.replace_boundary_actions,
                "variable_horizon": self.variable_horizon,
                "use_chunk_wise_delta": self.use_chunk_wise_delta,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
                "state_dim": self.state_dim,
                "prompt_mode": self.prompt_mode,
                "subsample": self.subsample,
            },
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRldsDataConfig(DataConfigFactory):
    """Data config for LeRobot-built RLDS datasets (e.g. ``realworld_xarm_packing``).

    Routes to ``LeRobotRldsDataset`` via ``rlds_dataset_class = "lerobot"``. State and
    action are already 14D EEF, so there is no ``use_eef`` / ``state_dim`` knob.
    Supports both behavior cloning (``critic_mode=False``) and value-function training
    (``critic_mode=True``).
    """

    repo_id: str = "realworld_xarm_packing"
    assets: AssetsConfig = dataclasses.field(default_factory = AssetsConfig)

    rlds_data_dir: str = f"{DATA_ROOT}/datasets"
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    shuffle_buffer_size: int = 250_000
    # Host RSS scales with these: the trajectory-level maps buffer whole episodes.
    num_parallel_reads: int = 4
    num_parallel_calls: int = 4

    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    use_quantile_norm: bool = False
    use_chunk_wise_delta: bool = False
    # Semantics of the 14D action/state vector, which decide how use_chunk_wise_delta
    # builds its targets. Both layouts are 2 arms x (6 dims + 1 gripper):
    #   True:  dims 3:6 / 10:13 are extrinsic-xyz euler angles, so the delta is a
    #          relative rotation (R_action @ R_state.inv()), not a subtraction.
    #   False: every masked dim is a joint position, so the delta is a plain
    #          elementwise subtraction and the rotation branch must stay off.
    # Also selects the action/state source in the loader, but only for datasets that
    # ship eef_sim_pose_*; EEF-native ones are unaffected and pass through as-is.
    use_eef: bool = False
    filter_n: int | None = None
    # Requires a per-step is_partial field; drops the trailing td_n (or
    # action_horizon when td_n is None) steps of every partial subtask, in every
    # prompt_mode (unlike filter_n, which counts to episode end under task_description).
    filter_partial: bool = False
    # Requires episode_metadata/is_adversarial; drops every episode carrying the flag.
    filter_adversarial: bool = False
    prompt_mode: lerobot_rlds_dataset.PromptMode = "subtask"

    # RL / value-function training
    critic_mode: bool = False
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    td_n: int | None = None
    mask_boundary_actions: bool = True
    subsample: bool = False
    counterfactual_action_store_dir: str | None = None

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        bounds = {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
        }
        if self.critic_mode:
            bounds.update({
                "next_state": (-clip_bound, clip_bound),
                "next_actions": (-clip_bound, clip_bound),
                "counterfactual_actions": (-clip_bound, clip_bound),
                "counterfactual_next_actions": (-clip_bound, clip_bound),
            })
        return bounds

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodePromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if tokenizer is not None:
            if self.prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            transforms.extend(
                [
                    DecodePromptBytes(),
                    tokenize_transform,
                ]
            )
        transforms.extend(_critic_subtask_id_transforms(self._get_critic_network_config(model_config)))
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("LeRobotRldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        if norm_stats is not None:
            if self.use_chunk_wise_delta and "action_diff" in norm_stats:
                norm_stats["actions"] = _slice_action_diff_norm_stats(
                    norm_stats["action_diff"], model_config.action_horizon,
                    subsample = self.subsample,
                )
            # Critic mode reuses the state/actions stats for the next_* and cached
            # counterfactual fields so they go through Normalize+Clip identically.
            if self.critic_mode:
                if "state" in norm_stats and "next_state" not in norm_stats:
                    norm_stats["next_state"] = norm_stats["state"]
                if "actions" in norm_stats and "next_actions" not in norm_stats:
                    norm_stats["next_actions"] = norm_stats["actions"]
                if "actions" in norm_stats and "counterfactual_actions" not in norm_stats:
                    norm_stats["counterfactual_actions"] = norm_stats["actions"]
                    norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert action_dim == 14, f"LeRobotRldsDataConfig expects 14D actions, got {action_dim}"
            # Same mask for both layouts: 6 delta dims + 1 absolute gripper per arm.
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            rpy_index_start = (3, 10) if self.use_eef else None
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = rpy_index_start)
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = rpy_index_start)
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "lerobot",
            datasets = self.datasets,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            rlds_kwargs = {
                "use_eef": self.use_eef,
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "filter_partial": self.filter_partial,
                "filter_adversarial": self.filter_adversarial,
                "mask_boundary_actions": self.mask_boundary_actions,
                "prompt_mode": self.prompt_mode,
                "subsample": self.subsample,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
            },
        )


def _slice_action_diff_norm_stats(
    stats: _transforms.NormStats, action_horizon: int, subsample: bool = False,
) -> _transforms.NormStats:
    """Slice the leading time axis of a 2-D `(H, D)` NormStats to `(action_horizon, D)`.

    `compute_norm_stats` writes `action_diff` stats at a fixed length covering the
    longest action_horizon any consumer uses; the runtime config slices down to its
    own action_horizon when routing `action_diff` into the `actions` key.

    When `subsample=True` (matches `Hdf5RldsDataset(subsample=True)`), strides the
    stored stats `[1::2]` so the kept slots align with the half-cadence action
    chunks the model actually sees. If the resulting array is shorter than
    `action_horizon`, pads the time axis with zeros to match.
    """
    import numpy as np

    # matches subsample=True in Hdf5RldsDataset processing
    def _stride(arr):
        return None if arr is None else arr[1::2]

    if subsample:
        stats = _transforms.NormStats(
            mean = _stride(stats.mean),
            std = _stride(stats.std),
            q01 = _stride(stats.q01),
            q99 = _stride(stats.q99),
        )

    def _slice_and_pad(arr):
        if arr is None:
            return None
        sliced = arr[:action_horizon]
        if sliced.shape[0] < action_horizon:
            pad_len = action_horizon - sliced.shape[0]
            pad = np.zeros((pad_len, *sliced.shape[1:]), dtype = sliced.dtype)
            sliced = np.concatenate([sliced, pad], axis = 0)
        return sliced

    return _transforms.NormStats(
        mean = _slice_and_pad(stats.mean),
        std = _slice_and_pad(stats.std),
        q01 = _slice_and_pad(stats.q01),
        q99 = _slice_and_pad(stats.q99),
    )


@dataclasses.dataclass(frozen=True)
class FineTuneConfig:
    """Configuration for fine-tuning or validation-only evaluation on a different dataset.

    Registered in _FINE_TUNE_CONFIGS and referenced by name from TrainConfig.fine_tune.
    When applied, overrides the base config's dataset, schedule, and checkpoint intervals.
    """

    name: str = ""

    # Dataset overrides (applied via dataclasses.replace on config.data).
    # Special keys: "data_dir" -> rlds_data_dir, "dataset_name" -> splits "name:version" into datasets tuple.
    # All other keys pass through directly.
    data_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # Full data-factory replacement. When set, replaces config.data entirely (e.g. to swap
    # from RoboCoinRldsDataConfig to Hdf5RldsDataConfig). data_overrides is ignored
    # if this is non-None.
    data_factory: "DataConfigFactory | None" = None

    # Validation overrides
    include_repos: tuple[str, ...] | None = None
    validation_cache_dir: str | None = None
    num_val_trajectories: int | None = None

    # Training schedule overrides (num_train_steps is relative to pretrained step)
    num_train_steps: int | None = None
    lr_schedule: _optimizer.LRScheduleConfig | None = None

    # If True, skip training: load checkpoint, run validation, exit.
    val_only: bool = False

    # Interval / checkpoint overrides
    save_interval: int | None = None
    plot_interval: int | None = None
    keep_period: int | None = None
    log_interval: int | None = None

    action_horizon: int | None = None

    # Model config overrides (applied via dataclasses.replace on config.model)
    model_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # Policy config overrides (applied via dataclasses.replace on config.policy).
    # E.g. ``{"action_horizon": 60}`` to bump a Best-of-N wrapper's expected chunk
    # length to match a fine-tune-time data pipeline that differs from the
    # pretrain horizon.
    policy_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # If true, will overwrite the fine-tune checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume fine-tuning from the last fine-tune checkpoint.
    resume: bool = False

    # Fields that map 1:1 from FineTuneConfig to TrainConfig for apply_overrides.
    _TRAIN_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = (
        "save_interval", "plot_interval", "keep_period", "log_interval",
        "include_repos", "validation_cache_dir", "num_val_trajectories",
        "action_horizon",
    )

    def apply_overrides(self, config: "TrainConfig", pretrained_step: int | None = None) -> "TrainConfig":
        """Apply all non-None overrides from this FineTuneConfig to the given TrainConfig.

        Handles data overrides (data_dir, dataset_name, assets),
        direct field overrides (save_interval, log_interval, etc.), and — when
        pretrained_step is provided — num_train_steps (offset to absolute) and
        lr_schedule (wrapped with step offset).
        """
        config = self._apply_data_overrides(config)
        if self.model_overrides:
            config = dataclasses.replace(config, model = dataclasses.replace(config.model, **self.model_overrides))
            logging.info("Applied FineTuneConfig model_overrides: %s", list(self.model_overrides.keys()))
        if self.policy_overrides:
            if config.policy is None:
                # Skipped rather than fatal so one fine-tune recipe can serve bases with and
                # without a policy. A policy override is only meaningful to an objective that
                # samples one (CQL's Best-of-N candidates); SARSA and MC bootstrap off the
                # data's own next_actions and register no policy, so there is nothing to
                # override and nothing to get wrong. Logged, because a silently ignored
                # override is otherwise indistinguishable from one that was applied.
                logging.info(
                    "Skipping FineTuneConfig policy_overrides %s: %s registers no policy",
                    list(self.policy_overrides.keys()),
                    config.name,
                )
            else:
                config = dataclasses.replace(
                    config, policy = dataclasses.replace(config.policy, **self.policy_overrides)
                )
                logging.info("Applied FineTuneConfig policy_overrides: %s", list(self.policy_overrides.keys()))

        replacements: dict[str, Any] = {}
        for field in self._TRAIN_CONFIG_FIELDS:
            value = getattr(self, field)
            if value is not None:
                replacements[field] = value

        if pretrained_step is not None:
            if self.num_train_steps is not None:
                replacements["num_train_steps"] = pretrained_step + self.num_train_steps
            if self.lr_schedule is not None:
                replacements["lr_schedule"] = _optimizer.OffsetSchedule(
                    base = self.lr_schedule, offset = pretrained_step,
                )

        if replacements:
            config = dataclasses.replace(config, **replacements)
            logging.info("Applied FineTuneConfig overrides: %s", list(replacements.keys()))

        return config

    def initialize(
        self,
        config: "TrainConfig",
        pretrained_step: int,
        train_state: Any,
        mesh: Any,
    ) -> tuple["TrainConfig", Any, Any, Any, bool]:
        """Full fine-tuning initialization: apply overrides, create optimizer, and checkpoint manager.

        Returns (config, train_state, train_state_sharding, checkpoint_manager, ft_resuming).
        The caller is responsible for restoring from the FT checkpoint when ft_resuming is True,
        because the save structure (plain TrainState vs ActorCriticTrainState) is caller-specific.
        """
        import openpi.training.checkpoints as _checkpoints
        import openpi.training.sharding as _sharding

        config = self.apply_overrides(config, pretrained_step = pretrained_step)

        ft_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask = None)
        # The pretrained optimizer state is discarded here, so free its device buffers before
        # the fine-tune optimizer allocates its own; otherwise the two coexist on device until
        # the caller drops the old state.
        import jax

        jax.tree.map(lambda x: x.delete() if isinstance(x, jax.Array) else None, train_state.opt_state)
        ft_opt_state = ft_tx.init(train_state.params.filter(config.trainable_filter))
        train_state = train_state.replace(tx = ft_tx, opt_state = ft_opt_state)
        train_state_sharding = _sharding.fsdp_sharding(train_state, mesh)

        ft_checkpoint_dir = config.checkpoint_dir / self.name
        checkpoint_manager, ft_resuming = _checkpoints.initialize_checkpoint_dir(
            ft_checkpoint_dir,
            keep_period = config.keep_period,
            overwrite = self.overwrite,
            resume = self.resume,
        )

        logging.info(
            "Fine-tuning: %s steps from pretrained step %s, total steps = %s, checkpoint_dir = %s",
            config.num_train_steps - pretrained_step,
            pretrained_step,
            config.num_train_steps,
            ft_checkpoint_dir,
        )

        return config, train_state, train_state_sharding, checkpoint_manager, ft_resuming

    def _apply_data_overrides(self, config: "TrainConfig") -> "TrainConfig":
        """Apply data_overrides to config.data via dataclasses.replace.

        Special keys handled before passthrough:
        - "data_dir": mapped to rlds_data_dir
        - "dataset_name": split "name:version" into a new datasets tuple
        """
        # Full factory replacement takes precedence over data_overrides. The early return
        # also skips the isinstance whitelist below, which is what lets a data_factory be a
        # LeRobotRldsDataConfig; falling through would reject those.
        if self.data_factory is not None:
            logging.info(
                "Applied FineTuneConfig data_factory replacement: %s -> %s",
                type(config.data).__name__, type(self.data_factory).__name__,
            )
            return dataclasses.replace(config, data = self.data_factory)

        if not self.data_overrides:
            return config

        data_factory = config.data
        if not isinstance(data_factory, RoboCoinRldsDataConfig | Hdf5RldsDataConfig):
            raise TypeError(
                f"FineTuneConfig data overrides are only supported for RoboCoinRldsDataConfig "
                f"or Hdf5RldsDataConfig, got {type(data_factory).__name__}"
            )

        overrides = dict(self.data_overrides)
        replacements: dict[str, Any] = {}

        if "data_dir" in overrides:
            replacements["rlds_data_dir"] = overrides.pop("data_dir")

        if "dataset_name" in overrides:
            dataset_name = overrides.pop("dataset_name")
            parts = dataset_name.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"dataset_name must be in 'name:version' format, got '{dataset_name}'"
                )
            name, version = parts
            base_dataset = data_factory.datasets[0]
            new_dataset = dataclasses.replace(base_dataset, name = name, version = version)
            replacements["datasets"] = (new_dataset,)

        if "assets" in overrides:
            replacements["assets"] = overrides.pop("assets")

        replacements.update(overrides)

        new_data = dataclasses.replace(data_factory, **replacements)
        return dataclasses.replace(config, data = new_data)


# Parameters of a critic's target network / target head, which are updated by Polyak averaging
# rather than by the optimizer (see TrainConfig.trainable_filter). Same path pattern the train
# step uses for its target-parameter statistics.
_TARGET_MODULES = _nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "seeq"
    # Experiment name (defaults to config name).
    exp_name: str | None = None

    # Model or value function config.
    model: _model.BaseModelConfig | _value_functions_base.BaseValueFunctionConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)


    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    policy_lr_schedule: _optimizer.LRScheduleConfig | None = None
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 86
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 1000
    # How often (in steps) to save checkpoints.
    save_interval: int = 10000
    # How often (in steps) to generate validation plots.
    plot_interval: int = 50000
    # Number of validation trajectories to use for plotting.
    num_val_trajectories: int = 5
    # Repo IDs guaranteed to appear in validation plots. Must have length < num_val_trajectories.
    include_repos: tuple[str, ...] = ()
    # Optional directory to cache validation episodes. If not set, it defaults to {checkpoint_dir}/val_episodes.
    validation_cache_dir: str | None = None
    # Checkpoints matching step % keep_period == 0 will be preserved.
    keep_period: int | None = 100000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # Optional wandb group name for organizing runs within a project.
    wandb_group: str | None = None
    # If true, on --resume start a brand-new wandb run (and overwrite
    # wandb_id.txt) instead of continuing the existing run. Useful when logs
    # have advanced past the checkpoint step and would collide with wandb's
    # monotonic step requirement.
    wandb_new: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    # Optional frozen policy that the TD best-of-N objective samples its backup
    # candidates from (a BestOfNWrapperConfig over cached counterfactual actions).
    policy: _model.BaseModelConfig | None = None

    # === Fine-Tuning / Validation-Only Mode ===
    # A FineTuneConfig to apply. When set, overrides dataset, schedule, and intervals.
    # Holds the config object rather than its name so that tyro recurses into it and
    # exposes its fields as `--fine-tune.*` overrides; see cli().
    fine_tune: FineTuneConfig | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> epath.Path:
        """Get the checkpoint directory for this config."""
        exp_name = self.exp_name if self.exp_name else self.name
        base_path = epath.Path(self.checkpoint_base_dir)
        full_path = base_path / self.name / exp_name
        # Only resolve local paths - GCS paths (gs://) should not be resolved
        if "gs://" not in str(full_path):
            full_path = full_path.resolve()
        return full_path

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters.

        The critics' target network and target head are never optimized: their gradients are
        identically zero (the TD target sits under stop_gradient) and post_step_update moves
        them by Polyak averaging. Leaving them out halves the optimizer state and the gradient
        buffers.
        """
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter), nnx.Not(_TARGET_MODULES))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        # fine_tune holds a resolved FineTuneConfig, so there is no name left to validate:
        # cli() resolves it through get_fine_tune_config, which rejects unknown names.


# =============================================================================
# Fine-tune configs
# =============================================================================

_FINE_TUNE_CONFIGS: list[FineTuneConfig] = [
    # real_shirt_hang twin of the sim_bimanual _final config:
    # same recipe as real_shirt_hang_paligemma_cql_rlds_finetune_task_description but with the
    # longer 50k-step schedule and 5e-6 lr from sim_bimanual_assembly_..._final.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 50k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 50_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # Same as real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final but
    # with prompt_mode="subtask" (the single subtask fed directly as the text prompt),
    # for fine-tuning the robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp base.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 50k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 50_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # Same as real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final but
    # with predict_subtask_ar=True in the restated q_network_config, for fine-tuning the
    # robocoin_bimanual_paligemma_cql_rlds_subtask_ar base. Prompt mode stays
    # "task_description_predict_current_subtask" (same as that base); only the AR flag
    # differs from the plain _task_description_final config.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the subtask_ar base's
        # q_network_config (224x224 images, no_state=True, predict_subtask_ar=True,
        # default paligemma backbone, no layernorm) — restated so the pretrained
        # checkpoint loads without any shape mismatch and the AR behavior is preserved.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 20k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # Objective-agnostic shirt-hang fine-tune for any subtask_ar base: CQL, SARSA, MC.
    # Which objective runs is decided entirely by --config-name; this config carries only
    # the things that are genuinely about the fine-tune -- dataset, chunk size, schedule.
    #
    # It does NOT restate the base's network. The CQL-specific twin above restates
    # q_network_config verbatim so the pretrained checkpoint loads without a shape
    # mismatch, but that is exactly what pinned it to one objective: SARSA's config is a
    # ValueFunctionConfig with `network_config`, so the CQL key would not apply. Inheriting
    # the base's own network is both simpler and safer -- a fine-tune has no business
    # silently changing the architecture it is fine-tuning.
    #
    # policy_overrides is set for CQL's Best-of-N candidates and is skipped, with a log, on
    # a base that registers no policy. The counterfactual store is likewise CQL's: SARSA
    # and MC bootstrap off next_actions and ignore it.
    FineTuneConfig(
        name = "real_shirt_hang_finetune_subtask_ar_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # Only the chunk size. Everything else about the model comes from the base.
        model_overrides = {
            "action_horizon": 60,
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # Objective-agnostic shirt-hang fine-tune for the task_description bases (td_bon,
    # mc). Same shape as real_shirt_hang_finetune_subtask_ar_final -- no restated network,
    # so the architecture comes from whichever base is named -- but the data parameters
    # follow those bases rather than the subtask_ar one, and they are not interchangeable:
    #
    #   max_token_len 72, not 96. The task-level bases were moved to 72; the data value is
    #     passed to get_tokenizer and overrides the network's own, so a mismatch would have
    #     the tokenizer pad to one length while the network slices for another.
    #   prompt_mode "task_description", not "..._predict_current_subtask". These bases have
    #     no AR subtask head (predict_subtask_ar=False).
    #   discount 0.9995, not 0.999, matching the bases.
    #
    # This is why a second config is needed at all: the split is in the *data*, unlike the
    # CQL/SARSA split, which turned out to need no separate config once the network
    # restatement was dropped.
    #
    # The counterfactual store and policy_overrides serve CQL's Best-of-N candidates; on an
    # MC base the override is skipped with a log and the store is loaded but unused.
    FineTuneConfig(
        name = "real_shirt_hang_finetune_task_description_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.9995,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 72,
            prompt_mode = "task_description",
        ),
        model_overrides = {
            "action_horizon": 60,
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # real_lid twin of real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final:
    # identical subtask_ar FT recipe, swapped onto the real_lid HDF5 dataset and its
    # CF cache / norm-stats / validation cache.
    FineTuneConfig(
        name = "real_lid_paligemma_cql_rlds_finetune_subtask_ar_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_lid",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_lid",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_lid_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the subtask_ar base's
        # q_network_config (224x224 images, no_state=True, predict_subtask_ar=True,
        # default paligemma backbone, no layernorm) — restated so the pretrained
        # checkpoint loads without any shape mismatch and the AR behavior is preserved.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 20k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # LeRobot realworld_xarm_packing twin of the real_shirt_hang subtask_ar CQL FT:
    # same recipe, LeRobotRldsDataConfig (critic_mode, subsample) on the packing
    # dataset; max_token_len=160 (128 for the task + 32 for the predicted subtask).
    FineTuneConfig(
        name = "realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar",
        data_factory = LeRobotRldsDataConfig(
            use_eef = True,
            repo_id = "realworld_xarm_packing",
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (
                rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/realworld_xarm_packing_pi05_subtask",
            max_token_len = 160,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # Restated q_network_config (matching the subtask_ar base) so the pretrained
        # checkpoint loads cleanly; max_token_len bumped to 160 for the longer
        # (task, subtask) concatenation of the packing prompts.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # realworld_xarm_packing twins of the three shirt-hang fine-tunes, one per objective.
    #
    # Unlike the shirt-hang pair, these cannot be objective-agnostic. The shirt-hang FTs
    # restate no network because shirt-hang's token budget already matches its bases (72 for
    # the task-level ones, 96 for subtask_ar). Packing prompts are longer -- 128 for the task
    # alone, 160 with the predicted subtask, per realworld_xarm_packing_pi05 -- so the
    # network's max_token_len has to move, and the field holding it is objective-specific:
    # MCValueFunctionConfig and SARSAValueFunctionConfig call it network_config, while
    # CQLValueFunctionConfig calls it q_network_config. One dict cannot satisfy both, hence
    # three configs rather than two.
    #
    # The data max_token_len must match the network's: it is what reaches get_tokenizer, so a
    # mismatch pads to one length while the network slices for another.
    #
    # All three point at the packing CQL counterfactual store, which is the only one that
    # exists for this dataset. It serves TD-BoN's Best-of-N candidates; MC and SARSA bootstrap
    # off next_actions and load it without using it, exactly as on shirt-hang.
    FineTuneConfig(
        name = "realworld_xarm_packing_mc_finetune_task_description",
        data_factory = LeRobotRldsDataConfig(
            use_eef = True,
            repo_id = "realworld_xarm_packing",
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (
                rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/realworld_xarm_packing_pi05_subtask",
            discount = 0.9995,
            max_token_len = 160,
            prompt_mode = "task_description",
        ),
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = False,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    FineTuneConfig(
        name = "realworld_xarm_packing_td_bon_finetune_task_description",
        data_factory = LeRobotRldsDataConfig(
            use_eef = True,
            repo_id = "realworld_xarm_packing",
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (
                rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/realworld_xarm_packing_pi05_subtask",
            discount = 0.9995,
            max_token_len = 160,
            prompt_mode = "task_description",
        ),
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = False,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    FineTuneConfig(
        name = "realworld_xarm_packing_sarsa_finetune_subtask_ar",
        data_factory = LeRobotRldsDataConfig(
            use_eef = True,
            repo_id = "realworld_xarm_packing",
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (
                rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/realworld_xarm_packing_pi05_subtask",
            discount = 0.999,
            max_token_len = 160,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # lego twin of the packing subtask_ar fine-tunes. The lego policy trains in joint
    # space, but the subtask_ar base critic was pretrained on the 14D EEF layout, so
    # use_eef=True rebuilds actions/state from the dataset's eef_sim_pose_* fields and
    # switches the chunk-wise delta back to relative-rotation composition. The CF store
    # matches: it was cached with --convert-joint-actions-to-eef. Its norm stats are a
    # separate asset_id from the policy's joint-space stats, which must not be clobbered.
    FineTuneConfig(
        name = "lego_paligemma_cql_rlds_finetune_subtask_ar",
        data_factory = LeRobotRldsDataConfig(
            use_eef = True,
            repo_id = "lego",
            rlds_data_dir = f"{DATA_ROOT}/datasets/lego_new",
            datasets = (
                rlds_dataset.RLDSDataset(name = "lego", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/lego_new",
                asset_id = "norm_stats_eef",
            ),
            discount = 0.999,
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_partial = True,
            filter_adversarial = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/lego_pi05_subtask",
            max_token_len = 160,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 2_500,
        plot_interval = 1_000_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        include_repos = (),
    ),
]

if len({c.name for c in _FINE_TUNE_CONFIGS}) != len(_FINE_TUNE_CONFIGS):
    raise ValueError("FineTuneConfig names must be unique.")
_FINE_TUNE_CONFIGS_DICT: dict[str, FineTuneConfig] = {c.name: c for c in _FINE_TUNE_CONFIGS}


def get_fine_tune_config(name: str) -> FineTuneConfig:
    """Get a FineTuneConfig by name."""
    if name not in _FINE_TUNE_CONFIGS_DICT:
        closest = difflib.get_close_matches(name, _FINE_TUNE_CONFIGS_DICT.keys(), n = 1, cutoff = 0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"FineTuneConfig '{name}' not found.{closest_str}")
    return _FINE_TUNE_CONFIGS_DICT[name]


# =============================================================================
# Train configs
# =============================================================================

# Use `get_config` if you need to get a config by name in your code.
# Categorical subtask vocabulary of the real_shirt_hang HDF5 dataset (per-frame `subtask_1`
# strings, verified against the local shards). "dummy" marks frames without a subtask
# annotation and is kept as an explicit category so those frames still train the critic.
_REAL_SHIRT_HANG_SUBTASK_VOCAB: tuple[str, ...] = (
    "Grasp the hanger",
    "Lift the hanger off the rod",
    "Pass hanger from right to left arm",
    "Hook one side of the shirt onto the hanger",
    "Hook the other side of the shirt onto the hanger",
    "Place the hanger on the rod",
    "dummy",
)

_CONFIGS = [
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboCOIN CQL Q(s,a) with pi-0.5 (PaliGemma) backbone, Best-of-N policy wrapper
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds with predict_subtask_ar=True:
    # the subtask suffix stays visible to state/action/CLS queries at its natural
    # RoPE positions (no suffix blocking, no position shift) while remaining
    # causal for the next-token objective. Only the name and the flag differ.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_subtask_ar",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # =========================================================================
    # RoboCOIN pre-training arms of the objective x prompt ablation. Each differs
    # from robocoin_bimanual_paligemma_cql_rlds_subtask_ar (TD + subtask, above) in
    # exactly the way its sim_bimanual_assembly counterpart differs from
    # sim_bimanual_assembly_paligemma_td_bon_subtask_ar. Data pipeline, schedule,
    # batch size, held-out repos and PaliGemma init are shared with the base.
    # =========================================================================
    # SARSA: bootstraps off the DATASET's next action instead of the best of N
    # policy candidates, so there is no BestOfN policy and no action bounds / CQL
    # term. Prompt side is unchanged from the base (AR subtask + next-token loss).
    TrainConfig(
        name = "robocoin_bimanual_paligemma_sarsa_subtask_ar",
        model = _value_function.SARSAValueFunctionConfig(
            network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            head_config = _heads.RegressionHeadConfig(),
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            next_token_loss_weight = 0.1,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # Monte-Carlo on the task description: regresses the discounted return to the subtask
    # end, so there is no bootstrap at all -- no target network (tau), no discount on the
    # model (the return is built in the data pipeline) and no BestOfN policy, hence no
    # counterfactual action store either. Prompt matches the td_bon_task_description arm,
    # discount included, so the two differ only in objective.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_mc_task_description",
        model = _value_function.MCValueFunctionConfig(
            network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 72,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = False,
            ),
            head_config = _heads.RegressionHeadConfig(),
            action_horizon = 50,
            next_token_loss_weight = 0.0,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.9995,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            state_dim = 14,
            max_token_len = 72,
            subtask_prompt_mode = "task_description",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # TD on the task description only: same CQL/BestOfN objective as the base, but the
    # prompt carries no subtask, so the AR subtask head and its next-token loss are off
    # and the token budget halves. Mirrors sim_bimanual_assembly_paligemma_td_bon_task_description,
    # including its longer-horizon discount (0.9995 vs the base's 0.999).
    TrainConfig(
        name = "robocoin_bimanual_paligemma_td_bon_task_description",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 72,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = False,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.0,
            action_horizon = 50,
            discount = 0.9995,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.9995,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 72,
            subtask_prompt_mode = "task_description",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # TD-BoN + subtask_ar trained directly on real_shirt_hang, rather than as a fine-tune of a
    # robocoin base. Equivalent to apply_overrides(robocoin_bimanual_paligemma_cql_rlds_subtask_ar,
    # real_shirt_hang_finetune_subtask_ar_final) flattened into one config, verified field by
    # field against that merge. Everything -- objective, BestOfN policy, PaliGemma init, the
    # whole data block, action_horizon 60, predict_subtask_ar, max_token_len 96, discount 0.999
    # -- comes from that pair unchanged. The deliberate departures are only these:
    #
    #   num_train_steps 20_000 rather than pretrained_step + 20_000. There is no pretrained
    #     step to offset from, which is also why lr_schedule is a bare CosineDecaySchedule
    #     instead of the OffsetSchedule the fine-tune path wraps around it.
    #   warmup_steps 1000 rather than the fine-tune's 0. Training starts from PaliGemma, not
    #     from a critic that is already near its operating point.
    #   batch_size 128 rather than the base's 256, set here rather than passed as a launcher
    #     flag so the config is correct on its own.
    #
    # peak_lr / decay_lr stay at the fine-tune's 5e-6 / 5e-7 despite starting from PaliGemma.
    TrainConfig(
        name = "real_shirt_hang_paligemma_td_bon_subtask_ar",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 60,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 60,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 20_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-6,
            decay_steps = 20_000,
            decay_lr = 5e-7,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 60,
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds with the next-token-prediction
    # auxiliary loss disabled (next_token_loss_weight=0.0). Only the name,
    # validation_cache_dir, and the ntp weight differ from the base config.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_no_ntp",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.0,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
    ),
    # Task-specific ResNet-50 critic: ImageNet-pretrained
    # ResNet-50 per camera (ResNet50ImageNetWeightLoader) + spatial embeddings + MLP readout, with
    # categorical subtask prediction / conditioning over _REAL_SHIRT_HANG_SUBTASK_VOCAB. The
    # next_token_loss_weight weights the subtask cross-entropy of the predictor head. Data,
    # BestOfN Bellman backup and optimizer match the PaliGemma critic configs.
    TrainConfig(
        name = "real_shirt_hang_resnet_cql_rlds_subtask",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _resnet_network.ResNetNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                num_subtask_categories = len(_REAL_SHIRT_HANG_SUBTASK_VOCAB),
                subtask_vocab = _REAL_SHIRT_HANG_SUBTASK_VOCAB,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 60,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 60,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        weight_loader = weight_loaders.ResNet50ImageNetWeightLoader(params_path = f"{DATA_ROOT}/pretrained/resnet50_gn.npz"),
        data = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 100_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = f"{DATA_ROOT}/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 20_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 20_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 10_000,
        save_interval = 10_000,
        fsdp_devices = 16,
        action_horizon = 60,
        num_val_trajectories = 3,
        include_repos = (),
    ),
    # =========================================================================
    TrainConfig(
        name = "robocoin_bimanual_pi05_rlds",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/robocoin_bimanual",
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 5,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            state_dim = 14,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-5,
        ),
        optimizer = _optimizer.AdamW(weight_decay=1e-6),
        num_workers = 0,
        log_interval = 100,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    # Pi-0.5 on the real_shirt_hang dataset.
    TrainConfig(
        name = "real_shirt_hang_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "bfloat16",
        ),
        data = Hdf5RldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            # filter_intervention = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 200_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 200_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Identical to real_lid_pi05 but reading from GCS (europe-west4, for TPU training)
    # and keeping only episodes WITH subtask annotations. The real_lid build's
    # prompt_mode is task_description, so the loader's automatic
    # has_subtask_annotations filter (subtask prompt modes only) never fires; instead
    # filter_repo_index whitelists the 9 fully-annotated source dirs. repo_index is
    # the position in the build's sorted real_lid_*_hdf5 dir list
    # (hdf5_to_tfds/dexterous/dexterous_config.py); the annotated dirs per
    # annotations_final.json are:
    #   0  real_lid_full_success_r5_hdf5
    #   2  real_lid_full_success_r7_hdf5
    #   4  real_lid_full_success_r9_hdf5
    #   5  real_lid_round0_hdf5
    #   6  real_lid_round10_0801_hdf5
    #   9  real_lid_round2_0627_hdf5
    #   11 real_lid_round4_0703_hdf5
    #   13 real_lid_round6_0716_hdf5
    #   16 real_lid_round8_0712_hdf5
    TrainConfig(
        name = "real_lid_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "bfloat16",
        ),
        data = Hdf5RldsDataConfig(
            rlds_data_dir = f"{DATA_ROOT}/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_lid", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/hdf5/real_lid",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            filter_repo_index = (0, 2, 4, 5, 6, 9, 11, 13, 16),
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 60_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 60_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Subtask-conditioned copy of realworld_xarm_packing_pi05: the prompt is the
    # subtask text directly (prompt_mode="subtask") and max_token_len is reduced to 48.
    TrainConfig(
        name = "realworld_xarm_packing_pi05_subtask",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 160,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = LeRobotRldsDataConfig(
            use_eef = True,
            rlds_data_dir = f"{DATA_ROOT}/datasets",
            datasets = (rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            prompt_mode = "subtask",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 100_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 100_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 5_000,
        keep_period = 25_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Lego twin of realworld_xarm_packing_pi05_subtask: same pi-0.5 recipe on the
    # lego LeRobot RLDS dataset (3 cameras, subtask prompts). Unlike the packing
    # datasets the 14D action/state are joint positions (yam bimanual, 6 joints +
    # 1 gripper per arm), so action_space="joint" keeps the chunk-wise delta a
    # plain subtraction instead of the EEF relative-rotation composition.
    TrainConfig(
        name = "lego_pi05_subtask",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 160,
            pi05 = True,
            discrete_state_input = True,
            # Real 14D joint values occupy dims 0:14 (the packing configs use 14:28).
            # PadStatesAndActions takes the insertion offset from the mask's first True.
            action_dim_offset = 0,
            action_dim_mask = (True,) * 14 + (False,) * 18,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = LeRobotRldsDataConfig(
            repo_id = "lego",
            rlds_data_dir = f"{DATA_ROOT}/datasets/lego_new",
            datasets = (rlds_dataset.RLDSDataset(name = "lego", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = f"{DATA_ROOT}/datasets/lego_new",
                asset_id = "norm_stats",
            ),
            use_chunk_wise_delta = True,
            use_eef = False,
            use_quantile_norm = True,
            filter_n = 8,
            filter_partial = True,
            filter_adversarial = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            prompt_mode = "subtask",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 20_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 20_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 5_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _pop_fine_tune_name(argv: list[str]) -> tuple[list[str], str | None]:
    """Split `--fine-tune <name>` out of ``argv``, returning the rest and the name."""
    remaining: list[str] = []
    name: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--fine-tune":
            if index + 1 >= len(argv):
                raise ValueError("--fine-tune requires a FineTuneConfig name")
            name = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--fine-tune="):
            name = argument.split("=", 1)[1]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return remaining, name


def cli() -> TrainConfig:
    """Parse a TrainConfig from the command line.

    `--fine-tune <name>` selects a FineTuneConfig and is handled here rather than by tyro.
    tyro is given fully-instantiated configs, so every field always carries a default and
    `tyro.conf.AvoidSubcommands` (applied by overridable_config_cli) collapses unions onto
    it; a `None` default therefore renders nothing at all. Resolving the name first and
    seeding it as the default is what makes tyro recurse into the chosen config and expose
    `--fine-tune.data-factory.rlds-data-dir` and friends. Keeping selection out of the type
    also keeps fine-tunes orthogonal to base configs: any fine-tune applies to any config.
    """
    argv, fine_tune_name = _pop_fine_tune_name(sys.argv[1:])
    configs = {name: (name, config) for name, config in _CONFIGS_DICT.items()}
    if fine_tune_name is not None:
        fine_tune = get_fine_tune_config(fine_tune_name)
        configs = {
            name: (description, dataclasses.replace(config, fine_tune = fine_tune))
            for name, (description, config) in configs.items()
        }
    return _check_roots_filled(tyro.extras.overridable_config_cli(configs, args = argv))


def _check_roots_filled(config: "TrainConfig") -> "TrainConfig":
    """Fail loudly when a config still points at the DATA_ROOT placeholder."""
    hits = sorted(set(re.findall(rf"[\w:/.-]*{_UNSET}[\w/.-]*", repr(config))))
    if hits:
        raise ValueError(
            f"Config '{config.name}' references unset storage roots {hits}. "
            "Set DATA_ROOT at the top of src/openpi/training/config.py, or override the "
            "field on the command line (e.g. --data.rlds-data-dir)."
        )
    return config


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _check_roots_filled(_CONFIGS_DICT[config_name])
