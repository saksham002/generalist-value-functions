from collections.abc import Iterator, Sequence
import dataclasses
import logging
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.rlds_dataset as rlds_dataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split: str = "train",
    shuffle: bool = False,
    return_trajectories: bool = False,
    max_trajectories: int | None = None,
) -> Dataset:
    if data_config.rlds_dataset_class == "robocoin":
        from openpi.training.robocoin_rlds_dataset import RoboCoinRldsDataset

        return RoboCoinRldsDataset(
            data_dir = data_config.rlds_data_dir,
            batch_size = batch_size,
            split = split,
            shuffle = shuffle,
            action_chunk_size = action_horizon,
            datasets = data_config.datasets,
            critic_mode = data_config.critic_mode,
            discount = data_config.discount,
            reward_scale = data_config.reward_scale,
            reward_bias = data_config.reward_bias,
            use_eef = data_config.use_eef,
            counterfactual_action_store_dir = data_config.counterfactual_action_store_dir,
            max_num_demos = data_config.max_num_demos,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            **data_config.rlds_kwargs,
        )

    if data_config.rlds_dataset_class == "hdf5":
        from openpi.training.hdf5_rlds_dataset import Hdf5RldsDataset

        return Hdf5RldsDataset(
            data_dir = data_config.rlds_data_dir,
            batch_size = batch_size,
            split = split,
            shuffle = shuffle,
            action_chunk_size = action_horizon,
            datasets = data_config.datasets,
            critic_mode = data_config.critic_mode,
            discount = data_config.discount,
            reward_scale = data_config.reward_scale,
            reward_bias = data_config.reward_bias,
            use_eef = data_config.use_eef,
            counterfactual_action_store_dir = data_config.counterfactual_action_store_dir,
            max_num_demos = data_config.max_num_demos,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            **data_config.rlds_kwargs,
        )

    if data_config.rlds_dataset_class == "lerobot":
        from openpi.training.lerobot_rlds_dataset import LeRobotRldsDataset

        return LeRobotRldsDataset(
            data_dir = data_config.rlds_data_dir,
            batch_size = batch_size,
            split = split,
            shuffle = shuffle,
            action_chunk_size = action_horizon,
            datasets = data_config.datasets,
            critic_mode = data_config.critic_mode,
            discount = data_config.discount,
            reward_scale = data_config.reward_scale,
            reward_bias = data_config.reward_bias,
            counterfactual_action_store_dir = data_config.counterfactual_action_store_dir,
            max_num_demos = data_config.max_num_demos,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            **data_config.rlds_kwargs,
        )

    raise ValueError(f"Unknown rlds_dataset_class {data_config.rlds_dataset_class!r}.")


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms.

    Args:
        dataset: The dataset to transform.
        data_config: The data configuration.
        skip_norm_stats: Whether to skip data normalization.
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build the transform pipeline
    input_transforms = list(data_config.repack_transforms.inputs)
    input_transforms.extend(data_config.data_transforms.inputs)
    input_transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    if data_config.clip_normalized_bounds is not None:
        input_transforms.append(_transforms.Clip(data_config.clip_normalized_bounds))
    input_transforms.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, input_transforms)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    config_fields = {f.name: getattr(data_config, f.name) for f in dataclasses.fields(data_config)}
    if config_fields.get("norm_stats"):
        config_fields["norm_stats"] = f"<{len(config_fields['norm_stats'])} keys>"
    logging.info(f"data_config: {config_fields}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    if data_config.repo_id != "fake":
        raise ValueError("Only RLDS datasets (rlds_data_dir) and the fake dataset (repo_id='fake') are supported.")
    dataset = transform_dataset(FakeDataset(config.model, num_samples = 1024), data_config, skip_norm_stats = skip_norm_stats)
    data_loader = FakeDataLoader(
        dataset,
        local_batch_size = config.batch_size // jax.process_count(),
        sharding = sharding,
        shuffle = shuffle,
        num_batches = num_batches,
        seed = config.seed,
    )
    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires the `rlds` dependency group (tensorflow, tensorflow-datasets, dlimp).

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, split = "train", shuffle = shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class FakeDataLoader:
    """In-process numpy batching over a map-style dataset.

    Only the fake dataset goes through here (every real dataset streams through RLDS), so
    there is no worker pool: batches are stacked on the main process and sharded like the
    RLDS loader's.
    """

    def __init__(
        self,
        dataset: Dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        seed: int = 0,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")
        self._dataset = dataset
        self._local_batch_size = local_batch_size
        self._shuffle = shuffle
        self._num_batches = num_batches
        self._rng = np.random.default_rng(seed)
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding

    def __iter__(self):
        num_items = 0
        while True:
            order = self._rng.permutation(len(self._dataset)) if self._shuffle else np.arange(len(self._dataset))
            for start in range(0, len(order) - self._local_batch_size + 1, self._local_batch_size):
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                items = [self._dataset[int(i)] for i in order[start : start + self._local_batch_size]]
                # Convert to numpy before stacking since the fake dataset yields JAX arrays.
                batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis = 0), *items)
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class RLDSDataLoader:
    """Shallow wrapper around RLDS data loaders to make them compatible with openpi.

    All batching already happens in the RLDS dataset, so we don't need to do anything here.
    Supports multi-host training - each process loads its shard and batches are combined.
    """

    def __init__(
        self,
        dataset: rlds_dataset.BaseRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        dataset_size: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches
        self._dataset_size = dataset_size

        if sharding is None:
            # Use data parallel sharding by default across all devices (including multi-host).
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    @property
    def dataset_size(self) -> int | None:
        """Return the number of samples in the dataset, or None if not precomputed."""
        return self._dataset_size

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                batch = {
                    key: value
                    for key, value in batch.items()
                    if not np.issubdtype(np.asarray(value).dtype, np.str_)
                    and not np.issubdtype(np.asarray(value).dtype, np.bytes_)
                }
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: "FakeDataLoader | RLDSDataLoader"):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def dataset_size(self) -> int | None:
        """Return the number of samples in the dataset, or None if unknown."""
        return self._data_loader.dataset_size

    def __iter__(self):
        for batch in self._data_loader:
            if self._data_config.critic_mode:
                # Critic mode: yield raw batch dict for value function training
                yield batch
            else:
                # Policy mode: yield (Observation, Actions) tuple
                yield _model.Observation.from_dict(batch), batch["actions"]
