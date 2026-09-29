"""Partial-episode filtering for HDF5-schema RLDS episodes."""

import pytest
import tensorflow as tf

from openpi.training.hdf5_rlds_dataset import Hdf5RldsDataset
from openpi.training.rlds_dataset import BaseRldsDataset


@pytest.mark.parametrize("drop_partial", [False, True])
def test_trajectory_filter_drops_partial_episodes(monkeypatch, drop_partial):
    monkeypatch.setattr(BaseRldsDataset, "__init__", lambda self, *args, **kwargs: None)
    dataset = Hdf5RldsDataset(
        data_dir = "unused",
        batch_size = 1,
        datasets = (),
        drop_partial_episodes = drop_partial,
    )
    for is_partial in (False, True):
        trajectory = {"traj_metadata": {"episode_metadata": {"is_partial": tf.constant([is_partial, is_partial])}}}
        assert bool(dataset.trajectory_filter(trajectory)) == (not (drop_partial and is_partial))
