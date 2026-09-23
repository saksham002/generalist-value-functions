"""Annotation filtering for inference on unannotated HDF5-schema episodes."""

import numpy as np
import pytest
import tensorflow as tf

from openpi.training.hdf5_rlds_dataset import Hdf5RldsDataset
from openpi.training.rlds_dataset import BaseRldsDataset


@pytest.mark.parametrize("allow_unannotated", [False, True])
@pytest.mark.parametrize("transform_name", ["frame_filter", "_apply_frame_transforms_to_trajectory"])
def test_annotation_filter_matches_frame_and_trajectory_modes(monkeypatch, allow_unannotated, transform_name):
    monkeypatch.setattr(BaseRldsDataset, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(BaseRldsDataset, "_apply_frame_transforms_to_trajectory", lambda self, traj: traj)
    dataset = Hdf5RldsDataset(
        data_dir = "unused",
        batch_size = 1,
        datasets = (),
        return_trajectories = True,
        prompt_mode = "task_description_predict_current_subtask",
        allow_unannotated_episodes = allow_unannotated,
    )
    transform = getattr(dataset, transform_name)
    if transform_name == "frame_filter":
        for annotated in (False, True):
            assert bool(transform({"has_subtask_annotations": tf.constant(annotated)})) == (
                annotated or allow_unannotated
            )
    else:
        trajectory = {
            "actions": tf.zeros((2, 30, 14)),
            "has_subtask_annotations": tf.constant([False, True]),
        }
        retained = transform(trajectory)
        np.testing.assert_array_equal(
            retained["has_subtask_annotations"].numpy(),
            [False, True] if allow_unannotated else [True],
        )


def test_unannotated_episodes_cannot_enter_training():
    with pytest.raises(ValueError, match = "only for trajectory evaluation"):
        Hdf5RldsDataset(
            data_dir = "unused",
            batch_size = 1,
            datasets = (),
            allow_unannotated_episodes = True,
        )
