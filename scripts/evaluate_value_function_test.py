import numpy as np
import pytest

from .evaluate_value_function import _write_eval_npz


@pytest.mark.parametrize("gt_ids", [None, [0, 1]])
def test_write_categorical_predictions_without_ground_truth(tmp_path, gt_ids):
    path = _write_eval_npz(
        str(tmp_path), "episode", values = [0.1, 0.2], candidate_values = [], action_grad = [],
        mc_returns = [float("nan"), float("nan")], boundaries = [], num_frames = 2,
        sample_indices = [0, 1], predicted_texts = ["hang", "pick"], gt_texts = ["", ""],
        predicted_ids = [1, 0], gt_ids = gt_ids,
    )

    with np.load(path) as result:
        np.testing.assert_array_equal(result["predicted_subtask_ids"], [1, 0])
        assert ("gt_subtask_ids" in result) == (gt_ids is not None)
        if gt_ids is not None:
            np.testing.assert_array_equal(result["gt_subtask_ids"], gt_ids)
        assert np.isnan(result["mc_returns"]).all()
