from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpi.rlds_utils import utils


@pytest.mark.parametrize(("annotated", "use_predicted"), [(False, True), (True, True), (True, False), (False, False)])
def test_categorical_subtasks_with_optional_annotations(monkeypatch, annotated, use_predicted):
    frames = [{"fps": 30, "subtask_id": np.int32(i)} if annotated else {"fps": 30} for i in range(2)]
    network_config = SimpleNamespace(subtask_vocab = ("pick", "hang"), uses_subtask_id = True)
    model = SimpleNamespace(q_network = SimpleNamespace(config = network_config))
    value_passes = utils.TrajectoryValuePredictions(
        dataset = [0.1, 0.2], negative_prompt = [], random_actions = [], counterfactual_actions = [],
        shuffled_actions = [], attn_scores = [], candidate_values = [],
    )
    predict_values = Mock(return_value = value_passes)
    predict_ids = Mock(return_value = ({}, {"episode": [1, 0]}, {}))
    monkeypatch.setattr(utils, "predict_trajectory_values", predict_values)
    monkeypatch.setattr(utils, "predict_values_categorical_subtask", predict_ids)
    kwargs = {
        "tokenizer": None, "stride": 1, "use_predicted_subtask": use_predicted,
        "use_counterfactual_actions": False, "action_conditioned": True, "batch_size": 2, "mesh": None,
    }

    if not annotated and not use_predicted:
        with pytest.raises(ValueError, match = "requires a cached subtask_id"):
            utils.predict_values_with_subtasks(model, frames, "episode", **kwargs)
        predict_values.assert_not_called()
        return

    result = utils.predict_values_with_subtasks(model, frames, "episode", **kwargs)

    assert result.values == [0.1, 0.2]
    assert result.predicted_ids == [1, 0]
    assert result.predicted_texts == ["hang", "pick"]
    assert result.gt_ids == ([0, 1] if annotated else None)
    assert result.gt_texts == (["pick", "hang"] if annotated else ["", ""])
    assert predict_values.call_args.kwargs["strip_subtask_id"] == use_predicted
    assert predict_ids.call_args.kwargs["use_predicted_subtask"] is True
