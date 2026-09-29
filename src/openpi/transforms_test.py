import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_delta_actions_counterfactual_keys():
    # Cached counterfactual chunks ([k, ah, ad]) must be delta-converted exactly
    # like their non-counterfactual counterpart (state vs next_state) with the
    # RoboCOIN bimanual EEF mask + rpy composition.
    rng = np.random.default_rng(86)
    mask = _transforms.make_bool_mask(6, -1, 6, -1)
    rpy_index_start = (3, 10)
    state = rng.standard_normal(14).astype(np.float32)
    next_state = rng.standard_normal(14).astype(np.float32)
    actions = rng.standard_normal((50, 14)).astype(np.float32)
    next_actions = rng.standard_normal((50, 14)).astype(np.float32)
    k = 8
    item = {
        "state": state,
        "next_state": next_state,
        "actions": actions.copy(),
        "next_actions": next_actions.copy(),
        "counterfactual_actions": np.broadcast_to(actions, (k, 50, 14)).copy(),
        "counterfactual_next_actions": np.broadcast_to(next_actions, (k, 50, 14)).copy(),
    }
    out = _transforms.DeltaActions(mask=mask, rpy_index_start=rpy_index_start)(item)
    for i in range(k):
        assert np.allclose(out["counterfactual_actions"][i], out["actions"], atol=1e-5)
        assert np.allclose(out["counterfactual_next_actions"][i], out["next_actions"], atol=1e-5)


def test_delta_actions_counterfactual_absent_noop():
    # Without counterfactual keys behaviour is unchanged and no keys are added.
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}
    out = _transforms.DeltaActions(mask=[False, True])(item)
    assert "counterfactual_actions" not in out
    assert "counterfactual_next_actions" not in out
    assert np.all(out["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_normalize_clip_counterfactual_parity():
    # After Normalize(quantile)+Clip the counterfactual keys must be identical
    # (per candidate) to the non-counterfactual ones (shared action stats).
    rng = np.random.default_rng(86)
    stats = _normalize.NormStats(
        mean = np.zeros((50, 14), np.float32),
        std = np.ones((50, 14), np.float32),
        q01 = -np.ones((50, 14), np.float32),
        q99 = np.ones((50, 14), np.float32),
    )
    norm_stats = {
        "actions": stats,
        "next_actions": stats,
        "counterfactual_actions": stats,
        "counterfactual_next_actions": stats,
    }
    actions = rng.standard_normal((50, 14)).astype(np.float32)
    next_actions = rng.standard_normal((50, 14)).astype(np.float32)
    k = 8
    item = {
        "actions": actions.copy(),
        "next_actions": next_actions.copy(),
        "counterfactual_actions": np.broadcast_to(actions, (k, 50, 14)).copy(),
        "counterfactual_next_actions": np.broadcast_to(next_actions, (k, 50, 14)).copy(),
    }
    item = _transforms.Normalize(norm_stats, use_quantiles = True)(item)
    bound = 1.25
    item = _transforms.Clip(
        {
            "actions": (-bound, bound),
            "next_actions": (-bound, bound),
            "counterfactual_actions": (-bound, bound),
            "counterfactual_next_actions": (-bound, bound),
        }
    )(item)
    for i in range(k):
        assert np.allclose(item["counterfactual_actions"][i], item["actions"], atol=1e-6)
        assert np.allclose(item["counterfactual_next_actions"][i], item["next_actions"], atol=1e-6)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_tokenize_subtask_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len = 48)
    prefix = "Grasp the hanger. Lift the hanger off the rod."
    suffix = "Hook one side of the shirt onto the hanger."
    transform = _transforms.TokenizeSubtaskPrompt(tokenizer = tokenizer, prefix_text = prefix)

    data = transform({"prompt": prefix, "subtask_text": suffix})

    assert "subtask_start_index" in data
    assert "subtask_end_index" in data
    subtask_start_index = int(data["subtask_start_index"])
    subtask_end_index = int(data["subtask_end_index"])
    expected_prefix_ids = list(tokenizer._tokenizer.encode(f"{prefix} ", add_bos = True))
    actual_prefix_ids = [int(t) for t in data["tokenized_prompt"][: subtask_start_index]]
    assert actual_prefix_ids == expected_prefix_ids
    decoded_prefix = tokenizer.decode(data["tokenized_prompt"][: subtask_start_index])
    decoded_suffix = tokenizer.decode(data["tokenized_prompt"][subtask_start_index : subtask_end_index + 1])
    decoded_prompt = tokenizer.decode(data["tokenized_prompt"])
    assert "Grasp the hanger." in decoded_prefix
    assert "Lift the hanger off the rod." in decoded_prefix
    assert "Hook one side of the shirt onto the hanger." in decoded_suffix
    assert "rod. Hook" in decoded_prompt
    # subtask_end_index points at the trailing newline appended after the subtask, so the
    # next-token objective is supervised through the subtask's terminating newline.
    newline_id = int(tokenizer._tokenizer.encode("\n")[0])
    assert int(data["tokenized_prompt"][subtask_end_index]) == newline_id


def test_resize_images_resizes_next_image():
    transform = _transforms.ResizeImages(8, 8)
    image = np.zeros((4, 6, 3), dtype = np.uint8)
    next_image = np.zeros((5, 7, 3), dtype = np.uint8)

    result = transform(
        {
            "image": {"cam": image},
            "next_image": {"cam": next_image},
        }
    )

    assert result["image"]["cam"].shape == (8, 8, 3)
    assert result["next_image"]["cam"].shape == (8, 8, 3)


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_subtask_text_to_id():
    transform = _transforms.SubtaskTextToId(vocab = ("Grasp the hanger", "Place the hanger on the rod", "dummy"))

    assert transform({"prompt": "task", "subtask_text": b"Grasp the hanger."})["subtask_id"] == 0
    assert transform({"prompt": np.array(b"Place the hanger on the rod")})["subtask_id"] == 1
    # Empty subtask_text (task-description prompt modes) falls back to prompt.
    assert transform({"prompt": "dummy", "subtask_text": ""})["subtask_id"] == 2
    assert transform({"prompt": "dummy"})["subtask_id"].dtype == np.int32
    with pytest.raises(ValueError, match = "Unknown subtask"):
        transform({"prompt": "Lift the hanger"})
    with pytest.raises(ValueError, match = "duplicate"):
        _transforms.SubtaskTextToId(vocab = ("a", "a."))
