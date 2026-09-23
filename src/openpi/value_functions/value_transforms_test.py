"""Tests for value function transforms."""

import numpy as np

from openpi.value_functions.value_transforms import RegressionValueOutputs
from openpi.value_functions.value_transforms import ValueFunctionInputs
from openpi.value_functions.value_transforms import make_value_function_example


class TestValueFunctionInputs:
    """Tests for value function input transform."""

    def test_basic_transform(self):
        """Test basic input transformation."""
        data = make_value_function_example(obs_dim=29, action_dim=8)
        transform = ValueFunctionInputs()
        result = transform(data)

        assert "state" in result
        assert result["state"].shape == (29,)

    def test_includes_action(self):
        """Test that action is included if available."""
        data = make_value_function_example(obs_dim=29, action_dim=8)
        transform = ValueFunctionInputs()
        result = transform(data)

        assert "actions" in result
        assert result["actions"].shape == (8,)

    def test_includes_rl_fields(self):
        """Test that RL fields are included."""
        data = make_value_function_example(obs_dim=29, action_dim=8)
        transform = ValueFunctionInputs()
        result = transform(data)

        assert "reward" in result
        assert "next_state" in result
        assert "next_actions" in result
        assert "mc_return" in result
        assert "termination" in result
        assert "truncation" in result

    def test_includes_prompt(self):
        """Test that prompt is passed through."""
        data = make_value_function_example()
        transform = ValueFunctionInputs()
        result = transform(data)

        assert "prompt" in result
        assert result["prompt"] == "value_function_training"

    def test_includes_counterfactual_next_actions(self):
        """Test that cached next actions are passed through."""
        data = make_value_function_example(obs_dim=29, action_dim=8)
        data["counterfactual_next_actions"] = np.ones((6, 2, 8), dtype=np.float32)
        transform = ValueFunctionInputs()
        result = transform(data)

        assert "counterfactual_next_actions" in result
        assert result["counterfactual_next_actions"].shape == (6, 2, 8)


class TestRegressionValueOutputs:
    """Tests for regression value output transform."""

    def test_extract_value(self):
        """Test value extraction."""
        data = {"value": np.array([1.0, 2.0, 3.0])}
        transform = RegressionValueOutputs()
        result = transform(data)

        assert "value" in result
        np.testing.assert_array_equal(result["value"], np.array([1.0, 2.0, 3.0]))


class TestMakeValueFunctionExample:
    """Tests for example generator."""

    def test_example_structure(self):
        """Test that example has all required fields."""
        example = make_value_function_example(obs_dim=29, action_dim=8)

        assert "state" in example
        assert "action" in example
        assert "reward" in example
        assert "next_state" in example
        assert "next_action" in example
        assert "mc_return" in example
        assert "termination" in example
        assert "truncation" in example
        assert "prompt" in example

    def test_example_shapes(self):
        """Test example shapes."""
        example = make_value_function_example(obs_dim=10, action_dim=5)

        assert example["state"].shape == (10,)
        assert example["action"].shape == (5,)
        assert example["next_state"].shape == (10,)
        assert example["next_action"].shape == (5,)
