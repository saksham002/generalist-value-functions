"""Data transforms for value function training.

These transforms handle the preparation of SARSA tuples and Monte-Carlo returns
for training value functions with reinforcement learning.
"""

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class ValueFunctionInputs(transforms.DataTransformFn):
    """Transform inputs for value function training.

    Expects the dataset to contain the following RL fields:
    - state, next_state: State observations
    - actions, next_actions: Actions taken (singular action/next_action also supported)
    - reward, mc_return: Reward signals
    - termination: True if episode ended naturally (goal/failure)
    - truncation: True if episode was cut off (time limit)

    Note: For bootstrapping (TD learning), only use termination.
    Truncated states should still bootstrap to avoid bias.
    """

    def __call__(self, data: dict) -> dict:
        # Helper to convert to float32 array, handling arrays and scalars
        def to_float_array(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            return np.asarray(x, dtype=np.float32)

        # Helper to convert to bool array, handling arrays and scalars
        def to_bool_array(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            return np.asarray(x, dtype=np.bool_)

        mc_return = to_float_array(data["mc_return"])

        # Support both actions (LeRobot convention) and action (legacy)
        action = data.get("actions", data.get("action"))
        next_action = data.get("next_actions", data.get("next_action"))

        result = {
            "state": data["state"],
            "actions": action,  # Keep as 'actions' for normalization compatibility
            "reward": to_float_array(data["reward"]),
            "next_state": data["next_state"],
            "next_actions": next_action,  # Keep as 'next_actions' for consistency
            "mc_return": mc_return,
            "termination": to_bool_array(data["termination"]),
            "truncation": to_bool_array(data["truncation"]),
        }

        # Pass through prompt if available
        if "prompt" in data:
            result["prompt"] = data["prompt"]

        # Pass through cached counterfactual actions (current + next state) for
        # BestOfN when available.
        if "counterfactual_actions" in data:
            result["counterfactual_actions"] = data["counterfactual_actions"]
        if "counterfactual_next_actions" in data:
            result["counterfactual_next_actions"] = data["counterfactual_next_actions"]

        return result


@dataclasses.dataclass(frozen=True)
class RegressionValueOutputs(transforms.DataTransformFn):
    """Output transform for regression critics.

    Simply returns the scalar value prediction.
    """

    def __call__(self, data: dict) -> dict:
        return {"value": np.asarray(data["value"])}


def make_value_function_example(
    obs_dim: int = 29,
    action_dim: int = 8,
) -> dict:
    """Creates a random input example for value function training."""
    return {
        "state": np.random.rand(obs_dim).astype(np.float32),
        "action": np.random.rand(action_dim).astype(np.float32),
        "reward": np.float32(np.random.rand()),
        "next_state": np.random.rand(obs_dim).astype(np.float32),
        "next_action": np.random.rand(action_dim).astype(np.float32),
        "mc_return": np.float32(np.random.rand() * 10.0),
        "termination": np.array(False, dtype=np.bool_),  # noqa: FBT003
        "truncation": np.array(False, dtype=np.bool_),  # noqa: FBT003
        "prompt": "value_function_training",
    }

