import pathlib

import numpy as np

from openpi.training import config as _config
from openpi.training import rlds_dataset
import openpi.transforms as _transforms
from openpi.value_functions import heads as _heads
from openpi.value_functions import value_function as _value_function
from openpi.value_functions.networks import paligemma as _paligemma_network
from openpi.value_functions.networks import resnet as _resnet_network


def _make_norm_stats():
    zeros = np.zeros(14, dtype = np.float32)
    ones = np.ones(14, dtype = np.float32)
    stats = _transforms.NormStats(mean = zeros, std = ones, q01 = zeros, q99 = ones)
    return {
        "Split_aloha": {
            "state": stats,
            "actions": stats,
            "next_state": stats,
            "next_actions": stats,
        }
    }


def test_robocoin_rlds_data_config_chunk_wise_create(monkeypatch):
    monkeypatch.setattr(_config.RoboCoinRldsDataConfig, "_load_norm_stats", lambda self, *_: _make_norm_stats())

    data_config_factory = _config.RoboCoinRldsDataConfig(
        rlds_data_dir = f"{_config.DATA_ROOT}/robocoin_bimanual",
        datasets = (rlds_dataset.RLDSDataset(name = "robocoin_bimanual", version = "1.0.0", weight = 1.0),),
        use_eef = True,
        td_n = 50,
        use_chunk_wise_delta = True,
        use_quantile_norm = True,
    )
    model_config = _value_function.SARSAValueFunctionConfig(
        network_config = _paligemma_network.PaliGemmaNetworkConfig(
            state_dim = 14,
            num_cameras = 3,
            image_size = (224, 224),
            max_token_len = 48,
            action_dim = 14,
            dtype = "float32",
        ),
        head_config = _heads.RegressionHeadConfig(),
    )

    data_config = data_config_factory.create(pathlib.Path("."), model_config)

    assert data_config.rlds_dataset_class == "robocoin"
    assert data_config.rlds_data_dir == f"{_config.DATA_ROOT}/robocoin_bimanual"
    assert data_config.datasets[0].name == "robocoin_bimanual"
    assert data_config.critic_mode
    assert data_config.use_eef
    assert data_config.use_quantile_norm
    assert data_config.val_split == "val"
    assert data_config.clip_normalized_bounds == {
        "state": (-1.25, 1.25),
        "actions": (-1.25, 1.25),
        "next_state": (-1.25, 1.25),
        "next_actions": (-1.25, 1.25),
        "counterfactual_actions": (-1.25, 1.25),
        "counterfactual_next_actions": (-1.25, 1.25),
    }
    assert data_config.rlds_kwargs == {
        "td_n": 50,
        "filter_n": None,
        "lower_action_horizon": 1,
        "mask_50fps": False,
        "mask_boundary_actions": True,
        "variable_horizon": False,
        "use_chunk_wise_delta": True,
        "shuffle_buffer_size": 250_000,
        "num_parallel_reads": 8,
        "num_parallel_calls": 8,
        "image_size": (224, 224),
        "state_dim": 14,
        "subtask_prompt_mode": "subtask_only",
    }
    (delta_transform,) = data_config.data_transforms.inputs
    assert isinstance(delta_transform, _transforms.DeltaActions)
    assert delta_transform.mask == _transforms.make_bool_mask(6, -1, 6, -1)
    assert delta_transform.rpy_index_start == (3, 10)
    assert isinstance(data_config.model_transforms.inputs[0], _config.DecodePromptBytes)
    assert isinstance(data_config.model_transforms.inputs[1], _transforms.TokenizePrompt)


def test_real_shirt_hang_pi05_only_pads_actions(monkeypatch):
    monkeypatch.setattr(_config.Hdf5RldsDataConfig, "_load_norm_stats", lambda self, *_: _make_norm_stats())
    # The registered configs carry the DATA_ROOT placeholder; nothing here touches storage.
    monkeypatch.setattr(_config, "_check_roots_filled", lambda config: config)

    cfg = _config.get_config("real_shirt_hang_pi05")

    assert cfg.model.pad_state_to_action_dim is False

    _, action_spec = cfg.model.inputs_spec()
    assert action_spec.shape == (1, 60, 32)

    data_config = cfg.data.create(pathlib.Path("."), cfg.model)
    pad_transform = data_config.model_transforms.inputs[-1]
    assert isinstance(pad_transform, _transforms.PadStatesAndActions)
    assert pad_transform.model_action_dim == 32
    assert pad_transform.action_dim_offset == 14
    assert pad_transform.pad_state is False


def test_resnet_critic_gets_subtask_id_transforms(monkeypatch):
    monkeypatch.setattr(
        _config.RoboCoinRldsDataConfig, "_load_norm_stats", lambda self, assets_dir, asset_id: _make_norm_stats()
    )
    data_config_factory = _config.RoboCoinRldsDataConfig(
        rlds_data_dir = f"{_config.DATA_ROOT}/robocoin_bimanual",
        datasets = (rlds_dataset.RLDSDataset(name = "robocoin_bimanual", version = "1.0.0", weight = 1.0),),
        use_eef = True,
        td_n = 50,
        use_chunk_wise_delta = True,
        use_quantile_norm = True,
    )

    def make_model_config(**network_overrides):
        return _value_function.CQLValueFunctionConfig(
            q_network_config = _resnet_network.ResNetNetworkConfig(
                state_dim = 14, num_cameras = 3, action_dim = 14, no_state = True, **network_overrides
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            action_horizon = 50,
        )

    with_vocab = make_model_config(num_subtask_categories = 2, subtask_vocab = ("a", "b"))
    assert with_vocab.weight_dtype == "float32"
    data_config = data_config_factory.create(pathlib.Path("."), with_vocab)
    assert [type(t).__name__ for t in data_config.model_transforms.inputs] == [
        "DecodePromptBytes", "SubtaskTextToId",
    ]

    data_config = data_config_factory.create(pathlib.Path("."), make_model_config())
    assert list(data_config.model_transforms.inputs) == []


def test_real_shirt_hang_resnet_config_loads(monkeypatch):
    monkeypatch.setattr(_config, "_check_roots_filled", lambda config: config)
    config = _config.get_config("real_shirt_hang_resnet_cql_rlds_subtask")
    network_config = config.model.q_network_config
    assert isinstance(network_config, _resnet_network.ResNetNetworkConfig)
    assert network_config.num_subtask_categories == len(network_config.subtask_vocab) == 7
    assert config.model.next_token_loss_weight > 0
