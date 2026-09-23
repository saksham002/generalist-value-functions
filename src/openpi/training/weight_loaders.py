import copy
import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        paligemma_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]

        # Value function models nest PaliGemma under network/ or q_network/
        # (and target variants for SARSA/CQL). Detect this and nest the loaded
        # params accordingly so keys align with the model state.
        if "q_network" in params and "PaliGemma" not in params:
            loaded_params = {"q_network": {"PaliGemma": paligemma_params}}
            if "target_q_network" in params:
                loaded_params["target_q_network"] = {"PaliGemma": copy.deepcopy(paligemma_params)}
        elif "network" in params and "PaliGemma" not in params:
            loaded_params = {"network": {"PaliGemma": paligemma_params}}
            if "target_network" in params:
                loaded_params["target_network"] = {"PaliGemma": copy.deepcopy(paligemma_params)}
        else:
            loaded_params = {"PaliGemma": paligemma_params}

        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen = True)
class ResNet50ImageNetWeightLoader(WeightLoader):
    """Loads torchvision ImageNet ResNet-50 conv weights into the shared trunk of a ResNetValueNetwork.

    The npz is produced by ``scripts/convert_torchvision_resnet50.py`` and keyed by ResNet50Trunk
    parameter paths (``stem_conv/kernel``, ``layers/layer1/block0/conv1/kernel``, ...). The
    ``encoder`` subtree of the value network(s) (``q_network`` / ``target_q_network`` or
    ``network`` / ``target_network``) receives a copy; GroupNorm, spatial embeddings and the
    readout keep their random init. Any npz leaf that does not exist in the model with the same
    shape is an error, since the npz is generated for exactly this trunk layout.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(self.params_path)
        with path.open("rb") as f:
            trunk_params = dict(np.load(f, allow_pickle = False))

        network_names = [name for name in ("q_network", "target_q_network", "network", "target_network") if name in params]
        if not network_names:
            raise ValueError(
                f"ResNet50ImageNetWeightLoader expects a value-function param tree, got top-level keys {sorted(params)}"
            )
        flat_ref = flax.traverse_util.flatten_dict(params, sep = "/")
        loaded: dict[str, np.ndarray] = {}
        for network_name in network_names:
            if "encoder" not in params[network_name]:
                raise ValueError(f"{network_name} has no `encoder` subtree; is it a ResNetValueNetwork?")
            for key, value in trunk_params.items():
                full_key = f"{network_name}/encoder/{key}"
                ref = flat_ref.get(full_key)
                if ref is None or tuple(ref.shape) != tuple(value.shape):
                    raise ValueError(
                        f"ResNet50ImageNetWeightLoader: {full_key} missing from the model or shape mismatch "
                        f"(npz {value.shape} vs model {None if ref is None else ref.shape})"
                    )
                loaded[full_key] = copy.deepcopy(value)
        logger.info("ResNet50ImageNetWeightLoader: loaded %d trunk leaves into %s", len(loaded), network_names)
        return _merge_params(flax.traverse_util.unflatten_dict(loaded, sep = "/"), params, missing_regex = ".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    loaded_not_in_ref = []
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        else:
            loaded_not_in_ref.append(k)

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    filled_from_ref = []
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]
            filled_from_ref.append(k)

    logger.info(
        f"_merge_params: {len(result) - len(filled_from_ref)} loaded keys matched, "
        f"{len(loaded_not_in_ref)} loaded keys not in ref, "
        f"{len(filled_from_ref)} ref keys filled from reference"
    )
    if loaded_not_in_ref:
        loaded_prefixes = sorted({k.split("/")[0] for k in loaded_not_in_ref})
        ref_prefixes = sorted({k.split("/")[0] for k in flat_ref})
        logger.info(f"  Loaded top-level prefixes (unmatched): {loaded_prefixes}")
        logger.info(f"  Reference top-level prefixes: {ref_prefixes}")
        logger.info(f"  Sample unmatched loaded keys: {sorted(loaded_not_in_ref)[:5]}")
        logger.info(f"  Sample reference keys: {sorted(flat_ref.keys())[:5]}")
    if filled_from_ref:
        logger.info(f"  Ref keys not filled by loaded: {sorted(filled_from_ref)}")

    return flax.traverse_util.unflatten_dict(result, sep="/")
