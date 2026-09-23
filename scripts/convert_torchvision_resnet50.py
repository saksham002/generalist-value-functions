"""Convert torchvision ResNet-50 ImageNet weights to the NNX ResNet50Trunk parameter layout.

Downloads (or reads) a torchvision ``resnet50`` state dict, drops the BatchNorm and classifier
parameters (the value network uses freshly initialised GroupNorm and no classifier — the same
convention as the eqxvision fork used by jflow_match) and writes an ``.npz`` whose keys are the
``/``-joined parameter paths of ``openpi.value_functions.networks.resnet.ResNet50Trunk``, e.g.
``stem_conv/kernel``, ``layers/layer1/block0/conv1/kernel``, ``layers/layer2/block0/downsample_conv/kernel``.
Conv kernels are transposed from torch ``(Cout, Cin, kh, kw)`` to flax ``(kh, kw, Cin, Cout)``.

Example:
    python scripts/convert_torchvision_resnet50.py --output <DATA_ROOT>/pretrained/resnet50_gn.npz
"""

import dataclasses
import logging
import pathlib

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# torchvision IMAGENET1K_V2 weights (the checkpoint eqxvision's CLASSIFICATION_URLS["resnet50"] points at).
DEFAULT_TORCH_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


@dataclasses.dataclass
class Args:
    output: pathlib.Path
    torch_weights: str = DEFAULT_TORCH_WEIGHTS_URL


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Map torchvision ResNet conv weights to ResNet50Trunk parameter paths."""
    converted: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for name, tensor in state_dict.items():
        parts = name.split(".")
        if name == "conv1.weight":
            key = "stem_conv/kernel"
        elif parts[0].startswith("layer") and parts[2].startswith("conv") and parts[3] == "weight":
            key = f"layers/{parts[0]}/block{parts[1]}/{parts[2]}/kernel"
        elif parts[0].startswith("layer") and parts[2] == "downsample" and parts[3] == "0" and parts[4] == "weight":
            key = f"layers/{parts[0]}/block{parts[1]}/downsample_conv/kernel"
        else:
            # BatchNorm statistics / affine params and the classifier are intentionally dropped.
            skipped.append(name)
            continue
        weight = tensor.detach().cpu().numpy().astype(np.float32)
        if weight.ndim != 4:
            raise ValueError(f"Expected a 4-D conv kernel for {name}, got shape {weight.shape}")
        converted[key] = np.transpose(weight, (2, 3, 1, 0))
    logger.info("Converted %d conv kernels, skipped %d tensors (BatchNorm / fc)", len(converted), len(skipped))
    return converted


def main(args: Args) -> None:
    logging.basicConfig(level = logging.INFO)
    if pathlib.Path(args.torch_weights).exists():
        state_dict = torch.load(args.torch_weights, map_location = "cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(args.torch_weights, map_location = "cpu")
    converted = convert_state_dict(state_dict)
    args.output.parent.mkdir(parents = True, exist_ok = True)
    with args.output.open("wb") as f:
        np.savez(f, **converted)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main(tyro.cli(Args))
