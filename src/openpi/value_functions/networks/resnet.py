"""ResNet-50 network for V(s) / Q(s, a) value functions with categorical subtask conditioning.

Image-only sibling of ``PaliGemmaValueNetwork`` (no language backbone), ported from the
``ResNet_VNet`` architecture in jflow_match:

    per camera:  image (ImageNet-normalized) -> ResNet-50 trunk (GroupNorm, layer4 map)
                 -> spatial learned embeddings (kernel multiply + sum over h, w) -> Linear -> GELU -> Linear
                 (trunk and spatial embeddings are shared by all cameras)
    readout:     concat[camera embeddings, state, flattened action chunk, subtask embedding]
                 -> LayerNorm -> Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU  (features)

Subtask conditioning is categorical: a separate predictor MLP maps the camera embeddings to
subtask logits (``decode``). At train time the ground-truth ``observation.subtask_id`` embeds
into the readout input and the logits are trained with a softmax cross-entropy carried through
the ``next_token_*`` aux protocol (weighted by ``next_token_loss_weight`` in the objectives).
At inference the argmax of the predicted logits is embedded instead — the ResNet analog of
``predict_subtask_ar``.

The ResNet trunk mirrors the eqxvision fork used by jflow_match: torchvision ResNet-50 layout with
every BatchNorm replaced by a freshly initialised GroupNorm (``num_groups = max(1, channels // 16)``)
and the classifier head removed (features are the layer4 map, no average pooling).
"""

from __future__ import annotations

import dataclasses
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.transforms import normalize_subtask_text
from openpi.value_functions.networks.base_networks import BaseValueNetwork

logger = logging.getLogger(__name__)

# Module-level initializers so `nnx.eval_shape` and the real constructor produce identical
# graphdefs (same function identity), mirroring `mlp._get_orthogonal_init`.
_CONV_KERNEL_INIT = nnx.initializers.kaiming_normal()
_SPATIAL_KERNEL_INIT = nnx.initializers.kaiming_normal()

_RESNET50_BLOCKS = (3, 4, 6, 3)
_RESNET50_PLANES = (64, 128, 256, 512)
_BOTTLENECK_EXPANSION = 4
_RESNET_OUTPUT_STRIDE = 32
_STEM_FEATURES = 64

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _group_norm(num_features: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> nnx.GroupNorm:
    # BatchNorm -> GroupNorm replacement rule of eqxvision.norm_utils.replace_norm; torch/eqx epsilon.
    return nnx.GroupNorm(
        num_features, num_groups = max(1, num_features // 16), epsilon = 1e-5, dtype = dtype, rngs = rngs
    )


def feature_map_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Spatial size of the ResNet-50 layer4 map: five stride-2 stages, each ceil-dividing by 2."""
    return (-(-image_size[0] // _RESNET_OUTPUT_STRIDE), -(-image_size[1] // _RESNET_OUTPUT_STRIDE))


def imagenet_normalize(images: at.Float[at.Array, "*b h w c"]) -> at.Float[at.Array, "*b h w c"]:
    """Map openpi's [-1, 1] images to ImageNet-normalized inputs (torchvision convention)."""
    images_01 = images / 2.0 + 0.5
    mean = jnp.asarray(_IMAGENET_MEAN, dtype = images.dtype)
    std = jnp.asarray(_IMAGENET_STD, dtype = images.dtype)
    return (images_01 - mean) / std


class _Bottleneck(nnx.Module):
    """torchvision Bottleneck (v1.5: stride on the 3x3 conv) with GroupNorm."""

    def __init__(self, in_features: int, planes: int, stride: int, *, dtype: jnp.dtype, rngs: nnx.Rngs):
        out_features = planes * _BOTTLENECK_EXPANSION
        conv_kwargs = {"use_bias": False, "kernel_init": _CONV_KERNEL_INIT, "dtype": dtype, "rngs": rngs}
        self.conv1 = nnx.Conv(in_features, planes, (1, 1), strides = 1, padding = "VALID", **conv_kwargs)
        self.gn1 = _group_norm(planes, dtype = dtype, rngs = rngs)
        self.conv2 = nnx.Conv(planes, planes, (3, 3), strides = stride, padding = ((1, 1), (1, 1)), **conv_kwargs)
        self.gn2 = _group_norm(planes, dtype = dtype, rngs = rngs)
        self.conv3 = nnx.Conv(planes, out_features, (1, 1), strides = 1, padding = "VALID", **conv_kwargs)
        self.gn3 = _group_norm(out_features, dtype = dtype, rngs = rngs)
        self.downsample_conv: nnx.Conv | None = None
        self.downsample_gn: nnx.GroupNorm | None = None
        if stride != 1 or in_features != out_features:
            self.downsample_conv = nnx.Conv(
                in_features, out_features, (1, 1), strides = stride, padding = "VALID", **conv_kwargs
            )
            self.downsample_gn = _group_norm(out_features, dtype = dtype, rngs = rngs)

    def __call__(self, x: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b h2 w2 c2"]:
        identity = x if self.downsample_conv is None else self.downsample_gn(self.downsample_conv(x))
        out = jax.nn.relu(self.gn1(self.conv1(x)))
        out = jax.nn.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        return jax.nn.relu(out + identity)


class ResNet50Trunk(nnx.Module):
    """ResNet-50 up to layer4 (no pooling / classifier), NHWC, GroupNorm instead of BatchNorm."""

    def __init__(self, *, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.stem_conv = nnx.Conv(
            3, _STEM_FEATURES, (7, 7), strides = 2, padding = ((3, 3), (3, 3)),
            use_bias = False, kernel_init = _CONV_KERNEL_INIT, dtype = dtype, rngs = rngs,
        )
        self.stem_gn = _group_norm(_STEM_FEATURES, dtype = dtype, rngs = rngs)
        in_features = _STEM_FEATURES
        # Stages / blocks are keyed by name (`layer1/block0/...`) so the parameter tree matches
        # the torchvision naming used by the ImageNet weight converter and stays str-keyed for
        # `flatten_dict(sep = "/")` in the weight loaders.
        self.layers = nnx.Dict()
        for stage_index, (planes, num_blocks) in enumerate(zip(_RESNET50_PLANES, _RESNET50_BLOCKS, strict = True)):
            blocks = nnx.Dict()
            for block_index in range(num_blocks):
                stride = 2 if (stage_index > 0 and block_index == 0) else 1
                blocks[f"block{block_index}"] = _Bottleneck(in_features, planes, stride, dtype = dtype, rngs = rngs)
                in_features = planes * _BOTTLENECK_EXPANSION
            self.layers[f"layer{stage_index + 1}"] = blocks
        self.out_features = in_features

    def __call__(self, images: at.Float[at.Array, "b h w 3"]) -> at.Float[at.Array, "b fh fw c"]:
        """images: ImageNet-normalized NHWC. Returns the layer4 feature map."""
        x = jax.nn.relu(self.stem_gn(self.stem_conv(images)))
        x = nnx.max_pool(x, window_shape = (3, 3), strides = (2, 2), padding = ((1, 1), (1, 1)))
        for stage_index in range(len(_RESNET50_BLOCKS)):
            blocks = self.layers[f"layer{stage_index + 1}"]
            for block_index in range(_RESNET50_BLOCKS[stage_index]):
                x = blocks[f"block{block_index}"](x)
        return x


class _SpatialLearnedEmbeddings(nnx.Module):
    """jflow_match ``SpatialLearnedEmbeddingsV2`` for NHWC feature maps.

    The eqx kernel is ``(C, H, W, F)`` for CHW features; here it is ``(H, W, C, F)``
    (``kernel_nnx = transpose(kernel_eqx, (1, 2, 0, 3))``). Each channel is reduced over
    (h, w) with F learned spatial weightings, flattened to C * F, then bottlenecked to hidden_size.
    """

    def __init__(
        self, height: int, width: int, channels: int, num_features: int, hidden_size: int,
        *, dtype: jnp.dtype, rngs: nnx.Rngs,
    ):
        self.kernel = nnx.Param(
            _SPATIAL_KERNEL_INIT(rngs.params(), (height, width, channels, num_features), jnp.float32)
        )
        self._flat_dim = channels * num_features
        self.proj_in = nnx.Linear(self._flat_dim, hidden_size, dtype = dtype, rngs = rngs)
        self.proj_out = nnx.Linear(hidden_size, hidden_size, dtype = dtype, rngs = rngs)

    def __call__(self, features: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b hidden"]:
        embedded = jnp.einsum("bhwc,hwcf->bcf", features, self.kernel.value.astype(features.dtype))
        flat = embedded.reshape(features.shape[0], self._flat_dim)
        return self.proj_out(jax.nn.gelu(self.proj_in(flat)))


@dataclasses.dataclass(frozen = True)
class ResNetNetworkConfig:
    """Configuration for the ResNet-50 value network (sibling of ``PaliGemmaNetworkConfig``).

    Categorical subtask conditioning is enabled by ``num_subtask_categories``. ``subtask_vocab``
    (subtask string per category index) drives the ``SubtaskTextToId`` data transform that
    produces ``observation.subtask_id``; entries beyond the vocab length are unused ids.
    """

    # Proprioceptive state dimension
    state_dim: int

    # Number of camera images (default 3 for RoboCOIN)
    num_cameras: int = 3

    # Input image resolution (ImageNet-pretrained ResNet-50: 224x224 -> 7x7 layer4 map)
    image_size: tuple[int, int] = (224, 224)

    # Action dimension (required when action_horizon is provided)
    action_dim: int = 14

    # Compute dtype for convolutions / linears; parameters stay float32
    dtype: str = "float32"

    # Whether to drop the state from the readout input (for ablation studies)
    no_state: bool = False

    # Fix order in which to iterate through keys
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # Width of the spatial-embedding bottleneck, the readout MLP and the subtask predictor
    hidden_size: int = 512

    # Number of learned spatial weightings per channel in the spatial embeddings
    spatial_num_features: int = 8

    # Number of categorical subtask ids (None disables subtask prediction / conditioning)
    num_subtask_categories: int | None = None

    # Width of the learned subtask embedding concatenated to the readout input
    subtask_embed_dim: int = 64

    # Subtask string per category index, used by the SubtaskTextToId data transform
    subtask_vocab: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.image_keys) != self.num_cameras:
            raise ValueError(f"image_keys ({len(self.image_keys)}) must match num_cameras ({self.num_cameras})")
        if self.num_subtask_categories is not None and self.num_subtask_categories <= 0:
            raise ValueError(f"num_subtask_categories must be positive, got {self.num_subtask_categories}")
        if self.subtask_vocab is not None:
            if self.num_subtask_categories is None:
                raise ValueError("subtask_vocab requires num_subtask_categories")
            if len(self.subtask_vocab) > self.num_subtask_categories:
                raise ValueError(
                    f"subtask_vocab has {len(self.subtask_vocab)} entries but "
                    f"num_subtask_categories = {self.num_subtask_categories}"
                )
            normalized = [normalize_subtask_text(text) for text in self.subtask_vocab]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"subtask_vocab has duplicate entries after normalization: {self.subtask_vocab}")

    @property
    def uses_subtask_id(self) -> bool:
        return self.num_subtask_categories is not None

    @property
    def predict_subtask_ar(self) -> bool:
        """No autoregressive subtask decoding; kept so serving code can query it uniformly."""
        return False

    def get_tokenizer(self, max_len: int | None = None):
        """The ResNet network consumes no text; callers treat None as 'no tokenization'."""
        del max_len

    def create(self, rng: at.KeyArrayLike, action_horizon: int | None = None) -> ResNetValueNetwork:
        """Create a new ResNet value network with initialized parameters."""
        return ResNetValueNetwork(self, rngs = nnx.Rngs(rng), action_horizon = action_horizon)


class ResNetValueNetwork(BaseValueNetwork):
    """ResNet-50 value network for V(s) or Q(s, a) with categorical subtask conditioning.

    See the module docstring for the architecture. ``compute_features`` returns:
    - training (``rng`` given): ``features`` or, with subtask categories configured,
      ``(features, aux)`` where ``aux`` carries the ``next_token_*`` entries the objectives
      turn into the subtask cross-entropy.
    - inference (``rng`` None): ``features``.
    """

    # Cached prefixes are (image_features [B, K*hidden], subtask_id [B], None) with batch at axis 0.
    prefix_cache_batch_axis: int = 0

    def __init__(self, config: ResNetNetworkConfig, rngs: nnx.Rngs, action_horizon: int | None = None):
        super().__init__()

        self.config = config
        self._action_conditioned = action_horizon is not None
        self._action_horizon = action_horizon if action_horizon is not None else 0
        self._action_dim = config.action_dim
        self._state_dim = config.state_dim
        self._no_state = config.no_state
        self._image_keys = config.image_keys
        self._image_size = config.image_size
        self._hidden_size = config.hidden_size
        self._feature_map_size = feature_map_size(config.image_size)
        self._uses_subtask_id = config.uses_subtask_id
        dtype = jnp.dtype(config.dtype)
        self._dtype = dtype

        # Trunk and spatial embedding are shared by all cameras (camera identity is carried by the
        # slot position in the flattened readout input).
        self.encoder = ResNet50Trunk(dtype = dtype, rngs = rngs)
        self.spatial_embedding = _SpatialLearnedEmbeddings(
            self._feature_map_size[0], self._feature_map_size[1], self.encoder.out_features,
            config.spatial_num_features, config.hidden_size, dtype = dtype, rngs = rngs,
        )
        image_feature_dim = config.num_cameras * config.hidden_size

        self.subtask_predictor_fc1: nnx.Linear | None = None
        self.subtask_predictor_norm1: nnx.LayerNorm | None = None
        self.subtask_predictor_fc2: nnx.Linear | None = None
        self.subtask_predictor_norm2: nnx.LayerNorm | None = None
        self.subtask_logits: nnx.Linear | None = None
        self.subtask_embed: nnx.Embed | None = None
        if self._uses_subtask_id:
            hidden = config.hidden_size
            self.subtask_predictor_fc1 = nnx.Linear(image_feature_dim, hidden, dtype = dtype, rngs = rngs)
            self.subtask_predictor_norm1 = nnx.LayerNorm(hidden, rngs = rngs)
            self.subtask_predictor_fc2 = nnx.Linear(hidden, hidden, dtype = dtype, rngs = rngs)
            self.subtask_predictor_norm2 = nnx.LayerNorm(hidden, rngs = rngs)
            self.subtask_logits = nnx.Linear(hidden, config.num_subtask_categories, dtype = dtype, rngs = rngs)
            self.subtask_embed = nnx.Embed(config.num_subtask_categories, config.subtask_embed_dim, rngs = rngs)

        input_dim = image_feature_dim
        if not config.no_state:
            input_dim += config.state_dim
        if self._action_conditioned:
            input_dim += self._action_horizon * config.action_dim
        if self._uses_subtask_id:
            input_dim += config.subtask_embed_dim
        self._input_dim = input_dim

        # jflow_match "regular" value net minus its final Linear, which the value head supplies.
        self.input_norm = nnx.LayerNorm(input_dim, rngs = rngs)
        self.fc1 = nnx.Linear(input_dim, config.hidden_size, dtype = dtype, rngs = rngs)
        self.norm1 = nnx.LayerNorm(config.hidden_size, rngs = rngs)
        self.fc2 = nnx.Linear(config.hidden_size, config.hidden_size, dtype = dtype, rngs = rngs)
        self.norm2 = nnx.LayerNorm(config.hidden_size, rngs = rngs)

        logger.info(
            "ResNetValueNetwork: num_cameras=%d, image_size=%s, feature_map=%s, action_conditioned=%s, "
            "action_horizon=%d, no_state=%s, hidden_size=%d, input_dim=%d, num_subtask_categories=%s",
            config.num_cameras, config.image_size, self._feature_map_size, self._action_conditioned,
            self._action_horizon, config.no_state, config.hidden_size, input_dim, config.num_subtask_categories,
        )

    @property
    def action_conditioned(self) -> bool:
        return self._action_conditioned

    @property
    @override
    def feature_dim(self) -> int:
        return self._hidden_size

    def _image_features(self, observation: _model.Observation) -> at.Float[at.Array, "b image_feature_dim"]:
        """Shared trunk + spatial embedding per camera image; masked cameras contribute zeros. Expects preprocessed images."""
        embeddings = []
        for key in self._image_keys:
            images = imagenet_normalize(observation.images[key].astype(self._dtype))
            feature_map = self.encoder(images)
            if feature_map.shape[1:3] != self._feature_map_size:
                raise ValueError(
                    f"Unexpected layer4 map {feature_map.shape[1:3]} for image_size {self._image_size}; "
                    f"expected {self._feature_map_size}"
                )
            embedding = self.spatial_embedding(feature_map)
            embedding = embedding * observation.image_masks[key][:, None].astype(embedding.dtype)
            embeddings.append(embedding)
        return jnp.concatenate(embeddings, axis = -1)

    def _subtask_predictor_features(
        self, image_features: at.Float[at.Array, "b image_feature_dim"]
    ) -> at.Float[at.Array, "b hidden"]:
        x = jax.nn.relu(self.subtask_predictor_norm1(self.subtask_predictor_fc1(image_features)))
        return jax.nn.relu(self.subtask_predictor_norm2(self.subtask_predictor_fc2(x)))

    def decode(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        """Subtask logits from predictor features (mirrors the PaliGemma next-token decode)."""
        if self.subtask_logits is None:
            raise ValueError("decode requires num_subtask_categories to be configured")
        return self.subtask_logits(x)

    def _resolve_subtask_id(
        self, observation: _model.Observation, image_features: at.Float[at.Array, "b image_feature_dim"],
    ) -> at.Int[at.Array, "*b"]:
        """The observation's ids when it carries them, otherwise the predictor's argmax.

        A negative id is the serving path's "nothing cached yet" sentinel, so the same
        traced graph serves both before and after the first cached prediction. The switch
        is for the whole batch: serving sends one observation, so a batch never mixes
        cached and uncached ids.
        """
        predictor_features = self._subtask_predictor_features(image_features)
        logits = self.decode(predictor_features[:, None, :])[:, 0, :]
        predicted_id = jnp.argmax(logits, axis = -1).astype(jnp.int32)
        if observation.subtask_id is None:
            return predicted_id
        given_id = observation.subtask_id.astype(jnp.int32)
        return jnp.where(jnp.all(given_id >= 0), given_id, predicted_id)

    def predict_subtask_id(self, observation: _model.Observation) -> at.Int[at.Array, "*b"]:
        """Predicted subtask id from images alone (ignores any ground-truth id on the observation)."""
        if not self._uses_subtask_id:
            raise ValueError("predict_subtask_id requires num_subtask_categories to be configured")
        observation = _model.preprocess_observation(
            None, observation, train = False, image_keys = self._image_keys, image_resolution = self._image_size
        )
        image_features = self._image_features(observation)
        predictor_features = self._subtask_predictor_features(image_features)
        logits = self.decode(predictor_features[:, None, :])[:, 0, :]
        return jnp.argmax(logits, axis = -1).astype(jnp.int32)

    def compute_prefix_cache(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b image_feature_dim"], at.Int[at.Array, "*b"], None]:
        """Image features (and resolved subtask id) shared by every action candidate of a state.

        Returns ``(image_features, subtask_id, None)`` with batch at axis 0 on both arrays;
        ``subtask_id`` is all zeros when subtask conditioning is disabled. Inference only.
        """
        observation = _model.preprocess_observation(
            None, observation, train = False, image_keys = self._image_keys, image_resolution = self._image_size
        )
        image_features = self._image_features(observation)
        if self._uses_subtask_id:
            subtask_id = self._resolve_subtask_id(observation, image_features)
        else:
            subtask_id = jnp.zeros((image_features.shape[0],), dtype = jnp.int32)
        return image_features, subtask_id, None

    def _readout(
        self,
        observation: _model.Observation,
        image_features: at.Float[at.Array, "b image_feature_dim"],
        subtask_id: at.Int[at.Array, "*b"] | None,
        action: at.Float[at.Array, "b ah ad"] | None,
    ) -> at.Float[at.Array, "b hidden"]:
        batch_size = image_features.shape[0]
        parts = [image_features]
        if not self._no_state:
            parts.append(observation.state.astype(image_features.dtype))
        if self._action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned network")
            if action.shape[1:] != (self._action_horizon, self._action_dim):
                raise ValueError(
                    f"Expected actions of shape [B, {self._action_horizon}, {self._action_dim}], got {action.shape}"
                )
            if observation.action_mask is not None:
                action = action * observation.action_mask[..., None].astype(action.dtype)
            parts.append(
                action.reshape(batch_size, self._action_horizon * self._action_dim).astype(image_features.dtype)
            )
        if self._uses_subtask_id:
            parts.append(self.subtask_embed(subtask_id).astype(image_features.dtype))
        x = jnp.concatenate(parts, axis = -1)
        x = self.input_norm(x)
        x = jax.nn.relu(self.norm1(self.fc1(x)))
        return jax.nn.relu(self.norm2(self.fc2(x)))

    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
        prefix_cache: tuple[at.Array, at.Array, None] | None = None,
    ) -> at.Float[at.Array, "*b feature_dim"] | tuple[at.Float[at.Array, "*b feature_dim"], dict[str, at.Array]]:
        """Compute value features from observation (and optionally action).

        Args:
            observation: Observation with images, image_masks, state, optional action_mask and
                subtask_id. Images should already be in [-1, 1].
            action: [B, action_horizon, action_dim] chunk when action_conditioned.
            rng: Random key for image augmentation; its presence marks training mode.
            prefix_cache: Output of ``compute_prefix_cache`` (inference only) to skip the trunks.

        Returns:
            Training with subtask categories configured: ``(features, aux)`` where aux holds
            ``next_token_embeddings`` [B, 1, hidden], ``next_token_targets`` [B, 1] and
            ``next_token_mask`` [B, 1] for the subtask cross-entropy. Otherwise ``features``.
        """
        if observation.state.ndim != 2:
            raise ValueError(f"ResNetValueNetwork expects [B, state_dim] observations, got {observation.state.shape}")
        train = rng is not None

        if prefix_cache is not None:
            if train:
                raise ValueError("prefix_cache is only supported for inference.")
            image_features, subtask_id, _ = prefix_cache
            return self._readout(observation, image_features, subtask_id, action)

        observation = _model.preprocess_observation(
            rng, observation, train = train, image_keys = self._image_keys, image_resolution = self._image_size
        )
        image_features = self._image_features(observation)

        if not self._uses_subtask_id:
            return self._readout(observation, image_features, None, action)

        if train:
            if observation.subtask_id is None:
                raise ValueError(
                    "Training a subtask-conditioned ResNetValueNetwork requires observation.subtask_id "
                    "(is the SubtaskTextToId transform wired for this data config?)"
                )
            subtask_id = observation.subtask_id.astype(jnp.int32)
            features = self._readout(observation, image_features, subtask_id, action)
            predictor_features = self._subtask_predictor_features(image_features)
            aux = {
                "next_token_embeddings": predictor_features[:, None, :],
                "next_token_targets": subtask_id[:, None],
                "next_token_mask": jnp.ones((subtask_id.shape[0], 1), dtype = jnp.bool_),
            }
            return features, aux

        subtask_id = self._resolve_subtask_id(observation, image_features)
        return self._readout(observation, image_features, subtask_id, action)
