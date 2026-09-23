"""PaliGemma network for V(s, l) value functions with image+state+text inputs.

Uses a pre-trained PaliGemma backbone (ViT + Gemma LLM) as a feature encoder
for training subtask-conditioned state value functions V(s, l) where l is a subtask.

Sequence structure:
    [img1_patches(256)] [img2_patches(256)] [img3_patches(256)] [text_tokens...] [state_embed] [(actions)] [CLS]

Where:
- img*_patches: 256 patches per image from ViT (14x14 patches from 224x224)
- text_tokens: Tokenized subtask prompt (randomly sampled at data loading time)
- state_embed: Projected proprioceptive state (single token)
- CLS: Global learnable token for value extraction - attends to all but not attended by others

Attention Mask Design:
- Image patches and text tokens can attend to each other (bidirectional)
- State token can attend to images and text (but images/text cannot attend to state)
- CLS token (last position) can attend to ALL tokens but CANNOT be attended by others

This design allows:
1. CLS to gather global context from all modalities for value prediction
2. State to condition on visual-language features
3. Visual-language stream to remain clean (not "polluted" by state)
4. CLS at the end enables simple causal masking for its attention pattern

V(s, l) Training Design:
- At data loading time, one non-null subtask is randomly sampled per datapoint
- The sampled subtask's text is used as tokenized_prompt
- MC target = gamma ** steps_to_subtask_end[sampled_idx]

Position Embeddings:
- Uses cumulative position indices based on valid token masks
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.value_functions.networks.base_networks import BaseValueNetwork

logger = logging.getLogger(__name__)


# Number of patches per image (224/14 = 16, 16*16 = 256)
NUM_PATCHES_PER_IMAGE = 256

MAX_SUBTASK_TOKENS_FOR_LOSS = 48


def make_attn_mask(
    input_mask: jax.Array,
    mask_ar: jax.Array,
    *,
    suffix_mask: jax.Array | None = None,
) -> jax.Array:
    """Create attention mask for value network with state and CLS tokens at the end.

    Sequence structure: [images] [prefix_text] [suffix_text] [state] [CLS]

    The cumsum of mask_ar groups tokens: positions sharing the same cumsum value
    attend bidirectionally; a position with higher cumsum can attend to lower but
    not vice-versa.  Suffix text tokens have mask_ar=1 so they are causal among
    themselves and invisible to prefix tokens (lower cumsum).

    When suffix_mask is provided, an additional constraint prevents non-suffix
    queries (state, actions, CLS) from attending to suffix keys, keeping the
    value-relevant outputs independent of the subtask text.

    Args:
        input_mask: bool[B, N] true if part of the input, false if padding.
        mask_ar: bool[B, N] or bool[N]. 0=bidirectional, 1=causal.
        suffix_mask: Optional bool[B, N]. True for suffix (subtask) text positions.
            Non-suffix queries are blocked from attending to suffix keys.

    Returns:
        Attention mask [B, N, N] where True means "can attend".
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)

    cumsum = jnp.cumsum(mask_ar, axis=1)
    mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]

    valid = input_mask[:, None, :] * input_mask[:, :, None]
    mask = jnp.logical_and(mask, valid)

    if suffix_mask is not None:
        suffix_mask = jnp.broadcast_to(suffix_mask, input_mask.shape)
        non_suffix_seeing_suffix = (~suffix_mask[:, :, None]) & suffix_mask[:, None, :]
        mask = mask & ~non_suffix_seeing_suffix

    return mask


def compute_rope_positions(
    input_mask: jax.Array,
    *,
    shift_start_index: int,
    subtask_start_index: jax.Array | None = None,
    subtask_end_index: jax.Array | None = None,
) -> jax.Array:
    """RoPE position indices for the full-sequence LLM forward.

    Base is ``cumsum(input_mask) - 1``. When a subtask suffix is present,
    positions at column indices ``>= shift_start_index`` are pulled down by
    the suffix length so state/action/CLS land at the same RoPE position
    they would occupy without the suffix; suffix tokens keep their natural
    positions and overlap the shifted state/action/CLS slots (benign — the
    attention mask blocks non-suffix queries from seeing suffix keys).
    """
    positions = jnp.cumsum(input_mask.astype(jnp.int32), axis = 1) - 1
    if subtask_start_index is not None and subtask_end_index is not None:
        # subtask_end_index points at the trailing "\n" _tokenize_subtask_prompt
        # appends after the subtask at train time (append_newline=True; the newline is part
        # of the next-token objective), so the inclusive span end - start + 1 covers the
        # subtask tokens plus that newline. This shift is reached only by the non-AR critic;
        # the predict_subtask_ar critic passes subtask_start/end_index as None (via
        # _position_shift_indices), keeping the subtask at its natural positions.
        shift_by = (subtask_end_index - subtask_start_index + 1).astype(jnp.int32)
        positions = positions.at[:, shift_start_index:].add(-shift_by[:, None])
    return positions


def compute_suffix_positions(
    prefix_mask: jax.Array,
    suffix_mask: jax.Array,
    *,
    subtask_start_index: jax.Array | None = None,
    subtask_end_index: jax.Array | None = None,
) -> jax.Array:
    """RoPE positions for the action+CLS suffix in the KV-cache inference path.

    Continues from the prefix's last position, then subtracts the subtask
    length so action_0 lands at (shifted state position) + 1.
    """
    positions = (
        jnp.sum(prefix_mask.astype(jnp.int32), axis = -1)[:, None]
        + jnp.cumsum(suffix_mask.astype(jnp.int32), axis = -1) - 1
    )
    if subtask_start_index is not None and subtask_end_index is not None:
        # subtask_end_index points at the trailing "\n" _tokenize_subtask_prompt
        # appends after the subtask at train time (append_newline=True; the newline is part
        # of the next-token objective), so the inclusive span end - start + 1 covers the
        # subtask tokens plus that newline. This shift is reached only by the non-AR critic;
        # the predict_subtask_ar critic passes subtask_start/end_index as None (via
        # _position_shift_indices), keeping the subtask at its natural positions.
        shift_by = (subtask_end_index - subtask_start_index + 1).astype(jnp.int32)
        positions = positions - shift_by[:, None]
    return positions


@dataclasses.dataclass(frozen=True)
class PaliGemmaNetworkConfig:
    """Configuration for PaliGemma-based value network.

    This network uses a pre-trained PaliGemma backbone to encode images,
    text, and proprioceptive state for V(s) or Q(s,a) estimation.

    Sequence structure for V(s):
        [img1_patches] [img2_patches] [img3_patches] [text_tokens] [state_embed] [CLS]

    Sequence structure for Q(s,a) (when action_conditioned=True):
        [img1_patches] [img2_patches] [img3_patches] [text_tokens] [state_embed] [action_tokens] [CLS]

    Where:
    - action_tokens: action_horizon embedded action vectors (one token per action)

    The network outputs the CLS token embedding which is
    passed directly to the value head for final prediction.
    """

    # Proprioceptive state dimension
    state_dim: int

    # Number of camera images (default 3 for RoboCOIN)
    num_cameras: int = 3

    # Input image resolution (should match PaliGemma training: 224x224)
    image_size: tuple[int, int] = (224, 224)

    # Maximum token length for text prompts
    max_token_len: int = 48

    # PaliGemma variant (matches pi0 config)
    paligemma_variant: str = "gemma_2b"

    # Dtype for computations
    dtype: str = "bfloat16"

    # Action dimension (required when action_horizon is provided)
    action_dim: int = 14

    # Whether to mask out the state token in the attention mask (for ablation studies)
    no_state: bool = False

    # Fix order in which to iterate through keys
    image_keys: tuple[str, str, str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # Apply ``flax.nnx.LayerNorm`` (with learnable scale + bias) to CLS features
    # before the value head.
    use_layernorm: bool = False

    # When True (and subtask_start/end_index are present in the data), the subtask
    # suffix stays visible to state/action/CLS queries and keeps its natural RoPE
    # positions: the suffix-blocking attention constraint and the position shift
    # are both skipped, while the suffix remains causal so the next-token
    # objective still applies. When False, value features are attention- and
    # position-invariant to the subtask (current behavior).
    predict_subtask_ar: bool = False

    def get_tokenizer(self, max_len: int | None = None):
        """Return the appropriate text tokenizer for this variant."""
        from openpi.models.tokenizer import PaligemmaTokenizer

        if max_len is None:
            max_len = self.max_token_len
        return PaligemmaTokenizer(max_len = max_len)

    def create(self, rng: at.KeyArrayLike, action_horizon: int | None = None) -> PaliGemmaValueNetwork:
        """Create a new PaliGemma value network with initialized parameters."""
        return PaliGemmaValueNetwork(self, rngs = nnx.Rngs(rng), action_horizon = action_horizon)


class PaliGemmaValueNetwork(BaseValueNetwork):
    """PaliGemma-based value network for V(s, l) or Q(s, a, l) estimation.

    Architecture (following pi0.py initialization pattern):
    1. SigLIP ViT for image encoding -> 256 patches per image
    2. Gemma for text embedding + LLM processing
    3. Linear projection for state -> 1 embedding
    4. (If action_conditioned) Linear projection for each action -> action_horizon embeddings
    5. Learnable CLS token appended to sequence
    6. Forward through Gemma LLM with custom attention mask
    7. Extract CLS token output (last position) -> value head

    Token sequence for V(s) (example with 3 images):
        [256 patches img1] [256 patches img2] [256 patches img3] [L text tokens] [1 state token] [CLS]

    Token sequence for Q(s,a) (example with 3 images, action_horizon=H):
        [256 patches img1] [256 patches img2] [256 patches img3] [L text tokens] [1 state token] [H action tokens] [CLS]

    Attention Design for Q(s,a):
    - Image and text tokens: bidirectional with each other
    - State token: can attend to images/text, but images/text cannot attend to it
    - Action tokens: can attend to images/text/state, but state cannot attend to actions
    - CLS token (last): can attend to all, but others cannot attend to it
    """

    def __init__(self, config: PaliGemmaNetworkConfig, rngs: nnx.Rngs, action_horizon: int | None = None):
        super().__init__()

        self.config = config
        self._action_conditioned = action_horizon is not None
        self._num_cameras = config.num_cameras
        self._max_token_len = config.max_token_len
        self._action_horizon = action_horizon if action_horizon is not None else 0
        self._action_dim = config.action_dim
        self._no_state = config.no_state
        self._image_keys = config.image_keys
        self._image_size = config.image_size
        self._predict_subtask_ar = config.predict_subtask_ar

        logger.info(
            "PaliGemmaValueNetwork: variant=%s, action_conditioned=%s, action_horizon=%s, no_state=%s, "
            "use_layernorm=%s, predict_subtask_ar=%s",
            config.paligemma_variant, self._action_conditioned, self._action_horizon, self._no_state,
            config.use_layernorm, config.predict_subtask_ar,
        )

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        embed_dim = paligemma_config.width

        # Initialize Gemma LLM (single config, no action expert).
        llm = nnx_bridge.ToNNX(_gemma.Module(configs = [paligemma_config], embed_dtype = config.dtype, adarms = False))
        llm.lazy_init(rngs = rngs, method = "init", use_adarms = [False])

        # Initialize image encoder: SigLIP with num_classes=embed_dim, applying the projection
        # in the SigLIP head Dense layer.
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes = paligemma_config.width,
                variant = "So400m/14",
                pool_type = "none",
                scan = True,
                dtype_mm = config.dtype,
            )
        )
        self._num_soft_tokens_per_image = NUM_PATCHES_PER_IMAGE
        # Initialize with a fake image
        fake_image = jnp.zeros((1, config.image_size[0], config.image_size[1], 3), dtype=jnp.float32)
        img.lazy_init(fake_image, train=False, rngs=rngs)

        # Store as PaliGemma dict (matches weight loader key structure)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # CLS token - learnable embedding for value extraction
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02)

        # State projection: state_dim -> embed_dim (single token). Always initialised
        # so the parameter tree (and any pretrained checkpoint that includes
        # `state_proj`) stays identical regardless of `no_state`; usage of the
        # projection at forward time is gated on `self._no_state` instead.
        self.state_proj = nnx.Linear(config.state_dim, embed_dim, rngs=rngs)

        # Action projection: action_dim -> embed_dim (one token per action in chunk)
        self.action_proj: nnx.Linear | None = None
        if self._action_conditioned:
            self.action_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)

        # Feature dimension is the Gemma embedding dimension
        self._feature_dim = embed_dim
        self._embed_dim = embed_dim

        self._use_layernorm = config.use_layernorm
        self.cls_layer_norm = nnx.LayerNorm(embed_dim, rngs = rngs) if config.use_layernorm else None



    @property
    def action_conditioned(self) -> bool:
        """V(s) is not action-conditioned."""
        return self._action_conditioned

    @property
    @override
    def feature_dim(self) -> int:
        """Return the output feature dimension (Gemma embed_dim)."""
        return self._feature_dim

    def _suffix_blocking_mask(self, suffix_mask: jax.Array | None) -> jax.Array | None:
        """Suffix mask used to hide subtask keys from non-suffix queries.

        With predict_subtask_ar the subtask stays visible to state/action/CLS,
        so no blocking mask is applied.
        """
        return None if self._predict_subtask_ar else suffix_mask

    def _position_shift_indices(
        self, observation: _model.Observation,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """Subtask indices passed to the RoPE position helpers.

        With predict_subtask_ar the subtask keeps its natural positions, so the
        shift is disabled by passing None.
        """
        if self._predict_subtask_ar:
            return None, None
        return observation.subtask_start_index, observation.subtask_end_index

    def _embed_sequence(
        self,
        observation: _model.Observation,
        action: jax.Array | None = None,
        action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, int]:
        """Embed full sequence: images + text + state + [actions] + CLS.

        Args:
            observation: Observation with images, state, and tokenized_prompt.
            action: Optional action chunk [B, action_horizon, action_dim] for Q(s,a).
            action_mask: Optional mask [B, action_horizon] for valid actions in chunk.

        Returns:
            Tuple of (tokens, input_mask, ar_mask, suffix_mask, shift_start_index)
            - tokens: [B, seq_len, embed_dim]
            - input_mask: [B, seq_len] bool
            - ar_mask: [seq_len] or [B, seq_len] bool where 0=bidirectional
              (images, prefix text), 1=causal (subtask text, state, actions, CLS)
            - suffix_mask: [B, seq_len] bool or None — True for subtask text positions
            - shift_start_index: column index of the first post-text token (state, or
              first action/CLS if no_state). Used by compute_rope_positions to keep
              state/action/CLS RoPE positions independent of subtask presence.
        """
        tokens, input_mask, ar_mask = self._embed_prefix(observation)
        batch_size = observation.state.shape[0]

        # Column index of the first post-text token (state, or first action/CLS if no_state).
        shift_start_index = self._num_cameras * NUM_PATCHES_PER_IMAGE
        if observation.tokenized_prompt is not None:
            shift_start_index += observation.tokenized_prompt.shape[1]

        # 4. Add action tokens if action_conditioned
        if self._action_conditioned:
            action_tokens = self.action_proj(action)  # [B, action_horizon, embed_dim]
            tokens.append(action_tokens)
            input_mask.append(action_mask)

            ar_mask.append(True)
            ar_mask += [False] * (self._action_horizon - 1)

        # 5. CLS token (last in sequence)
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, self._embed_dim))
        tokens.append(cls_tokens)
        input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
        ar_mask.append(True)

        tokens = jnp.concatenate(tokens, axis = 1)
        input_mask = jnp.concatenate(input_mask, axis = 1)
        ar_mask = jnp.array(ar_mask)

        # Mark subtask text tokens as causal in ar_mask. The cumsum logic in
        # make_attn_mask then ensures suffix tokens are causal among themselves
        # and invisible to prefix tokens.
        suffix_mask = None
        if observation.subtask_start_index is not None and observation.tokenized_prompt is not None:
            text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
            text_len = observation.tokenized_prompt.shape[1]
            seq_positions = jnp.arange(ar_mask.shape[0])
            global_subtask_start = text_start + observation.subtask_start_index  # [B]
            suffix_mask = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
                seq_positions[None, :] < text_start + text_len
            )
            ar_mask = ar_mask[None, :] | suffix_mask  # [B, N]

        return tokens, input_mask, ar_mask, suffix_mask, shift_start_index

    def _embed_prefix(
        self,
        observation: _model.Observation,
    ) -> tuple[list[jax.Array], list[jax.Array], list[bool]]:
        """Embed the shared observation prefix: images + text + state.

        Returns all three outputs as plain Python lists (not concatenated) so callers
        can keep appending before a single final concatenation / jnp.array conversion.
        """
        batch_size = observation.state.shape[0]

        input_mask: list[jax.Array] = []
        ar_mask: list[bool] = []
        tokens: list[jax.Array] = []

        # 1. Embed images (following pi0.py pattern)
        for name in self._image_keys:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train = False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens: bidirectional
            ar_mask += [False] * image_tokens.shape[1]

        # 2. Add language (tokenized inputs)
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method = "embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 3. Add state token (skip entirely when no_state is True)
        if not self._no_state:
            state_token = self.state_proj(observation.state)[:, None, :]  # [B, 1, embed_dim]
            tokens.append(state_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
            # State uses ar_mask=True (causal - can attend to images/text but not be attended)
            ar_mask.append(True)

        return tokens, input_mask, ar_mask


    def decode(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        return self.PaliGemma.llm(x, method = "decode")


    def _next_token_outputs(
        self,
        output: jax.Array,
        observation: _model.Observation,
    ) -> dict[str, jax.Array]:
        if observation.subtask_start_index is None:
            raise ValueError("subtask_start_index is required for next-token outputs.")
        if observation.subtask_end_index is None:
            raise ValueError("subtask_end_index is required for next-token outputs.")
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("tokenized_prompt and tokenized_prompt_mask are required for next-token outputs.")

        text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
        prompt_tokens = observation.tokenized_prompt
        prompt_mask = observation.tokenized_prompt_mask
        subtask_start_index = observation.subtask_start_index
        subtask_end_index = observation.subtask_end_index

        text_len = prompt_tokens.shape[1]
        text_hidden = output[:, text_start : text_start + text_len, :]

        candidate_embeddings = text_hidden[:, :-1, :]
        candidate_targets = prompt_tokens[:, 1:]
        candidate_mask = prompt_mask[:, :-1] & prompt_mask[:, 1:]

        first_predict_pos = jnp.maximum(subtask_start_index - 1, 0)
        last_predict_pos = subtask_end_index - 1
        window_offsets = jnp.arange(MAX_SUBTASK_TOKENS_FOR_LOSS, dtype = first_predict_pos.dtype)[None, :]
        predict_positions = first_predict_pos[:, None] + window_offsets
        max_candidate_pos = candidate_targets.shape[1] - 1
        gather_positions = jnp.clip(predict_positions, 0, max_candidate_pos)
        valid_positions = (
            (predict_positions <= last_predict_pos[:, None])
            & (predict_positions <= max_candidate_pos)
            & jnp.take_along_axis(candidate_mask, gather_positions, axis = 1)
        )
        return {
            "next_token_embeddings": jnp.take_along_axis(candidate_embeddings, gather_positions[:, :, None], axis = 1),
            "next_token_targets": jnp.take_along_axis(candidate_targets, gather_positions, axis = 1),
            "next_token_mask": valid_positions,
        }

    @override
    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> (
        at.Float[at.Array, "*b feature_dim"]
        | tuple[at.Float[at.Array, "*b feature_dim"], dict[str, at.Array]]
        | tuple[at.Float[at.Array, "*b feature_dim"], at.Float[at.Array, "*b _n"]]
    ):
        """Compute features from observation (and optionally action) for value prediction.

        Args:
            observation: Observation containing images, image_masks, state,
                        tokenized_prompt, and tokenized_prompt_mask.
                        Images should already be in [-1, 1] range.
            action: Optional _model.Actions with:
                    - actions: [B, action_horizon, action_dim] action chunk
                    If action_conditioned=True, this should be provided.
            rng: Optional random key for image augmentation. If provided,
                 enables augmentation during training.

        Returns:
            When rng is not None (training): features of shape [batch, embed_dim].
            When rng is None (inference): (features, attn_scores) where attn_scores is
                [batch, n_modalities] = mean over Gemma layers of CLS attention, grouped
                by modality (img1..imgN, text, state, [actions if Q]).
        """
        # Preprocess observation (handles resizing, default masks, augmentation)
        # train=True enables augmentation when rng is provided
        train = rng is not None
        if prefix_cache is None:
            observation = _model.preprocess_observation(rng, observation, train = train, image_resolution = self._image_size)

        # Extract action array and mask if action_conditioned
        action_array = None
        action_mask_array = None
        if self._action_conditioned:
            action_array = action  # [B, action_horizon, action_dim]
            assert action_array.shape[1] == self._action_horizon
            action_mask_array = observation.action_mask  # [B, action_horizon] or None

        if prefix_cache is not None:
            if train:
                raise ValueError("prefix_cache is only supported for inference.")
            if not self._action_conditioned:
                raise ValueError("prefix_cache is only supported for action-conditioned networks.")

            kv_cache, prefix_mask, subtask_mask = prefix_cache

            suffix_tokens = []
            suffix_mask = []
            suffix_ar_mask = []

            action_tokens = self.action_proj(action_array)
            suffix_tokens.append(action_tokens)
            suffix_mask.append(action_mask_array)
            suffix_ar_mask.append(True)
            suffix_ar_mask += [False] * (self._action_horizon - 1)

            cls_tokens = jnp.broadcast_to(self.cls_token.value, (action_array.shape[0], 1, self._embed_dim))
            suffix_tokens.append(cls_tokens)
            suffix_mask.append(jnp.ones((action_array.shape[0], 1), dtype = jnp.bool_))
            suffix_ar_mask.append(True)

            suffix_tokens = jnp.concatenate(suffix_tokens, axis = 1)
            suffix_mask = jnp.concatenate(suffix_mask, axis = 1)
            suffix_ar_mask = jnp.array(suffix_ar_mask)

            suffix_to_suffix = make_attn_mask(suffix_mask, suffix_ar_mask)
            suffix_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s = suffix_tokens.shape[1])
            # Block actions / CLS from attending to subtask-text positions in the
            # cached prefix, matching the suffix_mask blocking on the full-forward
            # path. `subtask_mask` was computed once in compute_prefix_cache.
            blocked_subtask_keys = self._suffix_blocking_mask(subtask_mask)
            if blocked_subtask_keys is not None:
                suffix_to_prefix = suffix_to_prefix & ~blocked_subtask_keys[:, None, :]
            attn_mask = jnp.concatenate([suffix_to_prefix, suffix_to_suffix], axis = -1)
            shift_subtask_start, shift_subtask_end = self._position_shift_indices(observation)
            positions = compute_suffix_positions(
                prefix_mask, suffix_mask,
                subtask_start_index = shift_subtask_start,
                subtask_end_index = shift_subtask_end,
            )

            (output,), _ = self.PaliGemma.llm(
                [suffix_tokens],
                mask = attn_mask,
                positions = positions,
                kv_cache = kv_cache,
                adarms_cond = [None],
            )
            # Match the full-sequence path's CLS LayerNorm before the value head.
            cls_features = output[:, -1, :]
            if self._use_layernorm:
                cls_features = self.cls_layer_norm(cls_features)
            return cls_features

        # Build embeddings and attention mask
        tokens, input_mask, ar_mask, suffix_mask, shift_start_index = self._embed_sequence(
            observation, action = action_array, action_mask = action_mask_array
        )
        attn_mask = make_attn_mask(input_mask, ar_mask, suffix_mask = self._suffix_blocking_mask(suffix_mask))

        shift_subtask_start, shift_subtask_end = self._position_shift_indices(observation)
        positions = compute_rope_positions(
            input_mask,
            shift_start_index = shift_start_index,
            subtask_start_index = shift_subtask_start,
            subtask_end_index = shift_subtask_end,
        )

        if train:
            # Forward through LLM (single expert, no adarms)
            (output,), _ = self.PaliGemma.llm(
                [tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None],
            )
            # Extract CLS token output (last position) for value prediction
            cls_features = output[:, -1, :]  # [B, embed_dim]
            if self._use_layernorm:
                cls_features = self.cls_layer_norm(cls_features)
            if observation.subtask_start_index is not None:
                return cls_features, self._next_token_outputs(output, observation)
            return cls_features

        # Inference: also return per-modality CLS attention scores
        (output,), _, all_cls_attn = self.PaliGemma.llm(
            [tokens],
            mask = attn_mask,
            positions = positions,
            adarms_cond = [None],
            return_cls_attention_score_distribution = True,
        )
        cls_features = output[:, -1, :]  # [B, embed_dim]
        if self._use_layernorm:
            cls_features = self.cls_layer_norm(cls_features)

        # all_cls_attn: [B, L, S]; mean over L -> [B, S] -> group by modality -> [B, n_modalities]
        cls_attn_mean = all_cls_attn.mean(axis = 1)
        attn_scores = self._group_attn_scores(cls_attn_mean)

        return cls_features, attn_scores

    def compute_prefix_cache(
        self, observation: _model.Observation
    ) -> tuple[at.Array, at.Array, at.Array | None]:
        """Compute and return (kv_cache, prefix_mask, subtask_mask) for the prefix forward.

        ``subtask_mask`` is a ``[B, prefix_len]`` boolean over prefix columns that is
        True at subtask-text positions in the tokenized prompt; None when subtask
        boundaries aren't supplied. The suffix forward in ``compute_features``
        re-uses it to block actions / CLS from attending to subtask-text in the
        cached prefix, matching ``_embed_sequence``'s suffix_mask blocking on the
        full-forward path.
        """
        observation = _model.preprocess_observation(None, observation, train = False, image_resolution = self._image_size)

        prefix_tokens_list, prefix_mask_list, prefix_ar_mask_list = self._embed_prefix(observation)
        prefix_tokens = jnp.concatenate(prefix_tokens_list, axis = 1)
        prefix_mask = jnp.concatenate(prefix_mask_list, axis = 1)
        prefix_ar_mask = jnp.array(prefix_ar_mask_list)
        # Mirror _embed_sequence: when a subtask suffix is present in the prompt,
        # mark suffix-text positions as causal in ar_mask AND pass suffix_mask to
        # make_attn_mask so non-suffix tokens (images, prefix-text, state) can't
        # read suffix-text keys. Without this the cached prefix KVs encode a
        # different attention pattern than the full forward computes.
        suffix_mask = None
        if observation.subtask_start_index is not None and observation.tokenized_prompt is not None:
            text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
            text_len = observation.tokenized_prompt.shape[1]
            seq_positions = jnp.arange(prefix_ar_mask.shape[0])
            global_subtask_start = text_start + observation.subtask_start_index  # [B]
            suffix_mask = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
                seq_positions[None, :] < text_start + text_len
            )
            prefix_ar_mask = prefix_ar_mask[None, :] | suffix_mask  # [B, N]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, suffix_mask = self._suffix_blocking_mask(suffix_mask))
        # Same shift_start_index as _embed_sequence's original-PaliGemma branch.
        shift_start_index = self._num_cameras * NUM_PATCHES_PER_IMAGE
        if observation.tokenized_prompt is not None:
            shift_start_index += observation.tokenized_prompt.shape[1]
        shift_subtask_start, shift_subtask_end = self._position_shift_indices(observation)
        positions = compute_rope_positions(
            prefix_mask,
            shift_start_index = shift_start_index,
            subtask_start_index = shift_subtask_start,
            subtask_end_index = shift_subtask_end,
        )
        _, kv_cache = self.PaliGemma.llm([prefix_tokens], mask = prefix_attn_mask, positions = positions)
        return kv_cache, prefix_mask, suffix_mask

    def _group_attn_scores(self, cls_attn_mean: jax.Array) -> jax.Array:
        """Group per-position CLS attention [B, S] into per-modality scores [B, n_modalities].

        Modality order: [img1, img2, ..., imgN, text, (state if not no_state), (actions if Q)]
        The CLS self-attention at the last sequence position is
        excluded from all groups, so the per-modality scores sum to ~1 across
        groups (the small remainder is the CLS-self-attention probability).
        """
        attn_parts = []

        for i in range(self._num_cameras):
            start = i * NUM_PATCHES_PER_IMAGE
            end = start + NUM_PATCHES_PER_IMAGE
            attn_parts.append(cls_attn_mean[:, start : end].sum(axis = -1))

        text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
        text_end = text_start + self._max_token_len

        attn_parts.append(cls_attn_mean[:, text_start : text_end].sum(axis = -1))  # text
        if not self._no_state:
            attn_parts.append(cls_attn_mean[:, text_end])  # state (single token)
            action_start = text_end + 1
        else:
            action_start = text_end
        if self._action_conditioned:
            attn_parts.append(cls_attn_mean[:, action_start : action_start + self._action_horizon].sum(axis = -1))

        return jnp.stack(attn_parts, axis = -1)


# =============================================================================
# Inline self-test: python src/openpi/value_functions/networks/paligemma.py
# =============================================================================
if __name__ == "__main__":

    def _check(condition: bool, msg: str) -> None:
        if not condition:
            raise AssertionError(f"FAIL: {msg}")
        print(f"  PASS: {msg}")

    # Sequence layout used by both cases:
    #   [img (3)] [text (4)] [state (1)] [CLS (1)]   total = 9
    N_IMG, N_TEXT = 3, 4
    SEQ_LEN = N_IMG + N_TEXT + 2  # +state +CLS
    IMG = slice(0, N_IMG)
    TEXT = slice(N_IMG, N_IMG + N_TEXT)
    STATE = N_IMG + N_TEXT
    CLS = SEQ_LEN - 1

    # =========================================================================
    # Case 1 — no subtask (subtask_start_index is None)
    # =========================================================================
    print("Case 1: No subtask text")

    ar = jnp.array([False] * N_IMG + [False] * N_TEXT + [True, True])
    inp = jnp.ones((1, SEQ_LEN), dtype=jnp.bool_)
    m = make_attn_mask(inp, ar)[0]

    _check(m[IMG, IMG].all().item(), "img <-> img bidirectional")
    _check(m[IMG, TEXT].all().item(), "img -> text")
    _check(m[TEXT, IMG].all().item(), "text -> img")
    _check(m[TEXT, TEXT].all().item(), "text <-> text bidirectional")

    _check(m[STATE, IMG].all().item(), "state -> img")
    _check(m[STATE, TEXT].all().item(), "state -> text")
    _check(not m[IMG, STATE].any().item(), "img -/-> state")
    _check(not m[TEXT, STATE].any().item(), "text -/-> state")

    _check(m[CLS, :CLS].all().item(), "CLS -> all prior")
    _check(not m[:CLS, CLS].any().item(), "others -/-> CLS")

    # =========================================================================
    # Case 2 — with subtask text (batch=2, different subtask starts)
    # =========================================================================
    print("\nCase 2: With subtask text (batch=2, subtask starts at text offset 2 and 3)")

    N_PREFIX_A, N_SUFFIX_A = 2, 2  # batch element 0
    N_PREFIX_B, N_SUFFIX_B = 3, 1  # batch element 1

    inp2 = jnp.ones((2, SEQ_LEN), dtype=jnp.bool_)

    # Build per-element ar_mask [B, N]: suffix text positions + state + CLS are True.
    base_ar = jnp.array([False] * N_IMG + [False] * N_TEXT + [True, True])  # [N]
    subtask_start_index = jnp.array([N_PREFIX_A, N_PREFIX_B])  # within text region
    text_start = N_IMG
    seq_positions = jnp.arange(SEQ_LEN)
    global_subtask_start = text_start + subtask_start_index
    suffix = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
        seq_positions[None, :] < text_start + N_TEXT
    )
    ar2 = base_ar[None, :] | suffix  # [2, N]

    m2 = make_attn_mask(inp2, ar2, suffix_mask=suffix)

    for b, (n_pre, n_suf) in enumerate([(N_PREFIX_A, N_SUFFIX_A), (N_PREFIX_B, N_SUFFIX_B)]):
        tag = f"[b={b}, prefix={n_pre}, suffix={n_suf}]"
        mb = m2[b]

        prefix_text = slice(N_IMG, N_IMG + n_pre)
        suffix_text = slice(N_IMG + n_pre, N_IMG + n_pre + n_suf)

        _check(mb[IMG, IMG].all().item(), f"{tag} img <-> img bidirectional")
        _check(mb[IMG, prefix_text].all().item(), f"{tag} img -> prefix_text")
        _check(mb[prefix_text, IMG].all().item(), f"{tag} prefix_text -> img")
        _check(mb[prefix_text, prefix_text].all().item(), f"{tag} prefix_text <-> prefix_text bidirectional")

        _check(not mb[prefix_text, suffix_text].any().item(), f"{tag} prefix_text -/-> suffix_text")
        _check(not mb[IMG, suffix_text].any().item(), f"{tag} img -/-> suffix_text")

        _check(mb[suffix_text, prefix_text].all().item(), f"{tag} suffix_text -> prefix_text")
        _check(mb[suffix_text, IMG].all().item(), f"{tag} suffix_text -> img")

        # Suffix tokens are causal among themselves.
        for qi in range(n_suf):
            for ki in range(n_suf):
                q_abs = N_IMG + n_pre + qi
                k_abs = N_IMG + n_pre + ki
                if ki <= qi:
                    _check(mb[q_abs, k_abs].item(), f"{tag} suffix[{qi}] -> suffix[{ki}] (causal ok)")
                else:
                    _check(not mb[q_abs, k_abs].item(), f"{tag} suffix[{qi}] -/-> suffix[{ki}] (causal block)")

        _check(mb[STATE, IMG].all().item(), f"{tag} state -> img")
        _check(mb[STATE, prefix_text].all().item(), f"{tag} state -> prefix_text")
        _check(not mb[STATE, suffix_text].any().item(), f"{tag} state -/-> suffix_text")

        _check(mb[CLS, IMG].all().item(), f"{tag} CLS -> img")
        _check(mb[CLS, prefix_text].all().item(), f"{tag} CLS -> prefix_text")
        _check(mb[CLS, STATE].item(), f"{tag} CLS -> state")
        _check(not mb[CLS, suffix_text].any().item(), f"{tag} CLS -/-> suffix_text")

        _check(not mb[:CLS, CLS].any().item(), f"{tag} others -/-> CLS")

    print("\nAll tests passed!")
