"""Reusable subtask-token decoder for PaliGemma value-function critics.

Extracted from `SubtaskPredictorPolicy` so any policy that already holds a
loaded PaliGemma critic (e.g. `BestOfNPolicy`) can optionally decode the current
subtask off the same observation, without re-forking the whole policy class.

Given a critic `Observation` (images + critic-tokenized prompt), the critic
autoregressively decodes up to `max_tokens` subtask tokens via a KV-cache fast
path that mirrors the training-time prefix attention pattern: the prefix is cached
once and a fixed-length suffix is re-forwarded against it on every step, so the
decode step compiles once.

The decode dominates per-call latency (~14s on L40S), so callers gate it with
`should_decode(force=...)`: it runs only once every `decode_every` calls, or
immediately when `force=True` (e.g. to decode a terminal frame on demand).
"""

from __future__ import annotations

import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model

logger = logging.getLogger(__name__)


def critic_value_network(critic_model: Any) -> Any:
    """Resolve the PaliGemma value network used for subtask decoding.

    SARSAValueFunction names the backbone `network`; CQLValueFunction exposes it
    as `q_network`. Accept either so the decoder works with both critic flavors.
    """
    net = getattr(critic_model, "network", None)
    if net is None:
        net = critic_model.q_network
    return net


class SubtaskDecoder:
    """Autoregressively decodes the current subtask from a loaded critic.

    The decode reads the task-description prefix, so a task description is required on
    every call (``needs_task_description``).

    Args:
        critic_model: A loaded value-function model whose backbone is a
            PaliGemma value network (resolved via `critic_value_network`).
        critic_tokenizer: The critic's tokenizer (used to decode token ids back
            to text and to build the early-exit target sequences).
        decode_every: Run the decode only once every N `should_decode()` calls.
            Use a large value to keep per-infer latency low; `force=True`
            overrides it for on-demand decodes.
        max_tokens: Max subtask tokens to decode per call. Decoding stops at the
            trailing "\n" (the model's learned terminator, emitted after the
            subtask) or once `max_tokens` is reached, whichever comes first.
    """

    needs_task_description: bool = True

    def __init__(
        self,
        critic_model: Any,
        critic_tokenizer: Any,
        *,
        decode_every: int = 20,
        max_tokens: int = 16,
    ) -> None:
        self._critic_model = critic_model
        self._critic_tokenizer = critic_tokenizer
        self.decode_every = int(decode_every)
        self.max_tokens = int(max_tokens)
        self._call_count = 0
        self._eos_id = self._resolve_eos_id()
        self._newline_id = self._resolve_newline_id()
        self._build_closures()

    # ------------------------------------------------------------------ API
    def should_decode(self, *, force: bool = False) -> bool:
        """Decide whether to decode on this call, advancing the cadence counter.

        Returns True when `force` is set or the per-call counter hits a multiple
        of `decode_every`. Call exactly once per infer so the cadence advances
        whether or not the decode actually runs.
        """
        decode = force or (self.decode_every > 0 and self._call_count % self.decode_every == 0)
        self._call_count += 1
        return decode

    def predict(self, critic_observation: _model.Observation) -> dict[str, Any]:
        """Decode the subtask; returns the fields to merge into an infer result."""
        tokens, text, perplexity = self._predict_tokens(critic_observation)
        logger.info(f"[subtask_predict] tokens={tokens} decoded={text!r} perplexity={perplexity:.4f}")
        return {
            "predicted_subtask": text,
            "predicted_subtask_tokens": list(map(int, tokens)),
            "subtask_perplexity": perplexity,
        }

    def run_lockstep(self, critic_observation: _model.Observation) -> None:
        """Drive the same JIT chain on a participating (non-rank-0) host.

        Each `predict` call is a sequence of cross-host JIT collectives; on
        multi-host setups every rank must run them in lockstep or the collectives
        desync. Outputs are discarded here. No-op effect on single-host.
        """
        self._predict_tokens(critic_observation)

    # ------------------------------------------------------------- internals
    def _resolve_eos_id(self) -> int | None:
        eos = getattr(self._critic_tokenizer, "eos_token_id", None)
        if eos is None:
            inner = getattr(self._critic_tokenizer, "tokenizer", None)
            eos = inner.eos_id() if inner is not None and hasattr(inner, "eos_id") else None
        return eos

    def _resolve_newline_id(self) -> int | None:
        # The critic is trained to emit the trailing "\n" appended after the
        # subtask (NTP through the newline), so "\n" is the decode terminator.
        inner = getattr(self._critic_tokenizer, "_tokenizer", None)
        if inner is None:
            return None
        ids = inner.encode("\n")
        return int(ids[-1]) if ids else None

    def _build_closures(self) -> None:
        from openpi.value_functions.networks.paligemma import NUM_PATCHES_PER_IMAGE
        from openpi.value_functions.networks.paligemma import compute_rope_positions
        from openpi.value_functions.networks.paligemma import make_attn_mask

        @nnx.jit
        def _prefix_forward(critic_model, observation):
            # gemma_2b path: replicates `compute_prefix_cache` with
            # suffix_mask=None (no subtask suffix in the prompt yet) so
            # images + task-description text get full bidirectional
            # attention exactly as at training time.
            n = critic_value_network(critic_model)
            obs = _model.preprocess_observation(
                None, observation, train = False, image_resolution = n._image_size,
            )
            prefix_tokens_list, prefix_mask_list, prefix_ar_mask_list = n._embed_prefix(obs)
            prefix_tokens = jnp.concatenate(prefix_tokens_list, axis = 1)
            prefix_mask = jnp.concatenate(prefix_mask_list, axis = 1)
            prefix_ar_mask = jnp.array(prefix_ar_mask_list)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, suffix_mask = None)
            text_start = n._num_cameras * NUM_PATCHES_PER_IMAGE
            text_len = obs.tokenized_prompt.shape[1]
            positions = compute_rope_positions(
                prefix_mask,
                shift_start_index = text_start + text_len,
                subtask_start_index = None,
                subtask_end_index = None,
            )
            (hidden,), kv_cache = n.PaliGemma.llm(
                [prefix_tokens], mask = prefix_attn_mask, positions = positions,
            )
            last_text_pos = text_start + jnp.sum(
                obs.tokenized_prompt_mask.astype(jnp.int32), axis = 1,
            ) - 1
            last_hidden = jnp.take_along_axis(
                hidden, last_text_pos[:, None, None], axis = 1,
            )
            return last_hidden, kv_cache, prefix_mask, last_text_pos

        @nnx.jit
        def _decode_suffix(critic_model, suffix_token_ids, num_emitted, kv_cache, prefix_mask, last_text_pos):
            # JIT'd decode. Re-forwards the FIXED-length [1, max_tokens]
            # padded suffix against the cached prefix on every step: feeding a
            # constant number of suffix tokens keeps the concatenated cache
            # (prefix_len + max_tokens) at a constant shape, so this compiles
            # ONCE and is reused for all steps. The old one-token path grew the
            # cache by 1 each call, so a JIT'd version would recompile per step —
            # which is why it ran eagerly (~1.85 s/token of op-by-op dispatch).
            # `num_emitted` is traced (it only drives the key-validity mask), so
            # it never triggers a recompile. The short suffix k/v are recomputed
            # each step (cheap for <=16 tokens); the expensive prefix stays cached.
            n = critic_value_network(critic_model)
            max_tokens = suffix_token_ids.shape[1]
            prefix_len = prefix_mask.shape[1]
            suffix_embeds = n.PaliGemma.llm(suffix_token_ids, method = "embed")
            to_prefix = jnp.broadcast_to(prefix_mask[:, None, :], (1, max_tokens, prefix_len))
            causal = jnp.tril(jnp.ones((max_tokens, max_tokens), dtype = jnp.bool_))
            valid_keys = (jnp.arange(max_tokens) < num_emitted)[None, :]
            to_suffix = (causal & valid_keys)[None, :, :]
            attn_mask = jnp.concatenate([to_prefix, to_suffix], axis = -1)
            positions = last_text_pos[:, None] + 1 + jnp.arange(max_tokens)[None, :]
            (hidden,), _ = n.PaliGemma.llm(
                [suffix_embeds], mask = attn_mask, positions = positions, kv_cache = kv_cache,
            )
            # Decode ONLY the last emitted position (num_emitted-1) and pick the
            # next token + its log-prob inside the JIT, so the per-step host
            # transfer is two scalars (no eager argmax / log_softmax over the
            # full vocab and no full-logits materialisation).
            last_hidden = jax.lax.dynamic_index_in_dim(
                hidden[0], num_emitted - 1, axis = 0, keepdims = True,
            )  # [1, embed_dim]
            logits_row = n.decode(last_hidden[None])[0, 0]  # [vocab]
            next_id = jnp.argmax(logits_row).astype(jnp.int32)
            chosen_logp = jax.nn.log_softmax(logits_row)[next_id]
            return next_id, chosen_logp

        @nnx.jit
        def _logits_from_hidden(critic_model, hidden):
            return critic_value_network(critic_model).decode(hidden)

        self._prefix_forward = _prefix_forward
        self._decode_suffix = _decode_suffix
        self._logits = _logits_from_hidden

    def _predict_tokens(self, critic_obs: _model.Observation) -> tuple[list[int], str, float]:
        last_hidden, kv_cache, prefix_mask, last_text_pos = self._prefix_forward(self._critic_model, critic_obs)
        logits = self._logits(self._critic_model, last_hidden)
        # log-softmax + select chosen token's log-probability for the
        # autoregressive perplexity accumulator.
        log_probs = jax.nn.log_softmax(logits[0, 0])
        next_tok = int(np.asarray(jnp.argmax(logits[0, 0])))
        chosen_logp = float(np.asarray(log_probs[next_tok]))
        predicted: list[int] = [next_tok]
        total_neg_logp: float = -chosen_logp
        # Host-side suffix buffer (filled as tokens are emitted, fed to the fixed-shape
        # jitted `_decode_suffix`).
        suffix_ids = np.zeros((1, self.max_tokens), dtype = np.int32)
        for k in range(1, self.max_tokens):
            # Stop only at the trailing "\n" (the model's learned terminator) or
            # once max_tokens is reached. The "\n" is kept in `predicted` (so the
            # per-token cost counts it) and stripped from the decoded text below.
            if next_tok == self._newline_id or (self._eos_id is not None and next_tok == self._eos_id):
                break
            # Feed the k emitted tokens (slots 0..k-1); the jitted step reads
            # the hidden at the last emitted position (k-1) and returns the
            # next token + its log-prob (both scalars).
            suffix_ids[0, k - 1] = predicted[k - 1]
            next_id, logp = self._decode_suffix(
                self._critic_model, jnp.asarray(suffix_ids), jnp.asarray(k, dtype = jnp.int32),
                kv_cache, prefix_mask, last_text_pos,
            )
            next_tok = int(np.asarray(next_id))
            chosen_logp = float(np.asarray(logp))
            total_neg_logp += -chosen_logp
            predicted.append(next_tok)
        if hasattr(self._critic_tokenizer, "decode"):
            decoded = self._critic_tokenizer.decode(predicted)
        else:
            inner = getattr(self._critic_tokenizer, "tokenizer", None)
            decoded = inner.decode(predicted) if inner is not None else " ".join(map(str, predicted))
        # Strip the trailing "\n" (and any surrounding whitespace) the model
        # emits as its terminator — the value prompt re-adds a clean "\n" via
        # `append_newline=True`, so keeping it here would double the newline.
        decoded = str(decoded).strip()
        # PP = exp(-(1/T) * sum_t log p(x_t | x_<t)); using the argmax token at
        # each step (greedy decode), so this is the perplexity of the model's
        # own most-confident path conditioned on the current observation.
        n_tokens = max(1, len(predicted))
        perplexity = float(np.exp(total_neg_logp / n_tokens))
        return predicted, decoded, perplexity


class CategoricalSubtaskPredictor:
    """Predicts the current subtask id from a critic with a categorical subtask head.

    The ResNet sibling of ``SubtaskDecoder`` with the same ``predict`` / ``run_lockstep``
    surface: no text pathway, so the id is the argmax of the network's
    ``predict_subtask_id`` over the images alone, mapped back to text through the network
    config's ``subtask_vocab``. The caller feeds the cached prediction back as
    ``observation.subtask_id`` (``vocab_index``), the analog of the AR path's prompt suffix.
    Reads no prompt, so no task description is needed.

    Args:
        critic_model: A loaded value-function model whose backbone declares
            ``uses_subtask_id`` (resolved via ``critic_value_network``).
    """

    needs_task_description: bool = False

    def __init__(self, critic_model: Any) -> None:
        network = critic_value_network(critic_model)
        config = getattr(network, "config", None)
        if not getattr(config, "uses_subtask_id", False):
            raise ValueError(
                f"{type(network).__name__} has no categorical subtask head "
                "(uses_subtask_id is False); CategoricalSubtaskPredictor does not apply."
            )
        self._critic_model = critic_model
        self._vocab: tuple[str, ...] = tuple(config.subtask_vocab or ())

        @nnx.jit
        def _predict(critic_model, observation):
            return critic_value_network(critic_model).predict_subtask_id(observation)

        self._predict = _predict

    def predict(self, critic_observation: _model.Observation) -> dict[str, Any]:
        """Predict the subtask id; returns the fields to merge into an infer result."""
        subtask_id = int(np.asarray(self._predict(self._critic_model, critic_observation))[0])
        text = self._vocab[subtask_id] if subtask_id < len(self._vocab) else f"subtask_{subtask_id}"
        logger.info(f"[subtask_predict] id={subtask_id} subtask={text!r}")
        return {
            "predicted_subtask": text,
            "predicted_subtask_id": subtask_id,
        }

    def run_lockstep(self, critic_observation: _model.Observation) -> None:
        """Drive the same JIT call on a participating (non-rank-0) host; output discarded."""
        self._predict(self._critic_model, critic_observation)

    def vocab_index(self, subtask: str) -> int:
        """Categorical id of a subtask string this predictor produced."""
        return self._vocab.index(subtask)


SubtaskPredictor = SubtaskDecoder | CategoricalSubtaskPredictor


def build_subtask_predictor(
    critic_model: Any,
    critic_tokenizer: Any,
    *,
    decode_every: int,
    max_tokens: int,
) -> SubtaskPredictor | None:
    """The subtask predictor a critic's network calls for, or None when it predicts none.

    ``predict_subtask_ar`` networks decode text through the tokenizer; ``uses_subtask_id``
    networks classify. A network with neither head conditions on no predicted subtask.
    """
    config = getattr(critic_value_network(critic_model), "config", None)
    if getattr(config, "predict_subtask_ar", False):
        return SubtaskDecoder(critic_model, critic_tokenizer, decode_every = decode_every, max_tokens = max_tokens)
    if getattr(config, "uses_subtask_id", False):
        return CategoricalSubtaskPredictor(critic_model)
    return None
