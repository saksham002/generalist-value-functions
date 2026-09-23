import dataclasses
import logging
import os
import pickle

logger = logging.getLogger(__name__)

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
import openpi.value_functions.base_value_functions as _value_fn
from openpi.value_functions.networks.base_networks import BaseValueNetwork
from openpi.value_functions.networks.paligemma import NUM_PATCHES_PER_IMAGE
from openpi.value_functions.networks.paligemma import compute_rope_positions
from openpi.value_functions.networks.paligemma import make_attn_mask

RLDS_TO_STANDARD_CAMERA_MAP = {
    "cam_0": "base_0_rgb",
    "cam_1": "left_wrist_0_rgb",
    "cam_2": "right_wrist_0_rgb",
}


@dataclasses.dataclass(frozen = True)
class SnapshotConfig:
    """Extra per-interval value snapshots for one validation episode.

    When set on the eval config, the non-subtask validation path renders, for
    each shade interval (inclusive ``[a, b]``), the critic's predicted values
    over the interval's subtask (blue line, no MC returns) with a light
    red/green shaded span, plus the snapshot camera frame at the interval
    midpoint. All outputs are keyed under the ``snapshot/`` section. The three
    parallel tuples are indexed together (one entry per interval).
    """

    episode_file: str
    shade_intervals: tuple[tuple[int, int], ...]
    shade_colours: tuple[str, ...]
    snapshot_camera: tuple[str, ...]

    def __post_init__(self):
        num_intervals = len(self.shade_intervals)
        if not (len(self.shade_colours) == num_intervals and len(self.snapshot_camera) == num_intervals):
            raise ValueError(
                "snapshot shade_intervals, shade_colours and snapshot_camera must be equal length; got "
                f"{num_intervals}, {len(self.shade_colours)}, {len(self.snapshot_camera)}."
            )
        for colour in self.shade_colours:
            if colour not in ("r", "g"):
                raise ValueError(f"snapshot shade_colours entries must be 'r' or 'g', got {colour!r}.")


def extract_embodiment(repo_id: str | bytes) -> str:
    """Extract embodiment name from a RoboCOIN repo identifier."""
    if isinstance(repo_id, bytes):
        repo_id = repo_id.decode("utf-8")
    return "_".join(repo_id.split("/")[-1].split("_", 2)[:2])


def decode_text(text: str | bytes | np.ndarray | np.generic) -> str:
    """Decode a scalar text-like value to a Python string."""
    if hasattr(text, "item"):
        text = text.item()
    if isinstance(text, bytes):
        return text.decode("utf-8")
    return str(text)


def swap_left_right_text(text: str) -> str:
    """Swap left/right mentions in a task string."""
    swapped = text.replace("left", "TEMP_LEFT_MARKER")
    swapped = swapped.replace("right", "left")
    return swapped.replace("TEMP_LEFT_MARKER", "right")


def generate_negative_subtask_text(subtask_text: str) -> str:
    """Generate the negative counterfactual text used for RoboCOIN validation."""
    if "Place the plate" in subtask_text:
        return "Place the plate on the dish rack"
    if "Grab the knife" in subtask_text:
        return "Grab the banana with your right hand"
    if "Place the knife" in subtask_text:
        return "Place the knife on the board"
    if "Pass the plate" in subtask_text:
        return "Rotate the plate with the right gripper"
    return swap_left_right_text(subtask_text)


def sample_random_actions(
    actions: np.ndarray,
    *,
    use_quantile_norm: bool,
    _traj_index: int | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    """Sample deterministic per-frame random normalized actions for validation."""
    clip_bound = 1.25 if use_quantile_norm else 5.0
    if _traj_index is None or frame_index is None:
        seed = 86
    else:
        seed = ((_traj_index + 1) * 1_000_003 + frame_index) % (2**32)
    rng = np.random.default_rng(seed = seed)
    return rng.uniform(-clip_bound, clip_bound, size = actions.shape).astype(actions.dtype, copy = False)


def detokenize_prompt(token_ids: np.ndarray, mask: np.ndarray) -> str:
    """Decode a token ID vector back to text using the PaliGemma SentencePiece model.

    Args:
        token_ids: 1D array of token IDs.
        mask: Boolean mask of valid (non-padding) positions.

    Returns:
        Decoded text string.
    """
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer()
    ids = np.asarray(token_ids)[np.asarray(mask, dtype = bool)]
    return tokenizer._tokenizer.decode(ids.tolist())


# Logged once (first prediction batch) so HBM can be read right after the first
# JIT compile — the point of a fast measurement loop is not to wait for the full run.
_HBM_LOGGED = False


@nnx.jit
def _jitted_compute_value(
    model_to_use: _value_fn.BaseValueFunction,
    obs: _model.Observation,
    act: _model.Actions | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Returns (values, attn_scores); attn_scores is None for networks without per-modality attention."""
    out = model_to_use.compute_value(obs, act)
    return out if isinstance(out, tuple) else (out, None)


@nnx.jit
def _jitted_compute_value_best_cached(
    model_to_use: _value_fn.BaseValueFunction,
    obs: _model.Observation,
    actions: _model.Actions,
) -> jnp.ndarray:
    """Per-frame value of the highest-value cached counterfactual action.

    ``actions`` is ``[b, num_samples, action_horizon, action_dim]``. The prefix
    KV cache (images + prompt + state) is computed once per frame, then all
    ``num_samples`` candidates are scored against the shared cache. Every candidate's
    value is returned (shape ``[b, num_samples]``) so callers can reduce with max or pick
    one at random; the npz analysis needs the full set.
    """
    from openpi.models.best_of_n import expand_observation

    num_samples = actions.shape[1]
    kv_cache, prefix_mask, subtask_mask = model_to_use.compute_prefix_cache(obs)
    expanded_obs = expand_observation(obs, num_samples)
    flat_actions = actions.reshape(actions.shape[0] * num_samples, actions.shape[2], actions.shape[3])
    # gemma_2b KV caches stack layers with batch at axis 1; networks with a plain feature cache
    # (e.g. ResNet image features) declare `prefix_cache_batch_axis`.
    network = getattr(model_to_use, "q_network", None) or getattr(model_to_use, "network", None)
    kv_batch_axis = getattr(network, "prefix_cache_batch_axis", 1)
    repeated_kv_cache = jax.tree.map(lambda x: jnp.repeat(x, num_samples, axis = kv_batch_axis), kv_cache)
    repeated_prefix_mask = jnp.repeat(prefix_mask, num_samples, axis = 0)
    repeated_subtask_mask = None if subtask_mask is None else jnp.repeat(subtask_mask, num_samples, axis = 0)
    out = model_to_use.compute_value(
        expanded_obs, flat_actions,
        prefix_cache = (repeated_kv_cache, repeated_prefix_mask, repeated_subtask_mask),
    )
    val = out[0] if isinstance(out, tuple) else out
    return val.reshape(actions.shape[0], num_samples)


@nnx.jit
def _jitted_compute_value_categorical_subtask(
    model_to_use: _value_fn.BaseValueFunction,
    obs: _model.Observation,
    act: _model.Actions | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Value and resolved subtask id for a categorical-subtask critic, in one trunk pass.

    ``compute_prefix_cache`` already resolves the subtask id — ground truth when
    ``obs.subtask_id`` is set, the predictor's argmax when it is None — and returns the
    image features alongside it. Feeding that tuple straight back as ``prefix_cache``
    skips the (3-camera) encoder on the value pass, so the reported id is by construction
    the one the value was conditioned on.
    """
    image_features, subtask_id, _ = model_to_use.compute_prefix_cache(obs)
    out = model_to_use.compute_value(
        obs, act,
        prefix_cache = (image_features, subtask_id, None),
    )
    val = out[0] if isinstance(out, tuple) else out
    return val, subtask_id


def stack_frames(frame_dicts: list[dict], key: str) -> jax.Array | None:
    """Stack a single key across all frame dicts into a JAX array.

    Args:
        frame_dicts: List of frame dictionaries.
        key: Key to stack from each frame dict.

    Returns:
        JAX array of stacked values, or None if key not present.
    """
    if key not in frame_dicts[0]:
        return None

    values = []
    for f in frame_dicts:
        val = f[key]
        if hasattr(val, "device"):
            val = np.asarray(val)
        values.append(val)
    return jnp.asarray(np.stack(values, axis=0))


def stack_images(frame_dicts: list[dict], image_key: str) -> tuple[dict, dict]:
    """Stack images from frame dicts with proper preprocessing.

    Args:
        frame_dicts: List of frame dictionaries.
        image_key: Key for the image dict (e.g., "image", "mirror_image").

    Returns:
        Tuple of (images_dict, image_masks_dict) as JAX arrays.
    """
    if image_key not in frame_dicts[0]:
        return {}, {}

    batch_size = len(frame_dicts)
    images_dict = {}
    image_masks_dict = {}

    for cam_key in frame_dicts[0][image_key].keys():
        cam_images = []
        for f in frame_dicts:
            img = f[image_key][cam_key]
            if hasattr(img, "device"):
                img = np.asarray(img)
            # Convert uint8 [0, 255] to float32 [-1, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 127.5 - 1.0
            cam_images.append(img)
        images_dict[cam_key] = jnp.asarray(np.stack(cam_images, axis=0))
        image_masks_dict[cam_key] = jnp.ones((batch_size,), dtype=jnp.bool_)

    return images_dict, image_masks_dict


def get_obs_and_action(
    frame_dicts: list[dict],
    prefix: str,
    action_conditioned: bool,
) -> tuple[_model.Observation, jax.Array | None]:
    """Build an Observation and action from frame dicts with a given prefix.

    Args:
        frame_dicts: List of frame dictionaries.
        prefix: Key prefix (e.g., "", "negative_", "random_").
                For prefix="", uses keys like "state", "image", "actions".
                For prefix="negative_", only changes tokenized_prompt keys.
                For prefix="random_" or prefix="counterfactual_", only changes the action key.
        action_conditioned: Whether to include actions and action_mask.

    Returns:
        Tuple of (Observation, action) where action is None if not action_conditioned.
    """
    if prefix == "negative_":
        # Negative only changes the prompt, uses same state/image/action as default
        state_key = "state"
        image_key = "image"
        actions_key = "actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_negative_prompt"
        prompt_mask_key = "tokenized_negative_prompt_mask"
    elif prefix == "random_":
        state_key = "state"
        image_key = "image"
        actions_key = "random_actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"
    elif prefix == "shuffled_":
        state_key = "state"
        image_key = "image"
        actions_key = "shuffled_actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"
    elif prefix == "counterfactual_":
        state_key = "state"
        image_key = "image"
        actions_key = "counterfactual_actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"
    else:
        # Default (empty prefix)
        state_key = "state"
        image_key = "image"
        actions_key = "actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"

    # Subtask boundary indices belong to the positive (TokenizeSubtaskPrompt)
    # prompt; the negative prompt is a plain tokenization with no subtask suffix.
    if prefix == "negative_":
        subtask_start_index_key = None
        subtask_end_index_key = None
    else:
        subtask_start_index_key = "subtask_start_index"
        subtask_end_index_key = "subtask_end_index"

    state = stack_frames(frame_dicts, state_key)
    if state is None:
        raise ValueError(f"Missing required key '{state_key}' in frame dicts")

    images_dict, image_masks_dict = stack_images(frame_dicts, image_key)

    tokenized_prompt = stack_frames(frame_dicts, prompt_key)
    tokenized_prompt_mask = stack_frames(frame_dicts, prompt_mask_key)

    subtask_start_index = (
        stack_frames(frame_dicts, subtask_start_index_key) if subtask_start_index_key is not None else None
    )
    subtask_end_index = (
        stack_frames(frame_dicts, subtask_end_index_key) if subtask_end_index_key is not None else None
    )

    action = None
    action_mask = None
    if action_conditioned:
        action = stack_frames(frame_dicts, actions_key)
        action_mask = stack_frames(frame_dicts, action_mask_key)

    # Ground-truth categorical subtask id (ResNet critics); like the positive prompt it is
    # shared by every prefix variant.
    subtask_id = stack_frames(frame_dicts, "subtask_id")

    obs = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        action_mask=action_mask,
        subtask_start_index=subtask_start_index,
        subtask_end_index=subtask_end_index,
        subtask_id=subtask_id,
    )

    return obs, action


def count_subtask_segments(frames: list[dict], prefix: str = "") -> tuple[int, int, list[str]]:
    """Count distinct subtask segments in a trajectory and find the midpoint split.

    Args:
        frames: List of frame dictionaries, each containing subtask text keys.
        prefix: Key prefix to use (e.g., "", "negative_", "mirror_").

    Returns:
        Tuple of (num_segments, split_frame_idx, segments) where split_frame_idx is the
        frame index of the subtask boundary closest to the midpoint.
    """
    key = prefix + "subtask_1_text"
    count = 0
    prev = None
    boundaries = []
    segments = []
    segment_starts = []
    for i, f in enumerate(frames):
        st = f[key]
        if hasattr(st, "item"):
            st = st.item()
        if st != prev:
            if st.lower() not in ("null", "static", "abnormal"):
                count += 1
                segments.append(st)
                segment_starts.append(i)
                if prev is not None:
                    boundaries.append(i)
            prev = st

    labeled_segments = []
    for idx, (text, start) in enumerate(zip(segments, segment_starts)):
        end = segment_starts[idx + 1] if idx + 1 < len(segment_starts) else len(frames)
        labeled_segments.append(f"{text} - [{start}, {end})")

    split_frame_idx = boundaries[count // 2 - 1] if boundaries else len(frames) // 2
    return count, split_frame_idx, labeled_segments


_CACHE_KEYS = {
    "state", "image", "image_mask", "actions", "action_mask",
    "tokenized_prompt", "tokenized_prompt_mask",
    "subtask_start_index", "subtask_end_index", "subtask_id",
    "tokenized_negative_prompt", "tokenized_negative_prompt_mask",
    "negative_subtask_1_text", "random_actions", "counterfactual_actions",
    "mc_return", "include_subtask", "fps",
    "repo_id", "episode_index", "_frame_index", "_traj_index",
    "is_partial", "has_subtask_annotations",
}


def _unstack_trajectory(traj: dict, t: int) -> dict:
    """Extract a single frame at timestep t from a stacked trajectory dict."""
    frame: dict = {}
    for key, value in traj.items():
        if isinstance(value, dict):
            frame[key] = {k: np.asarray(v[t]) for k, v in value.items()}
        else:
            frame[key] = np.asarray(value[t])
    return frame


def _extract_cache_frame(frame: dict, prompt_text: str) -> dict:
    """Keep only plotting-relevant keys and add subtask_1_text from the original prompt."""
    cached: dict = {}
    for key in _CACHE_KEYS:
        if key in frame:
            cached[key] = frame[key]
    cached["subtask_1_text"] = prompt_text
    return cached


def _decode_repo_id(raw) -> str:
    if isinstance(raw, np.ndarray):
        if raw.ndim != 0:
            raise ValueError(f"Expected scalar repo_id, got shape {raw.shape}")
        raw = raw.item()
    elif isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _decode_and_resize_image_bytes(
    image_bytes: bytes | np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """Decode a single JPEG/PNG image to uint8 and resize to target_size.

    Used at prediction time so that cache_val_episodes can store the raw
    compressed bytes (≈10–30x smaller than decoded uint8) and we only
    materialize the decoded array for one episode at a time.
    """
    import tensorflow as tf

    if isinstance(image_bytes, np.ndarray):
        image_bytes = image_bytes.item() if image_bytes.shape == () else bytes(image_bytes)
    image = tf.io.decode_image(image_bytes, expand_animations = False, dtype = tf.uint8)
    image = tf.image.resize(
        image, target_size, method = tf.image.ResizeMethod.BILINEAR, antialias = True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)
    return image.numpy()


def decode_episode_images(frames: list[dict], target_size: tuple[int, int]) -> None:
    """In-place decode + resize of all `image` entries in a list of cached frames.

    Frames cached via cache_val_episodes (with the upstream dataset configured
    with decode_images=False) hold raw compressed bytes per camera. This
    converts them to uint8 (target_size, 3) arrays so downstream stack_images /
    plotting can consume them. Only operates on frames whose `image[cam]` is
    bytes / scalar object array; already-decoded frames are passed through.
    """
    for frame in frames:
        if "image" not in frame:
            continue
        image_dict = frame["image"]
        for cam_key, value in list(image_dict.items()):
            if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3:
                continue  # already decoded
            image_dict[cam_key] = _decode_and_resize_image_bytes(value, target_size)


def cache_val_episodes(
    trajectory_iter,
    num_val_trajectories: int,
    cache_dir: str | None,
    include_repos: tuple[str, ...],
    save_only: bool,
    input_transform = None,
    allow_duplicate_repos: bool = False,
    episode_indices: tuple[int, ...] | None = None,
    episode_keys: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, list[dict]]:
    """Load or collect validation episodes, optionally caching them to disk.

    Iterates an RLDS trajectory dataset (return_trajectories=True), applies
    input_transform to each frame, and caches the transformed frames as
    per-trajectory pickle files keyed by ``repo_key`` (``repo_index`` when the
    underlying dataset exposes it, else ``repo_id``; sanitized: ``/`` → ``__``).
    Using ``repo_index`` lets a single-task fine-tune cache multiple distinct
    trajectories from the same ``repo_id``.
    On subsequent calls the cache is loaded instead of re-iterating the dataset.

    Multi-worker safe: every host calls this concurrently. Worker 0 is the
    full-coverage worker (caches all ``num_val_trajectories`` repos including
    non-required ones); workers > 0 only cache repos in ``include_repos``. Each
    worker checks ``<sanitized_repo_key>.pkl`` existence at two points
    — when deciding whether to claim the repo and again right before writing
    — and skips if another worker already produced that file.

    With ``episode_indices`` the roles collapse: the trajectory dataset is sharded
    across hosts, so a pinned episode can only be seen by the host whose shard holds
    it, and that host caches it whatever its rank or ``include_repos``. The cache is
    then complete once it holds one file per pinned episode. Files keep the
    ``<repo_key>_<n>`` naming, so two pinned episodes with the same ``repo_key`` on
    different hosts would collide on ``_0``; pin episodes with distinct keys.

    Args:
        trajectory_iter: RLDS dataset yielding full trajectories (stacked dicts),
            or None when loading from an existing cache.
        num_val_trajectories: Number of unique-repo trajectories to collect.
        cache_dir: Directory to save/load cached trajectories. None disables caching.
        include_repos: Repo IDs that must be included (up to len(include_repos) slots reserved).
        save_only: If True, collect and cache episodes then return {}.
        input_transform: Composed transform pipeline (Normalize + ResizeImages + TokenizePrompt)
            applied to each frame before caching. Required when collecting, ignored when loading.
        episode_indices: Cache only these ``episode_index`` values, in place of taking whichever
            trajectories come first. Default None keeps the original positional behaviour.
            Selection is otherwise position-based, so two configs whose filters drop different
            episodes end up caching different ones; naming the episodes is what makes a cache
            comparable across configs.
        episode_keys: Like ``episode_indices`` but as ``(repo_id, episode_index)`` pairs, for
            datasets whose ``episode_index`` restarts per repo (RoboCOIN: every repo has an
            episode 0), where a bare index would match one episode in each repo. Mutually
            exclusive with ``episode_indices``.

    Returns:
        Dict mapping ``<sanitized_repo_key>`` (str) -> sorted list of frame dicts,
        or {} if save_only=True.
    """
    import jax  # local import to keep this util usable from non-JAX callers

    if episode_indices is not None and episode_keys is not None:
        raise ValueError("Pass either episode_indices or episode_keys, not both.")
    pinned_count = None
    if episode_indices is not None:
        pinned_count = len(episode_indices)
    elif episode_keys is not None:
        pinned_count = len(episode_keys)
    pinned_key_set = set(episode_keys) if episode_keys is not None else None

    traj_frames: dict[str, list[dict]] = {}
    process_index = jax.process_index() if jax is not None else 0
    include_set = set(include_repos)

    def _sanitize(repo_id: str) -> str:
        return repo_id.replace("/", "__")

    cache_exists = (
        cache_dir is not None
        and os.path.exists(cache_dir)
        and any(f.endswith(".pkl") for f in os.listdir(cache_dir))
    )

    # Fast-path: if save_only=True and every .pkl this worker would have
    # written already exists, skip the trajectory_iter entirely. Iterating
    # the val tf.data pipeline a second time after the cache is full
    # accumulates host RAM (prefetch buffers + per-traj pickle bursts) and
    # has OOM-killed workers caching the largest trajectories
    # (tableware_cleaning, 2930 frames → 1.3 GB pickle).
    if save_only and cache_dir is not None and os.path.exists(cache_dir):
        existing_pkls = {f for f in os.listdir(cache_dir) if f.endswith(".pkl")}
        # When allow_duplicate_repos=True, files are written as `<repo>_<idx>.pkl`
        # (counter starts at 0 for each new repo), so the first occurrence of each
        # required repo always lands as `<repo>_0.pkl`. Otherwise files are bare
        # `<repo>.pkl`.
        if allow_duplicate_repos:
            required_pkls = {f"{_sanitize(repo)}_0.pkl" for repo in include_repos}
        else:
            required_pkls = {f"{_sanitize(repo)}.pkl" for repo in include_repos}
        all_required = required_pkls.issubset(existing_pkls)
        if pinned_count is not None:
            # Any host may hold a pinned episode, so every host has to keep iterating
            # until the cache holds one file per pinned episode.
            all_required = len(existing_pkls) >= pinned_count
        elif process_index == 0:
            # Worker 0 is the only worker that fills the non-required slots.
            all_required = all_required and len(existing_pkls) >= num_val_trajectories
        if all_required:
            logger.info(
                f"[pidx={process_index}] save_only fast-path: cache already "
                f"contains required .pkls ({len(existing_pkls)} present), "
                f"skipping trajectory iteration."
            )
            return {}

    if cache_exists and not save_only:
        logger.info(f"[pidx={process_index}] Loading cached validation episodes from {cache_dir}")
        for filename in os.listdir(cache_dir):
            if filename.endswith(".pkl"):
                key = filename[: -len(".pkl")]
                cache_file = os.path.join(cache_dir, filename)
                with open(cache_file, "rb") as f:
                    traj_frames[key] = pickle.load(f)
        logger.info(f"[pidx={process_index}] Loaded {len(traj_frames)} trajectories from cache")
    else:
        # Collect path. Runs for save_only=True regardless of cache_exists, so
        # that workers arriving after the first .pkl is written still iterate
        # their shard and contribute their include_repos. The per-file
        # os.path.exists + O_CREAT|O_EXCL checks below guard against
        # double-writes.
        assert trajectory_iter is not None, "No cache found and no trajectory iterator provided."
        assert input_transform is not None, "input_transform is required when collecting trajectories."

        seen_repo_keys: set[int | str] = set()
        seen_required_repo_ids: set[str] = set()
        # Per-repo counter for allow_duplicate_repos=True: pkls saved as
        # `<repo_key>_<num>.pkl` so multiple trajectories from the same repo coexist.
        repo_counters: dict[str, int] = {}
        num_non_required_slots = num_val_trajectories - len(include_repos)
        collected_count = 0
        pinned = pinned_count is not None
        # How many trajectories a host stops after: every pinned episode when pinned, the
        # full-coverage quota on worker 0 otherwise (workers > 0 stop on include_repos).
        target_count = pinned_count if pinned else num_val_trajectories
        if pinned:
            pinned_desc = tuple(episode_indices) if episode_indices is not None else tuple(episode_keys)
            collect_scope = f"pinned episodes {pinned_desc} in this host's shard"
        elif process_index == 0:
            collect_scope = f"all repos up to {num_val_trajectories}"
        else:
            collect_scope = "include_repos only"
        logger.info(f"[pidx={process_index}] Collecting validation trajectories ({collect_scope})")

        if cache_dir:
            os.makedirs(cache_dir, exist_ok = True)
            # TPU workers within the same pod can be assigned different uids
            # for the same user, so a dir created by one worker often
            # isn't writable by the others (mode 0775, group-mismatched).
            # Best-effort chmod to 0o777 — silently ignore if we don't own it.
            try:
                os.chmod(cache_dir, 0o777)
            except (PermissionError, OSError):
                pass

        for traj in trajectory_iter:
            # Some shards/transforms can yield zero-frame trajectories (all
            # frames filtered out by include_subtask / mask_50fps / etc.). Skip
            # rather than crash on traj["repo_id"][0].
            if len(traj["repo_id"]) == 0:
                continue

            repo_id = _decode_repo_id(traj["repo_id"][0])
            repo_key: int | str = int(traj["repo_index"][0]) if "repo_index" in traj else repo_id

            if pinned:
                if "episode_index" not in traj:
                    raise ValueError(
                        "episode_indices/episode_keys was given but the trajectories carry no "
                        "episode_index; this dataset cannot be filtered by episode."
                    )
                episode_index = int(np.asarray(traj["episode_index"][0]))
                if episode_indices is not None and episode_index not in episode_indices:
                    continue
                if pinned_key_set is not None and (repo_id, episode_index) not in pinned_key_set:
                    continue

            if not allow_duplicate_repos and repo_key in seen_repo_keys:
                continue

            # Worker > 0: only claim repos in include_repos, unless episodes are pinned —
            # then whichever host's shard holds the episode is the one that must cache it.
            if process_index != 0 and not pinned and repo_id not in include_set:
                continue

            is_required = repo_id in include_repos and repo_id not in seen_required_repo_ids
            if process_index == 0 and not pinned and not is_required and num_non_required_slots <= 0:
                continue

            sanitized_base = _sanitize(str(repo_key))
            if allow_duplicate_repos:
                idx = repo_counters.get(sanitized_base, 0)
                repo_counters[sanitized_base] = idx + 1
                sanitized = f"{sanitized_base}_{idx}"
            else:
                sanitized = sanitized_base
            cache_file = (
                os.path.join(cache_dir, f"{sanitized}.pkl") if cache_dir else None
            )
            # Skip if another worker already produced this file.
            if cache_file is not None and os.path.exists(cache_file):
                logger.info(
                    f"[pidx={process_index}] {sanitized}.pkl already exists in {cache_dir}, skipping"
                )
                seen_repo_keys.add(repo_key)
                if is_required:
                    seen_required_repo_ids.add(repo_id)
                elif process_index == 0:
                    num_non_required_slots -= 1
                collected_count += 1
                if (pinned or process_index == 0) and collected_count >= target_count:
                    break
                if not pinned and process_index != 0 and len(seen_required_repo_ids) >= len(include_set):
                    break
                continue

            # Un-stack trajectory, apply transforms, keep only cache keys
            traj_len = len(traj["repo_id"])
            frames = []
            for t in range(traj_len):
                frame = _unstack_trajectory(traj, t)
                prompt_text = frame["prompt"]
                if isinstance(prompt_text, bytes):
                    prompt_text = prompt_text.decode("utf-8")
                elif hasattr(prompt_text, "item"):
                    prompt_text = prompt_text.item()
                    if isinstance(prompt_text, bytes):
                        prompt_text = prompt_text.decode("utf-8")
                frame = input_transform(frame)
                frames.append(_extract_cache_frame(frame, prompt_text))

            frames.sort(key = lambda f: int(f["_frame_index"]))

            if cache_file is not None:
                # Re-check existence right before writing to handle the race
                # where another worker finished caching the same repo while we
                # were transforming frames.
                if os.path.exists(cache_file):
                    logger.info(
                        f"[pidx={process_index}] {sanitized}.pkl appeared during transform, skipping write"
                    )
                else:
                    # Use O_CREAT|O_EXCL for an atomic create-or-fail. Catching
                    # FileExistsError + PermissionError treats both as "another
                    # worker won the race, skip".
                    try:
                        fd = os.open(cache_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    except (FileExistsError, PermissionError) as exc:
                        logger.info(
                            f"[pidx={process_index}] race on {sanitized}.pkl ({type(exc).__name__}), skipping write"
                        )
                    else:
                        with os.fdopen(fd, "wb") as f:
                            pickle.dump(frames, f)
                        # chmod 0o777 so other workers in this pod (which may
                        # have inconsistent uids for the same user) can
                        # subsequently read/write this file.
                        try:
                            os.chmod(cache_file, 0o777)
                        except (PermissionError, OSError):
                            pass
                        logger.info(
                            f"[pidx={process_index}] Cached {sanitized} (repo {repo_id}, {traj_len} frames) to {cache_file}"
                        )

            if not save_only:
                traj_frames[sanitized] = frames

            seen_repo_keys.add(repo_key)
            if is_required:
                seen_required_repo_ids.add(repo_id)
            elif process_index == 0:
                num_non_required_slots -= 1
            collected_count += 1

            if (pinned or process_index == 0) and collected_count >= target_count:
                break

        logger.info(
            f"[pidx={process_index}] Collected {collected_count} trajectories, "
            f"unique repos: {len(seen_repo_keys)}"
        )

    if save_only:
        return {}

    return traj_frames


def _variant_obs_and_action(
    frame_dicts: list[dict],
    prefix: str,
    *,
    action_conditioned: bool,
    strip_subtask_id: bool,
) -> tuple[_model.Observation, _model.Actions | None]:
    """``get_obs_and_action`` for one prompt / action variant, minus the ground-truth id if asked."""
    obs, act = get_obs_and_action(frame_dicts, prefix = prefix, action_conditioned = action_conditioned)
    if strip_subtask_id:
        obs = dataclasses.replace(obs, subtask_id = None)
    return obs, act


def predict_values(
    model: _value_fn.BaseValueFunction,
    all_frames: list[tuple],
    ep_mc_returns: dict,
    action_conditioned: bool,
    batch_size: int = 64,
    mesh: "jax.sharding.Mesh | None" = None,
    *,
    strip_subtask_id: bool = False,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
]:
    """Run batched value function inference on collected validation frames.

    Performs up to four forward passes per batch where applicable: default prompt,
    negative (counterfactual) prompt, random actions, and shuffled actions
    (within-trajectory permutation; see ``inject_shuffled_actions``).

    ``strip_subtask_id`` drops the cached ground-truth ``subtask_id`` from every variant's
    observation so a categorical-subtask critic conditions on its own argmax instead; the
    same conditioning then holds across all variants of a frame.

    When the network's compute_value returns (val, attn_scores) (e.g. PaliGemma at
    inference), attention scores are collected alongside predictions and returned as
    the last element. Otherwise the last element is an empty dict.

    Args:
        model: The value function model.
        all_frames: List of (traj_idx, frame_idx_in_ep, frame_dict) tuples in order.
        ep_mc_returns: Dict mapping traj_idx -> list of mc_return values (used for keying output).
        action_conditioned: Whether the model expects actions as input.

    Returns:
        Tuple of (
            all_predictions,
            all_predictions_neg,
            all_predictions_random,
            all_predictions_counterfactual,
            all_predictions_shuffled,
            all_attn_scores,
            all_counterfactual_candidate_values,
        ).

    ``all_counterfactual_candidate_values`` holds the per-frame ``[num_samples]`` vector of
    cached-action values (empty when the store was not joined). The counterfactual
    prediction itself stays the per-frame max; keeping the full vector lets the npz
    analysis reduce with max or a random pick without a second pass over the episode.
    """
    all_predictions: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_predictions_neg: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_predictions_random: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_predictions_counterfactual: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_predictions_shuffled: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_attn_scores: dict[str, list[np.ndarray]] = {ep_idx: [] for ep_idx in ep_mc_returns}
    all_counterfactual_candidate_values: dict[str, list[np.ndarray]] = {
        ep_idx: [] for ep_idx in ep_mc_returns
    }

    # Opt-in data-parallel sharding of each batched forward. When a mesh is
    # given, the (padded) batch is split along DATA_AXIS across all devices;
    # combined with fsdp_devices=1 (replicated params) this is pure data
    # parallelism. mesh=None preserves the prior replicated-input behavior.
    data_sharding = None
    if mesh is not None:
        from openpi.training import sharding as _sharding
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))

    for batch_start in range(0, len(all_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(all_frames))
        batch_frames = all_frames[batch_start:batch_end]

        frame_dicts = [f[2] for f in batch_frames]
        # Pad partial last batches with copies of the final frame so every
        # ``_jitted_compute_value`` call sees the same leading-axis size. Without
        # this, each unique trailing-batch length specialises a new compiled
        # program in the XLA cache, and that cache (per-traj × multiple
        # variants) eats enough HBM that the next ptrain_step can OOM.
        if len(frame_dicts) < batch_size:
            frame_dicts = frame_dicts + [frame_dicts[-1]] * (batch_size - len(frame_dicts))

        obs, act = _variant_obs_and_action(
            frame_dicts, "", action_conditioned = action_conditioned, strip_subtask_id = strip_subtask_id,
        )

        # Shard the default-path batch across devices when a mesh is provided.
        # Other obs variants below (negative / random / counterfactual /
        # shuffled) are unused on the subtask-npz path, so they stay replicated.
        if data_sharding is not None:
            obs = jax.device_put(obs, data_sharding)
            if act is not None:
                act = jax.device_put(act, data_sharding)

        if batch_start == 0:
            logger.info(f"  Batch obs.state: shape={obs.state.shape}, dtype={obs.state.dtype}")
            if obs.images:
                for k, v in obs.images.items():
                    logger.info(f"  Batch obs.images[{k}]: shape={v.shape}, dtype={v.dtype}")
            if obs.tokenized_prompt is not None:
                logger.info(f"  Batch obs.tokenized_prompt: shape={obs.tokenized_prompt.shape}")

        pred_values_np, attn_np = jax.device_get(_jitted_compute_value(model, obs, act))

        global _HBM_LOGGED
        if not _HBM_LOGGED:
            _HBM_LOGGED = True
            for _dev in jax.local_devices():
                _s = _dev.memory_stats() or {}
                logger.info(
                    f"[HBM] proc={jax.process_index()} dev={_dev.id}: "
                    f"peak={_s.get('peak_bytes_in_use', 0) / 1e9:.2f} GB / "
                    f"limit={_s.get('bytes_limit', 0) / 1e9:.2f} GB"
                )

        pred_values_neg_np = None
        if "tokenized_negative_prompt" in frame_dicts[0]:
            obs_neg, act_neg = _variant_obs_and_action(
                frame_dicts, "negative_", action_conditioned = action_conditioned, strip_subtask_id = strip_subtask_id,
            )
            pred_values_neg_np, _ = jax.device_get(_jitted_compute_value(model, obs_neg, act_neg))

        pred_values_random_np = None
        if "random_actions" in frame_dicts[0]:
            obs_random, act_random = _variant_obs_and_action(
                frame_dicts, "random_", action_conditioned = action_conditioned, strip_subtask_id = strip_subtask_id,
            )
            pred_values_random_np, _ = jax.device_get(_jitted_compute_value(model, obs_random, act_random))

        pred_values_counterfactual_np = None
        candidate_values_np = None
        if action_conditioned and "counterfactual_actions" in frame_dicts[0]:
            obs_counterfactual, act_counterfactual = _variant_obs_and_action(
                frame_dicts, "counterfactual_", action_conditioned = True, strip_subtask_id = strip_subtask_id,
            )
            # act_counterfactual is [b, num_samples, ah, ad]; score all cached
            # candidates against a shared prefix cache, keep every candidate's value and
            # reduce to the per-frame max for the reported prediction.
            candidate_values_np = jax.device_get(
                _jitted_compute_value_best_cached(model, obs_counterfactual, act_counterfactual)
            )
            pred_values_counterfactual_np = np.max(candidate_values_np, axis = 1)

        pred_values_shuffled_np = None
        if "shuffled_actions" in frame_dicts[0]:
            obs_shuffled, act_shuffled = _variant_obs_and_action(
                frame_dicts, "shuffled_", action_conditioned = action_conditioned, strip_subtask_id = strip_subtask_id,
            )
            pred_values_shuffled_np, _ = jax.device_get(_jitted_compute_value(model, obs_shuffled, act_shuffled))

        for i, (ep_idx, _, _) in enumerate(batch_frames):
            all_predictions[ep_idx].append(float(pred_values_np[i]))
            if attn_np is not None:
                all_attn_scores[ep_idx].append(attn_np[i])
            if pred_values_neg_np is not None:
                all_predictions_neg[ep_idx].append(float(pred_values_neg_np[i]))
            if pred_values_random_np is not None:
                all_predictions_random[ep_idx].append(float(pred_values_random_np[i]))
            if pred_values_counterfactual_np is not None:
                all_predictions_counterfactual[ep_idx].append(float(pred_values_counterfactual_np[i]))
                all_counterfactual_candidate_values[ep_idx].append(np.asarray(candidate_values_np[i]))
            if pred_values_shuffled_np is not None:
                all_predictions_shuffled[ep_idx].append(float(pred_values_shuffled_np[i]))

    total_predictions = sum(len(preds) for preds in all_predictions.values())
    logger.info(f"Computed {total_predictions} predictions")

    return (
        all_predictions, all_predictions_neg, all_predictions_random,
        all_predictions_counterfactual, all_predictions_shuffled, all_attn_scores,
        all_counterfactual_candidate_values,
    )


@dataclasses.dataclass
class TrajectoryValuePredictions:
    """Every value pass ``predict_values`` ran over one trajectory, unpacked by name.

    A variant list is empty when its inputs were absent from the cached frames (no negative
    prompt, no random / shuffled / counterfactual actions) or the critic has no per-modality
    attention; a full list is one entry per frame. ``counterfactual_actions`` is the
    per-frame max over ``candidate_values``, which keeps every cached candidate's value.
    """

    dataset: list[float]
    negative_prompt: list[float]
    random_actions: list[float]
    counterfactual_actions: list[float]
    shuffled_actions: list[float]
    attn_scores: list[np.ndarray]
    candidate_values: list[np.ndarray]

    def reported(self, *, use_counterfactual_actions: bool, traj_key: str) -> list[float]:
        """The value the evaluation reports: at the best cached action, or at the dataset one."""
        if not use_counterfactual_actions:
            return self.dataset
        if not self.counterfactual_actions:
            raise ValueError(
                f"Counterfactual actions requested but none were cached for {traj_key}. The cached "
                "action store must be joined for this split (see the 'skipping join' warning)."
            )
        return self.counterfactual_actions


def predict_trajectory_values(
    model: _value_fn.BaseValueFunction,
    frames: list[dict],
    traj_key: str,
    *,
    action_conditioned: bool,
    batch_size: int = 64,
    mesh: "jax.sharding.Mesh | None" = None,
    strip_subtask_id: bool = False,
) -> TrajectoryValuePredictions:
    """``predict_values`` over one trajectory, with the per-episode dicts unpacked."""
    indexed = [(traj_key, i, f) for i, f in enumerate(frames)]
    preds, preds_neg, preds_random, preds_cf, preds_shuffled, attn, candidates = predict_values(
        model, indexed, {traj_key: [f["mc_return"] for f in frames]}, action_conditioned,
        batch_size = batch_size, mesh = mesh, strip_subtask_id = strip_subtask_id,
    )
    return TrajectoryValuePredictions(
        dataset = preds[traj_key],
        negative_prompt = preds_neg[traj_key],
        random_actions = preds_random[traj_key],
        counterfactual_actions = preds_cf[traj_key],
        shuffled_actions = preds_shuffled[traj_key],
        attn_scores = attn[traj_key],
        candidate_values = candidates[traj_key],
    )


def apply_override_prompt(frames: list[dict], override_prompt: tuple[np.ndarray, np.ndarray]) -> None:
    """Replace every frame's tokenized prompt with ``(tokens, mask)``.

    The override is prefix-only, so the cached subtask indices are dropped and the critic
    runs with ``subtask_start_index=None`` for the overridden prompt.
    """
    override_tokens, override_mask = override_prompt
    for frame in frames:
        frame["tokenized_prompt"] = override_tokens
        frame["tokenized_prompt_mask"] = override_mask
        frame.pop("subtask_start_index", None)
        frame.pop("subtask_end_index", None)


def inject_shuffled_actions(frames: list[dict], *, action_conditioned: bool) -> None:
    """Add a within-trajectory permutation of ``actions`` as ``shuffled_actions`` on each frame.

    Feeds the shuffled-actions pass of ``predict_values``. The permutation is seeded per
    call so the plot is reproducible across runs; state-only critics and single-frame
    trajectories get nothing.
    """
    if not (action_conditioned and len(frames) > 1 and "actions" in frames[0]):
        return
    permutation = np.random.default_rng(seed = 86).permutation(len(frames))
    shuffled = [frames[p]["actions"] for p in permutation]
    for frame, actions in zip(frames, shuffled, strict = True):
        frame["shuffled_actions"] = actions




@nnx.jit
def _jitted_compute_value_best_cached_categorical(
    model_to_use: _value_fn.BaseValueFunction,
    obs: _model.Observation,
    actions: _model.Actions,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Every cached counterfactual action's value, plus the resolved subtask id.

    The categorical twin of ``_jitted_compute_value_best_cached``: the encoder runs once
    per frame and all ``num_samples`` candidates are scored against the shared image
    features, so the subtask id resolved for the frame (ground truth or argmax) conditions
    every candidate identically. Returns ``[b, num_samples]``; the caller reduces.
    """
    from openpi.models.best_of_n import expand_observation

    num_samples = actions.shape[1]
    image_features, subtask_id, _ = model_to_use.compute_prefix_cache(obs)
    expanded_obs = expand_observation(obs, num_samples)
    flat_actions = actions.reshape(actions.shape[0] * num_samples, actions.shape[2], actions.shape[3])
    repeated_features = jnp.repeat(image_features, num_samples, axis = 0)
    repeated_subtask_id = jnp.repeat(subtask_id, num_samples, axis = 0)
    out = model_to_use.compute_value(
        expanded_obs, flat_actions,
        prefix_cache = (repeated_features, repeated_subtask_id, None),
    )
    val = out[0] if isinstance(out, tuple) else out
    return val.reshape(actions.shape[0], num_samples), subtask_id


def predict_values_categorical_subtask(
    model: _value_fn.BaseValueFunction,
    all_frames: list[tuple],
    ep_keys,
    action_conditioned: bool,
    *,
    use_predicted_subtask: bool,
    use_counterfactual_actions: bool = False,
    batch_size: int = 64,
    mesh: "jax.sharding.Mesh | None" = None,
) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, list[np.ndarray]]]:
    """Batched value inference for critics whose subtask conditioning is a categorical id.

    Unlike the PaliGemma path there is nothing to decode autoregressively: the subtask
    enters the network after the encoder, so a single forward yields both the value and
    the predicted subtask id. ``use_predicted_subtask`` drops ``subtask_id`` from the
    observation, which is what makes the network fall back to its own argmax.

    Returns (values, resolved subtask ids, per-frame cached-action value vectors) keyed by
    episode. The value is the per-frame max when ``use_counterfactual_actions`` is set; the
    candidate vectors are empty otherwise.
    """
    all_predictions: dict[str, list[float]] = {ep_key: [] for ep_key in ep_keys}
    all_subtask_ids: dict[str, list[int]] = {ep_key: [] for ep_key in ep_keys}
    all_candidate_values: dict[str, list[np.ndarray]] = {ep_key: [] for ep_key in ep_keys}

    data_sharding = None
    if mesh is not None:
        from openpi.training import sharding as _sharding
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))

    for batch_start in range(0, len(all_frames), batch_size):
        batch_frames = all_frames[batch_start : batch_start + batch_size]
        frame_dicts = [f[2] for f in batch_frames]
        # Same trailing-batch padding rationale as predict_values: a unique trailing
        # length would specialise another compiled program in the XLA cache.
        if len(frame_dicts) < batch_size:
            frame_dicts = frame_dicts + [frame_dicts[-1]] * (batch_size - len(frame_dicts))

        action_prefix = "counterfactual_" if use_counterfactual_actions else ""
        if use_counterfactual_actions and "counterfactual_actions" not in frame_dicts[0]:
            raise ValueError(
                "use_counterfactual_actions requested but the cached frames carry no "
                "counterfactual_actions (was the cached action store joined for this split?)."
            )
        obs, act = get_obs_and_action(
            frame_dicts,
            prefix = action_prefix,
            action_conditioned = action_conditioned or use_counterfactual_actions,
        )
        if use_predicted_subtask:
            obs = dataclasses.replace(obs, subtask_id = None)
        elif obs.subtask_id is None:
            raise ValueError(
                "Ground-truth subtask conditioning requested but the cached frames carry no "
                "subtask_id (was the cache written by a config with a subtask_vocab?)."
            )

        if data_sharding is not None:
            obs = jax.device_put(obs, data_sharding)
            if act is not None:
                act = jax.device_put(act, data_sharding)

        candidate_values_np = None
        if use_counterfactual_actions:
            candidate_values_np, subtask_ids_np = jax.device_get(
                _jitted_compute_value_best_cached_categorical(model, obs, act)
            )
            pred_values_np = np.max(candidate_values_np, axis = 1)
        else:
            pred_values_np, subtask_ids_np = jax.device_get(
                _jitted_compute_value_categorical_subtask(model, obs, act)
            )

        for i, (ep_key, _, _) in enumerate(batch_frames):
            all_predictions[ep_key].append(float(pred_values_np[i]))
            all_subtask_ids[ep_key].append(int(subtask_ids_np[i]))
            if candidate_values_np is not None:
                all_candidate_values[ep_key].append(np.asarray(candidate_values_np[i]))

    total_predictions = sum(len(preds) for preds in all_predictions.values())
    logger.info(
        f"Computed {total_predictions} predictions "
        f"({'predicted' if use_predicted_subtask else 'ground-truth'} subtask conditioning, "
        f"{'max over cached actions' if use_counterfactual_actions else 'dataset action'})"
    )

    return all_predictions, all_subtask_ids, all_candidate_values



def critic_network(model) -> BaseValueNetwork:
    """The value network behind a critic: SARSA/MC store it as ``network``, CQL as ``q_network``."""
    network = getattr(model, "network", None) or getattr(model, "q_network", None)
    if network is None:
        raise ValueError("Critic model exposes neither .network nor .q_network.")
    return network


# =============================================================================
# Subtask prediction: one seam over the two critic families
#
# A subtask critic predicts the current subtask either as free text (PaliGemma decodes it
# autoregressively) or as a categorical id (ResNet argmaxes a classifier head). Everything
# downstream of that — accuracy, npz, gradients, video — is identical, so the difference is
# confined to `predict_values_with_subtasks` and its two implementations below. The decode
# machinery that only the autoregressive implementation needs lives here with it.
# =============================================================================


def _build_subtask_decoder(critic_model) -> dict:
    """Return JIT'd {prefix_forward, decode_step, logits} closures.

    Mirrors the serving-time ``SubtaskDecoder``: the prefix includes the state token
    (via ``_embed_prefix``), matching the serving path.
    """
    @nnx.jit
    def _prefix_forward(model, observation):
        n = critic_network(model)
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
        last_hidden = jnp.take_along_axis(hidden, last_text_pos[:, None, None], axis = 1)
        return last_hidden, kv_cache, prefix_mask, last_text_pos

    def _decode_step(model, token_id, kv_cache, prefix_mask, last_text_pos, suffix_pos_so_far):
        # Not JIT'd: each step appends 1 to kv_cache, so shapes change every call.
        n = critic_network(model)
        tok_arr = jnp.asarray(token_id, dtype = jnp.int32).reshape(1, 1)
        tok_embed = n.PaliGemma.llm(tok_arr, method = "embed")
        to_prefix = prefix_mask[:, None, :]
        to_suffix = jnp.ones((1, 1, suffix_pos_so_far + 1), dtype = jnp.bool_)
        mask = jnp.concatenate([to_prefix, to_suffix], axis = -1)
        position = (last_text_pos + suffix_pos_so_far + 1).reshape(1, 1)
        (hidden,), kv_cache = n.PaliGemma.llm(
            [tok_embed], mask = mask, positions = position, kv_cache = kv_cache,
        )
        return hidden, kv_cache
    @nnx.jit
    def _logits_from_hidden(model, hidden):
        return critic_network(model).decode(hidden)

    return {
        "prefix_forward": _prefix_forward,
        "decode_step": _decode_step,
        "logits": _logits_from_hidden,
    }


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids))
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return str(inner.decode(token_ids))
    return " ".join(str(t) for t in token_ids)


def _run_prefix_forward(closures, critic_model, obs):
    last_hidden, kv_cache, prefix_mask, last_text_pos = closures["prefix_forward"](critic_model, obs)
    return {
        "last_hidden": last_hidden,
        "kv_cache": kv_cache,
        "prefix_mask": prefix_mask,
        "last_text_pos": last_text_pos,
    }


def _run_decode_step(closures, critic_model, prefix_state, token_id, suffix_pos):
    return closures["decode_step"](
        critic_model, token_id, prefix_state["kv_cache"],
        prefix_state["prefix_mask"], prefix_state["last_text_pos"], suffix_pos,
    )


def _score_gt_perplexity(
    closures,
    critic_model,
    critic_obs,
    gt_token_ids: list[int],
) -> float:
    """Teacher-forced perplexity of the cached ground-truth subtask tokens."""
    if not gt_token_ids:
        return float("nan")
    state = _run_prefix_forward(closures, critic_model, critic_obs)
    logits = closures["logits"](critic_model, state["last_hidden"])
    log_probs = jax.nn.log_softmax(logits[0, 0])
    total_neg_logp = -float(np.asarray(log_probs[gt_token_ids[0]]))
    kv_cache = state["kv_cache"]
    for k in range(1, len(gt_token_ids)):
        prev = gt_token_ids[k - 1]
        state["kv_cache"] = kv_cache
        hidden, kv_cache = _run_decode_step(closures, critic_model, state, prev, k - 1)
        logits = closures["logits"](critic_model, hidden)
        log_probs = jax.nn.log_softmax(logits[0, 0])
        total_neg_logp += -float(np.asarray(log_probs[gt_token_ids[k]]))
    return float(np.exp(total_neg_logp / len(gt_token_ids)))


def _extract_gt_subtask_tokens(frame: dict) -> list[int]:
    """Slice the cached ``tokenized_prompt`` between subtask_{start,end}_index."""
    tokens = np.asarray(frame["tokenized_prompt"]).tolist()
    start = int(np.asarray(frame["subtask_start_index"]))
    end = int(np.asarray(frame["subtask_end_index"]))
    if end < start:
        return []
    return [int(t) for t in tokens[start : end + 1]]


def _build_critic_obs_for_frame(
    frame: dict,
    prefix_tokens: np.ndarray,
    prefix_mask: np.ndarray,
    image_keys: tuple[str, ...],
) -> _model.Observation:
    """Single-frame Observation with prefix-only prompt and no subtask indices.

    Routes via `Observation.from_dict` so the uint8 -> float32 [-1, 1] image
    cast fires (the Observation constructor itself does NOT do this cast, and
    `preprocess_observation` only resizes / augments — it doesn't convert
    dtype). Skipping the cast feeds raw [0, 255] into the SigLIP encoder and
    completely OODs the model.
    """
    image_dict: dict = {}
    image_mask_dict: dict = {}
    for k in image_keys:
        img = np.asarray(frame["image"][k])
        image_dict[k] = jnp.asarray(img)[None, ...]
        image_mask_dict[k] = jnp.array([True], dtype = jnp.bool_)
    return _model.Observation.from_dict({
        "image": image_dict,
        "image_mask": image_mask_dict,
        "state": jnp.asarray(np.asarray(frame["state"]))[None, ...],
        "tokenized_prompt": jnp.asarray(prefix_tokens)[None, :],
        "tokenized_prompt_mask": jnp.asarray(prefix_mask)[None, :],
    })


def subtask_boundary_indices(frames: list[dict]) -> list[int]:
    """Frame indices where the cached ground-truth subtask changes.

    Reads whichever ground-truth form the cache carries: the categorical ``subtask_id``
    written by ``SubtaskTextToId``, else the tokenized subtask span. This is a property of
    the cache, not of the critic, so both critic families come through here — and both get
    boundaries at full frame resolution regardless of the decode stride, which is what the
    boundary-MAE metric needs.
    """
    boundaries: list[int] = []
    prev = None
    for i, f in enumerate(frames):
        if "subtask_id" in f:
            current = int(np.asarray(f["subtask_id"]))
        elif "subtask_start_index" in f and "subtask_end_index" in f:
            current = tuple(_extract_gt_subtask_tokens(f))
        else:
            continue
        if prev is not None and current != prev:
            boundaries.append(i)
        prev = current
    return boundaries


def subtask_segment_labels(boundaries: list[int], gt_texts: list[str | None], num_frames: int) -> list[str]:
    """``"<subtask> - [start, end)"`` per ground-truth segment, for the video's caption list."""
    starts = [0, *boundaries]
    labels = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else num_frames
        labels.append(f"{gt_texts[start] or ''} - [{start}, {end})")
    return labels

@dataclasses.dataclass(frozen = True)
class SubtaskPredictions:
    """Per-frame subtask predictions and values for one trajectory, family-agnostic.

    ``values`` is the reported value under the chosen subtask / action axes; ``value_passes``
    carries every pass the value run produced (dataset, negative prompt, random / shuffled /
    counterfactual actions, attention), which the standard validation plots draw from
    regardless of how the subtask was obtained. ``perplexities`` is NaN wherever the critic
    family has no teacher-forced perplexity (the categorical head has none) or scoring was
    disabled. ``predicted_ids`` / ``gt_ids`` are None for text-decoding critics, whose
    subtasks are not drawn from a fixed vocab; ``gt_ids`` is also None for unannotated
    episodes. ``value_frames`` are the frames as actually
    scored, so a caller taking gradients differentiates against the same prompt the value
    used.
    """

    values: list[float]
    value_passes: TrajectoryValuePredictions
    predicted_texts: list[str | None]
    gt_texts: list[str | None]
    perplexities: list[float | None]
    sample_indices: list[int]
    value_frames: list[dict]
    predicted_ids: list[int] | None = None
    gt_ids: list[int] | None = None


def _sample_indices_for(num_frames: int, stride: int | None, fps: int) -> list[int]:
    """Frames the subtask prediction is reported at; the last frame is always included."""
    step = stride if stride is not None else max(1, fps)
    indices = list(range(0, num_frames, step))
    if indices and indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return indices



# Decode machinery is built once per (model, tokenizer) and reused across trajectories.
# `_build_subtask_decoder` and `SubtaskDecoder` both define freshly `nnx.jit`-decorated
# closures, so rebuilding them per trajectory would retrace the prefix forward — minutes of
# recompilation per episode on a PaliGemma critic.
_DECODER_CACHE: dict[tuple[int, int], tuple] = {}


def _decoder_for(model, tokenizer, *, is_rank0: bool) -> tuple:
    key = (id(model), id(tokenizer))
    if key not in _DECODER_CACHE:
        from openpi.policies.subtask_decoder import SubtaskDecoder

        if is_rank0:
            logger.info("Building subtask decode + GT-perplexity closures (once per process).")
        _DECODER_CACHE[key] = (
            _build_subtask_decoder(model),
            SubtaskDecoder(model, tokenizer, decode_every = 1, max_tokens = 16),
        )
        if not is_rank0:
            # Quiet the per-decode INFO log on non-rank-0 ranks, which decode in lockstep
            # but discard the result.
            logging.getLogger("openpi.policies.subtask_decoder").setLevel(logging.WARNING)
    return _DECODER_CACHE[key]


def _predict_with_decoded_subtask(
    model, frames, traj_key, *, tokenizer, sample_indices, use_predicted_subtask,
    use_counterfactual_actions, action_conditioned, score_gt_perplexity, batch_size, mesh,
    is_rank0,
) -> SubtaskPredictions:
    """Autoregressive-text implementation (PaliGemma).

    Decodes the subtask at each sampled frame, optionally rebuilds every prompt as
    ``task_description + decoded_subtask + "\n"``, then scores values once. The decode and
    the perplexity scoring are JIT'd SPMD collectives, so every rank drives them in lockstep
    and only the logging is gated to rank 0.
    """
    import openpi.transforms as _transforms

    network = critic_network(model)
    image_keys = tuple(network.config.image_keys)
    closures, decode_module = _decoder_for(model, tokenizer, is_rank0 = is_rank0)

    prefix_text = decode_text(frames[0]["subtask_1_text"])
    prefix_tokens, prefix_mask, _, _ = _transforms._tokenize_subtask_prompt(
        tokenizer, prefix_text, "", append_newline = False,
    )

    predicted_texts: list[str | None] = [None] * len(frames)
    gt_texts: list[str | None] = [None] * len(frames)
    perplexities: list[float | None] = [None] * len(frames)
    for sample_pos, t in enumerate(sample_indices):
        critic_obs = _build_critic_obs_for_frame(frames[t], prefix_tokens, prefix_mask, image_keys)
        decoded = decode_module.predict(critic_obs)
        gt_tokens = _extract_gt_subtask_tokens(frames[t])
        gt_perp = (
            _score_gt_perplexity(closures, model, critic_obs, gt_tokens)
            if (gt_tokens and score_gt_perplexity) else float("nan")
        )
        predicted_texts[t] = decoded["predicted_subtask"]
        gt_texts[t] = _decode_token_ids(tokenizer, gt_tokens) if gt_tokens else ""
        perplexities[t] = gt_perp
        if is_rank0:
            logger.info(
                f"  [t={t}] gt_pp={gt_perp:.4f} pred={predicted_texts[t]!r} "
                f"pred_ids={decoded['predicted_subtask_tokens']} gt={gt_texts[t]!r} "
                f"gt_ids={gt_tokens} (sample {sample_pos + 1}/{len(sample_indices)})"
            )

    # Forward-fill the decode across the frames between samples, then rebuild the prompts.
    # Identical on every rank (the decode is deterministic), which matters because the value
    # pass below is SPMD over the frames these prompts produce.
    if use_predicted_subtask:
        filled = ""
        value_frames = []
        for i, f in enumerate(frames):
            if predicted_texts[i] is not None:
                filled = predicted_texts[i] or ""
            tok, msk, s0, s1 = _transforms._tokenize_subtask_prompt(
                tokenizer, prefix_text, filled, append_newline = True,
            )
            frame = dict(f)
            frame["tokenized_prompt"] = np.asarray(tok)
            frame["tokenized_prompt_mask"] = np.asarray(msk)
            frame["subtask_start_index"] = np.int32(s0)
            frame["subtask_end_index"] = np.int32(s1)
            value_frames.append(frame)
    else:
        value_frames = list(frames)

    value_passes = predict_trajectory_values(
        model, value_frames, traj_key, action_conditioned = action_conditioned, batch_size = batch_size, mesh = mesh,
    )

    return SubtaskPredictions(
        values = value_passes.reported(use_counterfactual_actions = use_counterfactual_actions, traj_key = traj_key),
        value_passes = value_passes,
        predicted_texts = predicted_texts,
        gt_texts = gt_texts,
        perplexities = perplexities,
        sample_indices = sample_indices,
        value_frames = value_frames,
    )


def _predict_with_categorical_subtask(
    model, frames, traj_key, *, sample_indices, use_predicted_subtask,
    use_counterfactual_actions, action_conditioned, batch_size, mesh,
) -> SubtaskPredictions:
    """Categorical-id implementation (ResNet).

    There is no autoregressive decode and no teacher-forced perplexity, so perplexity stays
    NaN. The values come from the same ``predict_values`` pass as every other critic, with
    the ground-truth id stripped when the critic is to condition on its own argmax; the
    predicted ids come from one extra trunk pass, which reads the image features alone and so
    scores the dataset action whatever the action axis is set to.
    """
    network = critic_network(model)
    vocab = network.config.subtask_vocab
    if vocab is None:
        raise ValueError("Categorical subtask evaluation requires subtask_vocab on the network config.")
    if "subtask_id" not in frames[0] and not use_predicted_subtask:
        raise ValueError(
            "Categorical subtask evaluation requires a cached subtask_id on every frame "
            "(was the cache written by a config with a subtask_vocab?)."
        )

    value_passes = predict_trajectory_values(
        model, frames, traj_key, action_conditioned = action_conditioned,
        batch_size = batch_size, mesh = mesh, strip_subtask_id = use_predicted_subtask,
    )
    indexed = [(traj_key, i, f) for i, f in enumerate(frames)]
    _, predicted_only, _ = predict_values_categorical_subtask(
        model, indexed, {traj_key}, action_conditioned,
        use_predicted_subtask = True,
        use_counterfactual_actions = False,
        batch_size = batch_size, mesh = mesh,
    )
    predicted_ids = predicted_only[traj_key]

    gt_ids = [int(np.asarray(f["subtask_id"])) for f in frames] if "subtask_id" in frames[0] else None
    return SubtaskPredictions(
        values = value_passes.reported(use_counterfactual_actions = use_counterfactual_actions, traj_key = traj_key),
        value_passes = value_passes,
        predicted_texts = [vocab[i] for i in predicted_ids],
        gt_texts = [vocab[i] for i in gt_ids] if gt_ids is not None else [""] * len(frames),
        perplexities = [float("nan")] * len(frames),
        sample_indices = sample_indices,
        value_frames = list(frames),
        predicted_ids = predicted_ids,
        gt_ids = gt_ids,
    )


def _uses_categorical_subtask(model, tokenizer) -> bool:
    """Whether this critic predicts the subtask as a categorical id rather than as text.

    Two independent signals have to agree: ``uses_subtask_id`` on the network config (only
    the categorical config declares it) and the presence of a critic tokenizer (only a
    text-decoding critic has one, since ``_get_critic_tokenizer`` returns None otherwise).

    The cross-check is what makes the ``getattr`` default safe. On its own, a categorical
    config that forgot to declare ``uses_subtask_id`` would silently fall through to the
    text path; because such a critic also has no tokenizer, the disagreement is caught here
    and raised instead.
    """
    config = critic_network(model).config
    declares_id = bool(getattr(config, "uses_subtask_id", False))
    has_tokenizer = tokenizer is not None
    if declares_id and has_tokenizer:
        raise ValueError(
            f"{type(config).__name__} declares uses_subtask_id but a critic tokenizer was "
            "supplied: the subtask is either a categorical id or decoded text, not both."
        )
    if not declares_id and not has_tokenizer:
        raise ValueError(
            f"{type(config).__name__} supplies no critic tokenizer and does not declare "
            "uses_subtask_id, so the subtask family is ambiguous. A categorical critic must "
            "expose uses_subtask_id = True; a text-decoding one must supply its tokenizer."
        )
    return declares_id


def predict_values_with_subtasks(
    model,
    frames: list[dict],
    traj_key: str,
    *,
    tokenizer,
    stride: int | None,
    use_predicted_subtask: bool,
    use_counterfactual_actions: bool,
    action_conditioned: bool,
    score_gt_perplexity: bool = True,
    batch_size: int = 64,
    mesh = None,
    is_rank0: bool = True,
) -> SubtaskPredictions:
    """Per-frame values and subtask predictions for one trajectory, either critic family.

    The family is chosen by ``_uses_categorical_subtask``, which cross-checks the config
    against the tokenizer so a mis-declared critic fails loudly rather than running down the
    wrong path.

    ``use_predicted_subtask`` selects the critic's own subtask over the cached ground truth;
    ``use_counterfactual_actions`` evaluates at the best cached policy action instead of the
    dataset action. The two are independent, as is the caller's decision to take gradients.
    """
    fps = int(frames[0]["fps"])
    sample_indices = _sample_indices_for(len(frames), stride, fps)

    if _uses_categorical_subtask(model, tokenizer):
        return _predict_with_categorical_subtask(
            model, frames, traj_key,
            sample_indices = sample_indices,
            use_predicted_subtask = use_predicted_subtask,
            use_counterfactual_actions = use_counterfactual_actions,
            action_conditioned = action_conditioned,
            batch_size = batch_size, mesh = mesh,
        )
    return _predict_with_decoded_subtask(
        model, frames, traj_key,
        tokenizer = tokenizer,
        sample_indices = sample_indices,
        use_predicted_subtask = use_predicted_subtask,
        use_counterfactual_actions = use_counterfactual_actions,
        action_conditioned = action_conditioned,
        score_gt_perplexity = score_gt_perplexity,
        batch_size = batch_size, mesh = mesh, is_rank0 = is_rank0,
    )
