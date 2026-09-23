"""Evaluate a value function checkpoint on cached RLDS validation trajectories."""

import dataclasses
import gc
import logging
import os
import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlds_utils.load_model_utils import load_critic
from openpi.rlds_utils.load_model_utils import load_train_module
from openpi.rlds_utils.utils import apply_override_prompt
from openpi.rlds_utils.utils import cache_val_episodes
from openpi.rlds_utils.utils import count_subtask_segments
from openpi.rlds_utils.utils import decode_episode_images
from openpi.rlds_utils.utils import get_obs_and_action
from openpi.rlds_utils.utils import inject_shuffled_actions
from openpi.rlds_utils.utils import predict_trajectory_values
from openpi.rlds_utils.utils import predict_values_with_subtasks
from openpi.rlds_utils.utils import subtask_boundary_indices
from openpi.rlds_utils.utils import subtask_segment_labels
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as _sharding
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen = True)
class EvalConfig:
    """Configuration for value function evaluation."""

    config_name: str
    checkpoint_path: str
    split: str = "val"
    fine_tune: str | None = None
    include_repos: tuple[str, ...] = ()
    num_trajectories: int = 10
    cache_dir: str | None = None
    output_dir: str | None = None
    project_name: str = "seeq_eval"
    counterfactual_action_store_dir: str | None = None
    # Critic checkpoint step to load. If None, uses the latest step.
    step: int | None = None
    override_task_prompt: str | None = None
    batch_size: int = 32
    # If True, populate the validation cache and exit before plotting.
    cache_only: bool = False
    # Optional override of the data config's rlds_data_dir (e.g. point at a local
    # mirror of the TFDS data instead of the GCS default baked into the config).
    rlds_data_dir: str | None = None
    # When set, additionally write one ``<traj>.npz`` per evaluated trajectory into this
    # dir — holding every per-frame value prediction, the ground-truth subtask boundaries
    # and, for a subtask critic, the subtask predictions sampled at
    # ``subtask_decode_stride`` frame intervals. Plots and videos are rendered either way.
    subtask_npz_dir: str | None = None
    # Frame stride between autoregressive subtask decodes. None decodes once per
    # second (i.e. fps frames); 1 decodes at every frame instead of forward-filling
    # between sparse decodes.
    subtask_decode_stride: int | None = None
    # Teacher-forced perplexity of the cached ground-truth subtask tokens, scored once
    # per decoded frame. Diagnostic only: it drives the video's perplexity curve and
    # never affects values or the decoded subtask. It
    # costs a second full prefix forward per frame plus a non-JIT'd decode walk, so
    # disable it when decoding at a small stride. False leaves the perplexities NaN,
    # which the plotting path already tolerates.
    score_gt_perplexity: bool = True
    # LANGUAGE axis (subtask critics only). Selects which subtask conditions the
    # per-frame value: False uses the ground-truth subtask cached in the .pkl,
    # True rebuilds each prompt as task_description + decoded_subtask + "\n" from
    # the SubtaskDecoder output. The subtask is always decoded either way (it
    # titles the video); this flag only decides whether it feeds the value.
    # Orthogonal to counterfactual_value_action.
    condition_on_decoded_subtask: bool = False
    # Admit unannotated HDF5-schema episodes for evaluation with predicted subtasks.
    # Ground-truth subtask diagnostics are unavailable in this mode.
    allow_unannotated_episodes: bool = False
    # Skip the subtask prediction (and its video) for a subtask critic, scoring the cached
    # ground-truth-subtask prompts with no decoding at all. The standard validation plots
    # are rendered regardless of this flag.
    disable_subtask_decoding: bool = False
    # ACTION axis (action-conditioned critics only). Selects which action the
    # per-frame Q is evaluated at: False uses the dataset (behaviour) action, True
    # uses the highest-value cached counterfactual (policy-generated) action, which
    # requires the counterfactual action store to be joined for this split.
    # Orthogonal to condition_on_decoded_subtask.
    counterfactual_value_action: bool = False
    # Cache exactly these episode_index values instead of whichever trajectories come first.
    # Empty (default) preserves the positional behaviour. Two configs that filter frames
    # differently otherwise cache different episodes, which makes their metrics
    # non-comparable per episode; naming the episodes pins them across arms.
    val_episode_indices: tuple[int, ...] = ()
    # Same pinning as ``val_episode_indices`` but repo-qualified, ``"<repo_id>:<episode_index>"``
    # per entry (e.g. ``RoboCOIN/Split_aloha_pour_tea:71``), for multi-repo datasets whose
    # episode_index restarts per repo, where a bare index would match one episode in every
    # repo. Mutually exclusive with ``val_episode_indices``.
    val_episodes: tuple[str, ...] = ()
    # Cache multiple trajectories from one repo id. Derived from --fine-tune by default,
    # because a fine-tune targets a single-repo dataset; set explicitly for a TrainConfig on
    # such a dataset (e.g. the shirt-hang ResNet), which would otherwise cache exactly one
    # episode and not be comparable with the fine-tuned arms.
    allow_duplicate_repos: bool = False
    # GRADIENT axis. When True, the subtask npz additionally carries per-frame
    # ``||grad_a Q(s, a)||^2`` at whichever action the value was read at. Costs one
    # backward pass per batch, so it is opt-in and orthogonal to the other two axes;
    # it needs an action-conditioned critic and only has an effect with
    # --subtask-npz-dir set (that is the sink it writes to).
    action_grad_norm: bool = False
    # IMAGE-GRADIENT axis. When True, the subtask npz additionally carries, per frame and per
    # camera, ``||grad_I Q(s, a)||^2`` at whichever action the value was read at. One backward
    # pass per batch, opt-in and independent of the other axes; only has an effect with
    # --subtask-npz-dir set.
    image_grad_norm: bool = False
    # Number of FSDP devices for the inference mesh. None → jax.device_count()
    # (prior default: pure FSDP, params sharded across devices, inputs
    # replicated). Set to 1 on a GPU node for pure data parallelism: params
    # replicated and each batched value forward split along the batch axis
    # across all devices (mesh = (device_count, 1)).
    fsdp_devices: int | None = None
    # Optional wandb run name (used when output_dir is None → wandb logging).
    # Defaults to f"eval_{config_name}" when unset.
    wandb_run_name: str | None = None
    # False skips wandb entirely: no run is created and neither the standard validation
    # plots nor the subtask videos are rendered (they only exist to be logged; --output-dir
    # still writes them to disk). Rendering and upload cost about as much as the value pass
    # itself, so this is the switch for runs whose only product is the npz -- which is why
    # it requires --subtask-npz-dir: with both off the evaluation computes nothing anyone
    # can read.
    wandb_logging: bool = True


def _parse_val_episodes(entries: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    """``"<repo_id>:<episode_index>"`` entries -> ``(repo_id, episode_index)`` pairs."""
    keys = []
    for entry in entries:
        repo_id, sep, index = entry.rpartition(":")
        if not sep or not repo_id or not index.isdigit():
            raise ValueError(f"--val-episodes entries must look like '<repo_id>:<episode_index>', got {entry!r}.")
        keys.append((repo_id, int(index)))
    return tuple(keys)


def _resolve_eval_cache_dir(eval_config: EvalConfig) -> str:
    if eval_config.cache_dir is not None:
        return eval_config.cache_dir
    return os.path.join(eval_config.checkpoint_path, "eval_cache", eval_config.split)


def _load_cached_trajectories(cache_dir: str) -> dict[str, list[dict]]:
    # Match generate_validation_plots_rlds's loader: accept any .pkl, use the
    # bare filename (sans .pkl) as the trajectory key. The older `traj_<int>`
    # naming is no longer produced by cache_val_episodes, which writes
    # `<sanitized_repo_key>.pkl` (e.g. RoboCOIN__Split_aloha_pour_tea.pkl).
    # epath so a gs:// cache dir works (reads pickle bytes straight from GCS),
    # in addition to local/NFS paths.
    from etils import epath

    traj_frames: dict[str, list[dict]] = {}
    for path in sorted(epath.Path(cache_dir).iterdir()):
        if path.name.endswith(".pkl"):
            key = path.name[: -len(".pkl")]
            traj_frames[key] = pickle.loads(path.read_bytes())
    return traj_frames


def _split_trajectory_frames(
    traj_frames: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, int, str]], dict[str, list[str]]]:
    max_subtask_segments = 16
    split_traj_frames: dict[str, list[dict]] = {}
    traj_to_repo_ep: dict[str, tuple[str, int, str]] = {}
    ep_subtasks: dict[str, list[str]] = {}

    for traj_idx, frames in traj_frames.items():
        if not frames:
            continue

        episode_index = int(frames[0]["episode_index"])
        repo_id = frames[0]["repo_id"]
        if isinstance(repo_id, np.ndarray):
            repo_id = repo_id.item()
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode("utf-8")

        num_segments, split_frame_idx, segments = count_subtask_segments(frames)
        if num_segments > max_subtask_segments:
            split_segment_idx = num_segments // 2
            key_part0 = f"{traj_idx}_p0"
            key_part1 = f"{traj_idx}_p1"
            split_traj_frames[key_part0] = frames[:split_frame_idx]
            split_traj_frames[key_part1] = frames[split_frame_idx:]
            traj_to_repo_ep[key_part0] = (repo_id, episode_index, "_part0")
            traj_to_repo_ep[key_part1] = (repo_id, episode_index, "_part1")
            ep_subtasks[key_part0] = segments[:split_segment_idx]
            ep_subtasks[key_part1] = segments[split_segment_idx:]
        else:
            key = str(traj_idx)
            split_traj_frames[key] = frames
            traj_to_repo_ep[key] = (repo_id, episode_index, "")
            ep_subtasks[key] = segments

    return split_traj_frames, traj_to_repo_ep, ep_subtasks


def _render_subtask_video(
    mc_returns: list[float],
    predicted_values: list[float],
    perplexities: list[float | None],
    predicted_texts: list[str | None],
    gt_texts: list[str | None],
    frame_images: list[np.ndarray],
    fps: int,
    ep_idx: int,
    subtask_texts: list[str] | None,
    output_dir: str | None,
    plot_key: str,
    *,
    condition_on_decoded: bool = False,
):
    """5-subplot video laid out 2 rows x 3 cols (6th cell empty).

    Layout:
        Row 0: left wrist  | value plot  | perplexity plot (title = "Pred: ...\nGT: ...")
        Row 1: right wrist | base camera | (empty)

    Per-frame Predicted and Ground-Truth subtask strings are stacked as the
    perplexity subplot's two-line title so they sit directly above the
    perplexity curve, visible in every frame.
    """
    import matplotlib.pyplot as plt

    T = len(mc_returns)
    timesteps = np.arange(T)
    perplexity_arr = np.array(
        [p if p is not None and np.isfinite(p) else np.nan for p in perplexities],
        dtype = np.float64,
    )
    # Y-axis clip for the perplexity subplot: cap the displayed max at 5x the
    # min so a single huge spike (e.g. a model collapse) doesn't squash the
    # rest of the curve into a flat line. Underlying values are unchanged.
    finite_perp = perplexity_arr[np.isfinite(perplexity_arr)]
    perp_ylim: tuple[float, float] | None = None
    if finite_perp.size > 0:
        perp_min = float(np.min(finite_perp))
        perp_max_raw = float(np.max(finite_perp))
        perp_ylim = (perp_min, min(perp_max_raw, 5.0 * perp_min))

    fig, axes = plt.subplots(2, 3, figsize = (20, 10))
    ax_left_wrist = axes[0, 0]
    ax_val = axes[0, 1]
    ax_perp = axes[0, 2]
    ax_right_wrist = axes[1, 0]
    ax_base = axes[1, 1]
    ax_empty = axes[1, 2]

    subtask_caption = None
    if subtask_texts:
        numbered = [f"{i+1}. {t}" for i, t in enumerate(subtask_texts)]
        lines = ["  ".join(numbered[i : i + 3]) for i in range(0, len(numbered), 3)]
        subtask_caption = "Subtasks:\n" + "\n".join(lines)
        bottom_margin = 0.10 + 0.03 * (len(lines) + 1)
        fig.subplots_adjust(bottom = bottom_margin)

    video_frames = []
    for t in range(T):
        for ax in axes.flat:
            ax.cla()

        ax_left_wrist.imshow(frame_images[t][0])
        ax_left_wrist.axis("off")
        ax_left_wrist.set_title("Left Wrist", fontsize = 12)

        ax_right_wrist.imshow(frame_images[t][1])
        ax_right_wrist.axis("off")
        ax_right_wrist.set_title("Right Wrist", fontsize = 12)

        ax_base.imshow(frame_images[t][2])
        ax_base.axis("off")

        ax_val.plot(timesteps, mc_returns, label = "MC Returns", color = "blue", linewidth = 2)
        ax_val.plot(
            timesteps, predicted_values, label = "Predicted Value",
            color = "orange", linewidth = 2, linestyle = "--",
        )
        ax_val.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_val.set_xlabel("Timestep", fontsize = 11)
        ax_val.set_ylabel("Value", fontsize = 11)
        if condition_on_decoded:
            decoded_title = predicted_texts[t] if predicted_texts[t] is not None else "(no decode yet)"
            ax_val.set_title(
                f"Episode {ep_idx} - Value | decoded subtask: {decoded_title}",
                fontsize = 12, wrap = True,
            )
        else:
            ax_val.set_title(f"Episode {ep_idx} - Value", fontsize = 12)
        ax_val.legend(fontsize = 10)
        ax_val.grid(visible = True, alpha = 0.3)

        pred_text = predicted_texts[t] if predicted_texts[t] is not None else "(no prediction this step)"
        gt_text = gt_texts[t] if gt_texts[t] is not None else "(no GT)"
        ax_perp.plot(timesteps, perplexity_arr, color = "purple", linewidth = 2, marker = "o", markersize = 3)
        ax_perp.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_perp.set_xlabel("Timestep", fontsize = 32)
        ax_perp.set_ylabel("GT Subtask Perplexity", fontsize = 32)
        ax_perp.set_title(f"Pred: {pred_text}\nGT:   {gt_text}", fontsize = 28, wrap = True)
        ax_perp.grid(visible = True, alpha = 0.3)
        if perp_ylim is not None:
            ax_perp.set_ylim(perp_ylim[0], perp_ylim[1])

        ax_empty.axis("off")

        if subtask_caption is None:
            plt.tight_layout()
        else:
            for txt in fig.texts:
                txt.remove()
            fig.text(
                0.5, 0.01, subtask_caption,
                ha = "center", va = "bottom", fontsize = 9,
                family = "monospace", linespacing = 1.5,
            )

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype = np.uint8).reshape(h, w, 4)[:, :, :3]
        video_frames.append(buf.copy())

    plt.close(fig)

    if output_dir is not None:
        import imageio

        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_")
        out_path = os.path.join(output_dir, f"{sanitized_key}.mp4")
        h, w = video_frames[0].shape[:2]
        # imageio requires both dims divisible by 16 (macro_block_size).
        h_crop = h - (h % 16)
        w_crop = w - (w % 16)
        cropped = [f[:h_crop, :w_crop] for f in video_frames]
        imageio.mimsave(out_path, cropped, format = "mp4", fps = fps, codec = "libx264", quality = 8)
        logger.info(f"Saved subtask video to {out_path}")
        return out_path

    import wandb

    video_array = np.stack(video_frames).transpose(0, 3, 1, 2)
    return wandb.Video(video_array, fps = fps, format = "gif")


def _action_grad_sq_norms(
    model,
    frames: list[dict],
    chosen_actions: list[np.ndarray] | None,
    *,
    strip_subtask_id: bool,
    batch_size: int,
) -> list[float]:
    """Per-frame ``||grad_a Q(s, a)||^2`` at the action the reported value was computed at.

    ``chosen_actions`` overrides each frame's ``actions`` (used when the value came from
    the best cached counterfactual, so the gradient is taken where the value was read);
    None differentiates at the dataset action. ``strip_subtask_id`` drops the ground-truth
    id so a categorical critic resolves its own, matching the value pass.
    """
    grads: list[float] = []
    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start : batch_start + batch_size]
        if chosen_actions is not None:
            batch = [
                {**f, "actions": chosen_actions[batch_start + i]} for i, f in enumerate(batch)
            ]
        padded = batch + [batch[-1]] * (batch_size - len(batch))
        obs, act = get_obs_and_action(padded, prefix = "", action_conditioned = True)
        if strip_subtask_id:
            obs = dataclasses.replace(obs, subtask_id = None)
        grad_sq_norm, _ = jax.device_get(_jitted_action_grad_sq_norm(model, obs, act))
        grads.extend(float(g) for g in np.asarray(grad_sq_norm)[: len(batch)])
    return grads


def _image_grad_sq_norms(
    model,
    frames: list[dict],
    chosen_actions: list[np.ndarray] | None,
    *,
    action_conditioned: bool,
    strip_subtask_id: bool,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Per-frame, per-camera ``||grad_I Q||^2`` at the action the reported value was read at.

    ``chosen_actions`` and ``strip_subtask_id`` follow ``_action_grad_sq_norms``: the gradient
    is taken at the action the value was computed at, under the prompt / subtask id the value
    used. Returns ``(sq_norm [T, C], camera names)``.
    """
    cameras = list(frames[0]["image"].keys())
    sq_norms: list[np.ndarray] = []
    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start : batch_start + batch_size]
        if chosen_actions is not None:
            batch = [{**f, "actions": chosen_actions[batch_start + i]} for i, f in enumerate(batch)]
        padded = batch + [batch[-1]] * (batch_size - len(batch))
        obs, act = get_obs_and_action(padded, prefix = "", action_conditioned = action_conditioned)
        if strip_subtask_id:
            obs = dataclasses.replace(obs, subtask_id = None)
        sq_norm = jax.device_get(_jitted_image_grad_sq_norm(model, obs, act))
        sq_norms.append(np.stack([np.asarray(sq_norm[c]) for c in cameras], axis = 1)[: len(batch)])
    return np.concatenate(sq_norms, axis = 0), cameras


def _chosen_actions_from_candidates(frames: list[dict], candidate_values: list) -> list | None:
    """Per-frame best cached action: the candidate whose Q was highest.

    This is the point the value was read at under --counterfactual-value-action, so it is
    also where the gradient must be taken — cached_action[0] would report the slope at an
    arbitrary candidate instead.
    """
    if not candidate_values:
        return None
    return [
        np.asarray(f["counterfactual_actions"])[int(np.argmax(candidate_values[i]))]
        for i, f in enumerate(frames)
    ]


def _write_eval_npz(
    npz_dir: str,
    traj_idx: str,
    *,
    values,
    candidate_values,
    action_grad,
    mc_returns,
    boundaries,
    num_frames: int,
    attn_scores = None,
    attn_modalities = None,
    sample_indices = None,
    predicted_texts = None,
    gt_texts = None,
    predicted_ids = None,
    gt_ids = None,
    image_grad_sq_norm = None,
    image_grad_cameras = None,
) -> str:
    """Write one trajectory's npz. Single writer for every eval path, so the payload cannot
    drift between critic families.

    Subtask fields are omitted rather than faked for critics that predict no subtask: a
    whole-task critic has no subtask head, and an empty array says so unambiguously. The
    same goes for attention: a network with no per-modality CLS attention (the ResNet)
    writes no ``attn_scores`` key at all.
    """
    import io as _io

    from etils import epath

    payload = {
        "predicted_values": np.asarray(values, dtype = np.float64),
        # [num_frames, num_samples] when the cached action store was joined, so the analysis
        # can take max or a random candidate without re-running the eval.
        "candidate_values": np.asarray(candidate_values, dtype = np.float64),
        "action_grad_sq_norm": np.asarray(action_grad, dtype = np.float64),
        "mc_returns": np.asarray(mc_returns, dtype = np.float64),
        "subtask_boundaries": np.asarray(boundaries, dtype = np.int32),
        "num_frames": np.int32(num_frames),
    }
    if attn_scores:
        # [num_frames, num_modalities]: per-group CLS attention at the dataset action, under
        # the same prompt the reported value used. ``attn_modalities`` names the columns.
        scores = np.stack(attn_scores, axis = 0).astype(np.float64)
        if scores.shape[0] != num_frames:
            raise ValueError(f"attn_scores has {scores.shape[0]} rows for {num_frames} frames")
        payload["attn_scores"] = scores
        payload["attn_modalities"] = np.asarray(attn_modalities)
    if sample_indices is not None:
        payload["subtask_pred_t"] = np.asarray(sample_indices, dtype = np.int32)
        payload["predicted_subtasks"] = np.asarray([(predicted_texts[t] or "") for t in sample_indices])
        payload["gt_subtasks"] = np.asarray([(gt_texts[t] or "") for t in sample_indices])
    if predicted_ids is not None:
        payload["predicted_subtask_ids"] = np.asarray(predicted_ids, dtype = np.int32)
    if gt_ids is not None:
        payload["gt_subtask_ids"] = np.asarray(gt_ids, dtype = np.int32)
    if image_grad_sq_norm is not None:
        # [num_frames, num_cameras]; ``image_grad_cameras`` labels the camera axis.
        payload["image_grad_sq_norm"] = np.asarray(image_grad_sq_norm, dtype = np.float64)
        payload["image_grad_cameras"] = np.asarray(image_grad_cameras)

    npz_path = f"{npz_dir.rstrip('/')}/{traj_idx}.npz"
    # Serialize to bytes, then write via epath so a gs:// dir works as well as local/NFS
    # (np.savez can't write to gs:// directly).
    _buf = _io.BytesIO()
    np.savez(_buf, **payload)
    _out = epath.Path(npz_path)
    _out.parent.mkdir(parents = True, exist_ok = True)
    _out.write_bytes(_buf.getvalue())
    return npz_path


def _forward_fill_subtask_predictions(
    result,
    num_frames: int,
) -> tuple[list[str | None], list[str | None], list[float]]:
    """Per-frame display state: frames between samples show the most recent reading.

    A per-frame critic fills every slot already, making this a no-op there.
    """
    predicted_texts = list(result.predicted_texts)
    gt_texts = list(result.gt_texts)
    perplexities = list(result.perplexities)
    last_pred = last_gt = None
    last_perp = float("nan")
    for t in range(num_frames):
        if predicted_texts[t] is None:
            predicted_texts[t] = last_pred
        if gt_texts[t] is None:
            gt_texts[t] = last_gt
        if perplexities[t] is None:
            perplexities[t] = last_perp
        last_pred, last_gt, last_perp = predicted_texts[t], gt_texts[t], perplexities[t]
    return predicted_texts, gt_texts, perplexities


def _log_subtask_accuracy(
    sample_indices: list[int],
    predicted_texts: list[str | None],
    gt_texts: list[str | None],
) -> None:
    """Predicted-vs-GT agreement over the sampled frames.

    Exact match after ``normalize_subtask_text``; for a categorical critic the texts are
    vocab entries, so this is identical to comparing ids, and for a decoding critic it is
    stricter — free-form text can miss by a word where an argmax cannot.
    """
    scored = [t for t in sample_indices if predicted_texts[t] is not None and gt_texts[t]]
    if not scored:
        logger.warning("  Subtask accuracy: no sampled frame carried a GT subtask.")
        return
    num_correct = sum(
        1 for t in scored
        if _transforms.normalize_subtask_text(predicted_texts[t]).casefold()
        == _transforms.normalize_subtask_text(gt_texts[t]).casefold()
    )
    logger.info(
        f"  Subtask accuracy ({len(scored)} sampled frames): "
        f"{num_correct}/{len(scored)} = {num_correct / len(scored):.3f}"
    )


def _run_trajectory_evaluation(
    model,
    train_module,
    val_tokenizer,
    cache_dir: str,
    image_size: tuple[int, int],
    eval_config: EvalConfig,
    action_conditioned: bool,
    *,
    subtask_mode: bool,
    override_prompt: tuple[np.ndarray, np.ndarray] | None,
    mesh = None,
) -> None:
    """Score every cached trajectory once and derive every output from that one pass.

    The per-trajectory pipeline is fixed; the config only switches steps on and off, so no
    output depends on a setting it has nothing to do with:

    1. decode the cached images and apply the prompt override, if any;
    2. subtask mode only: predict the subtask (decoded text or classifier argmax) and,
       under --condition-on-decoded-subtask, rebuild the prompts the value pass sees;
    3. the value pass — ``predict_values`` with every variant the cache supports (negative
       prompt, random / shuffled / counterfactual actions, attention);
    4. the action-gradient pass at the action the value was read at, when requested;
    5. rank 0: the npz (--subtask-npz-dir), the subtask accuracy and video (subtask mode),
       and the standard validation plots, rendered from step 3 whatever the other steps did.

    Everything that issues JIT'd SPMD collectives — steps 2 to 4 — runs on EVERY rank in
    lockstep; otherwise non-rank-0 hosts wait at the next barrier for collectives that never
    fire. Only logging, the npz write and rendering are gated to rank 0.
    """
    is_rank0 = jax.process_index() == 0
    batch_size = eval_config.batch_size

    split_traj_frames, traj_to_repo_ep, ep_subtasks = _split_trajectory_frames(_load_cached_trajectories(cache_dir))
    if not split_traj_frames:
        if is_rank0:
            logger.warning(f"No cached trajectories found in {cache_dir}.")
        return

    npz_mode = eval_config.subtask_npz_dir is not None
    grad_mode = npz_mode and eval_config.action_grad_norm and action_conditioned
    image_grad_mode = npz_mode and eval_config.image_grad_norm

    if is_rank0:
        if not subtask_mode:
            subtask_source = "none (cached prompt)"
        else:
            family = "categorical id" if val_tokenizer is None else "decoded text"
            conditioning = "predicted" if eval_config.condition_on_decoded_subtask else "ground-truth"
            subtask_source = f"{family}, value conditioned on {conditioning}"
        logger.info(
            "Trajectory evaluation | subtask: %s | value action source: %s | action grad: %s | "
            "image grad: %s | prompt override: %s | npz: %s",
            subtask_source,
            "max over cached actions" if eval_config.counterfactual_value_action else "dataset",
            "on" if grad_mode else "off",
            "on" if image_grad_mode else "off",
            "on" if override_prompt is not None else "off",
            eval_config.subtask_npz_dir or "off",
        )
        if eval_config.action_grad_norm and not npz_mode:
            logger.warning("--action-grad-norm set without --subtask-npz-dir: nothing to write it to.")
        if eval_config.action_grad_norm and not action_conditioned:
            logger.warning("--action-grad-norm set on a state-only critic: no action to differentiate.")
        if eval_config.image_grad_norm and not npz_mode:
            logger.warning("--image-grad-norm set without --subtask-npz-dir: nothing to write it to.")
    # Plots and videos are rendered only where something consumes them: a wandb run or
    # --output-dir.
    render_outputs = eval_config.wandb_logging or eval_config.output_dir is not None

    # Accumulators for the standard validation plots, one entry per trajectory segment.
    plot_predictions: dict[str, list[float]] = {}
    plot_predictions_neg: dict[str, list[float]] = {}
    plot_predictions_random: dict[str, list[float]] = {}
    plot_predictions_counterfactual: dict[str, list[float]] = {}
    plot_predictions_shuffled: dict[str, list[float]] = {}
    plot_attn_scores: dict[str, list[np.ndarray]] = {}
    ep_mc_returns: dict[str, list[float]] = {}
    ep_frame_images: dict[str, list[np.ndarray]] = {}
    ep_fps: dict[str, int] = {}
    ep_include_masks: dict[str, list[bool]] = {}
    ep_negative_subtasks: dict[str, list[str]] = {}
    subtask_videos: dict[str, object] = {}

    # Pop each trajectory so its decoded frames are released before the next one loads;
    # the plot accumulators keep only the stacked camera images.
    for traj_idx in list(split_traj_frames):
        frames = split_traj_frames.pop(traj_idx)
        decode_episode_images(frames, image_size)
        if override_prompt is not None:
            apply_override_prompt(frames, override_prompt)
        inject_shuffled_actions(frames, action_conditioned = action_conditioned)

        repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]
        mc_returns = [float(np.asarray(f["mc_return"])) for f in frames]
        if eval_config.allow_unannotated_episodes:
            mc_returns = [
                value if bool(frame["has_subtask_annotations"]) else float("nan")
                for value, frame in zip(mc_returns, frames, strict = True)
            ]
        fps = int(frames[0]["fps"])

        if is_rank0:
            logger.info(
                f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}{part_suffix}): {len(frames)} frames."
            )

        subtask_result = None
        if subtask_mode:
            subtask_result = predict_values_with_subtasks(
                model, frames, traj_idx,
                tokenizer = val_tokenizer,
                stride = eval_config.subtask_decode_stride,
                use_predicted_subtask = eval_config.condition_on_decoded_subtask,
                use_counterfactual_actions = eval_config.counterfactual_value_action,
                action_conditioned = action_conditioned,
                score_gt_perplexity = eval_config.score_gt_perplexity,
                batch_size = batch_size,
                mesh = mesh,
                is_rank0 = is_rank0,
            )
            value_passes = subtask_result.value_passes
            values = subtask_result.values
            value_frames = subtask_result.value_frames
        else:
            value_passes = predict_trajectory_values(
                model, frames, traj_idx, action_conditioned = action_conditioned, batch_size = batch_size, mesh = mesh,
            )
            values = value_passes.reported(
                use_counterfactual_actions = eval_config.counterfactual_value_action, traj_key = traj_idx,
            )
            value_frames = frames

        # The gradient is taken at the action the value was read at — the best cached
        # candidate under --counterfactual-value-action, else the dataset action — and
        # against the prompt the value used (``value_frames``).
        action_grad: list[float] = []
        if grad_mode:
            action_grad = _action_grad_sq_norms(
                model, value_frames,
                _chosen_actions_from_candidates(value_frames, value_passes.candidate_values),
                strip_subtask_id = subtask_mode and eval_config.condition_on_decoded_subtask,
                batch_size = batch_size,
            )

        # Image gradients at the same action and prompt the value used. Unlike the action
        # gradient above this is explicit about the action source: the argmax cached candidate
        # only under --counterfactual-value-action, else the dataset action.
        image_grad: dict = {}
        if image_grad_mode:
            chosen = (
                _chosen_actions_from_candidates(value_frames, value_passes.candidate_values)
                if eval_config.counterfactual_value_action else None
            )
            sq_norm, cameras = _image_grad_sq_norms(
                model, value_frames, chosen,
                action_conditioned = action_conditioned,
                strip_subtask_id = subtask_mode and eval_config.condition_on_decoded_subtask,
                batch_size = batch_size,
            )
            image_grad = {"image_grad_sq_norm": sq_norm, "image_grad_cameras": cameras}
            if is_rank0:
                logger.info(f"Computed image gradients for {len(sq_norm)} frames x {len(cameras)} cameras")

        if not is_rank0:
            continue

        plot_predictions[traj_idx] = value_passes.dataset
        plot_predictions_neg[traj_idx] = value_passes.negative_prompt
        plot_predictions_random[traj_idx] = value_passes.random_actions
        plot_predictions_counterfactual[traj_idx] = value_passes.counterfactual_actions
        plot_predictions_shuffled[traj_idx] = value_passes.shuffled_actions
        plot_attn_scores[traj_idx] = value_passes.attn_scores
        ep_mc_returns[traj_idx] = mc_returns
        ep_fps[traj_idx] = fps
        ep_include_masks[traj_idx] = [bool(f.get("include_subtask", True)) for f in frames]
        ep_negative_subtasks[traj_idx] = (
            count_subtask_segments(frames, prefix = "negative_")[2]
            if "negative_subtask_1_text" in frames[0] else []
        )
        frame_images = [
            np.stack([
                np.asarray(f["image"]["left_wrist_0_rgb"]),
                np.asarray(f["image"]["right_wrist_0_rgb"]),
                np.asarray(f["image"]["base_0_rgb"]),
            ])
            for f in frames
        ]
        ep_frame_images[traj_idx] = frame_images

        boundaries = subtask_boundary_indices(frames)
        npz_subtask_fields: dict = {}
        if subtask_result is not None:
            predicted_texts, gt_texts, perplexities = _forward_fill_subtask_predictions(subtask_result, len(frames))
            _log_subtask_accuracy(subtask_result.sample_indices, predicted_texts, gt_texts)
            npz_subtask_fields = {
                "sample_indices": subtask_result.sample_indices,
                "predicted_texts": predicted_texts,
                "gt_texts": gt_texts,
                "predicted_ids": subtask_result.predicted_ids,
                "gt_ids": subtask_result.gt_ids,
            }
            plot_key = f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}_subtask"
            if render_outputs:
                subtask_videos[plot_key] = _render_subtask_video(
                    mc_returns = mc_returns,
                    predicted_values = values,
                    perplexities = perplexities,
                    predicted_texts = predicted_texts,
                    gt_texts = gt_texts,
                    frame_images = frame_images,
                    fps = fps,
                    ep_idx = ep_idx,
                    subtask_texts = subtask_segment_labels(boundaries, gt_texts, len(frames)),
                    output_dir = eval_config.output_dir,
                    plot_key = plot_key,
                    condition_on_decoded = eval_config.condition_on_decoded_subtask,
                )

        if npz_mode:
            npz_path = _write_eval_npz(
                eval_config.subtask_npz_dir, traj_idx,
                values = values,
                candidate_values = value_passes.candidate_values,
                action_grad = action_grad,
                mc_returns = mc_returns,
                boundaries = boundaries,
                num_frames = len(frames),
                attn_scores = value_passes.attn_scores,
                attn_modalities = (
                    train_module.attn_modality_labels(
                        len(value_passes.attn_scores[0]), action_conditioned = action_conditioned,
                    )
                    if value_passes.attn_scores else None
                ),
                **npz_subtask_fields,
                **image_grad,
            )
            logger.info(f"Saved eval npz to {npz_path}: {len(values)} value preds.")

        del frames, value_frames, subtask_result, value_passes
        gc.collect()

    if not is_rank0 or not render_outputs:
        return

    if eval_config.output_dir is None and subtask_videos:
        import wandb

        wandb.log(subtask_videos)

    train_module.start_render_thread(
        plot_predictions, plot_predictions_neg, plot_predictions_random,
        plot_predictions_shuffled, plot_attn_scores,
        ep_mc_returns, ep_frame_images, ep_fps, ep_include_masks,
        ep_subtasks, ep_negative_subtasks,
        traj_to_repo_ep, action_conditioned, 0,
        all_predictions_counterfactual = plot_predictions_counterfactual,
        output_dir = eval_config.output_dir,
    )


def _get_critic_network(model):
    """Return the value network, tolerating CQL critics that expose ``q_network``.

    SARSA/MC value functions store the encoder as ``model.network``; CQL stores it
    as ``model.q_network``. The rest of this script assumes ``model.network``.
    """
    net = getattr(model, "network", None)
    if net is None:
        net = getattr(model, "q_network", None)
    if net is None:
        raise ValueError("Critic model exposes neither .network nor .q_network.")
    return net


@nnx.jit
def _jitted_action_grad_sq_norm(model, obs, act):
    """Per-sample ``||grad_a Q(s, a)||^2`` and Q for an action-conditioned critic.

    Summing Q over the batch and differentiating w.r.t. the batched action is a
    standard trick: each sample's Q depends only on its own action, so the grad
    of the sum w.r.t. ``act`` yields per-sample gradients ``[B, action_horizon,
    action_dim]``. The squared L2 norm is taken over the full action chunk.
    """
    def _q_sum(a):
        out = model.compute_value(obs, a)
        q = out[0] if isinstance(out, tuple) else out
        return jnp.sum(q), q

    grads, q = jax.grad(_q_sum, has_aux = True)(act)
    grad_sq_norm = jnp.sum(grads ** 2, axis = tuple(range(1, grads.ndim)))
    return grad_sq_norm, q


@nnx.jit
def _jitted_image_grad_sq_norm(model, obs, act):
    """Per-camera ``||grad_I Q||^2``: the batch-sum trick of ``_jitted_action_grad_sq_norm``,
    differentiated w.r.t. the dict of camera images (in the network's [-1, 1] space) instead
    of the action; the squared L2 norm is over the whole H x W x C image."""
    def _q_sum(images):
        out = model.compute_value(dataclasses.replace(obs, images = images), act)
        q = out[0] if isinstance(out, tuple) else out
        return jnp.sum(q)

    grads = jax.grad(_q_sum)(obs.images)
    return {camera: jnp.sum(grad ** 2, axis = tuple(range(1, grad.ndim))) for camera, grad in grads.items()}


def main(eval_config: EvalConfig):
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

    if eval_config.split not in {"train", "val"}:
        raise ValueError(f"--split must be 'train' or 'val', got {eval_config.split!r}.")
    if eval_config.val_episode_indices and eval_config.val_episodes:
        raise ValueError("--val-episode-indices and --val-episodes are mutually exclusive.")
    if eval_config.allow_unannotated_episodes:
        if not eval_config.condition_on_decoded_subtask or eval_config.disable_subtask_decoding:
            raise ValueError("Unannotated episodes require --condition-on-decoded-subtask and subtask decoding.")
        if eval_config.counterfactual_value_action and eval_config.counterfactual_action_store_dir is None:
            raise ValueError("Unannotated counterfactual evaluation requires an explicit action store for these episodes.")
        eval_config = dataclasses.replace(eval_config, score_gt_perplexity = False)
    if not eval_config.wandb_logging and eval_config.subtask_npz_dir is None:
        raise ValueError(
            "--no-wandb-logging without --subtask-npz-dir: nothing would be logged or written, "
            "so the evaluation would be wasted compute. Set --subtask-npz-dir or re-enable wandb."
        )

    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    train_module = load_train_module()
    def _config_override(config):
        config = dataclasses.replace(config, include_repos = eval_config.include_repos)
        if eval_config.allow_unannotated_episodes:
            if (
                not isinstance(config.data, _config.Hdf5RldsDataConfig)
                or config.data.prompt_mode != "task_description_predict_current_subtask"
            ):
                raise ValueError("Unannotated evaluation requires an HDF5 data config with subtask text prediction.")
            config = dataclasses.replace(
                config,
                data = dataclasses.replace(
                    config.data,
                    counterfactual_action_store_dir = eval_config.counterfactual_action_store_dir,
                ),
            )
        if eval_config.counterfactual_action_store_dir is not None:
            config = dataclasses.replace(
                config,
                data = dataclasses.replace(config.data, counterfactual_action_store_dir = eval_config.counterfactual_action_store_dir),
            )
        if eval_config.rlds_data_dir is not None:
            config = dataclasses.replace(
                config,
                data = dataclasses.replace(config.data, rlds_data_dir = eval_config.rlds_data_dir),
            )
        return dataclasses.replace(
            config,
            num_val_trajectories = eval_config.num_trajectories,
        )

    resolved_fsdp_devices = eval_config.fsdp_devices or jax.device_count()
    model, _, config, critic_step = load_critic(
        eval_config.config_name,
        eval_config.checkpoint_path,
        fine_tune = eval_config.fine_tune,
        step = eval_config.step,
        config_override = _config_override,
        fsdp_devices = resolved_fsdp_devices,
    )
    # Rebuild the same mesh load_critic used so we can shard inference inputs
    # consistently with the restored param sharding. With fsdp_devices=1 this is
    # (device_count, 1) → data-parallel: replicated params, batch-axis-split inputs.
    # Single-host only: on multi-host TPU, device_put of a host-local array across
    # the global mesh is invalid, so leave inputs unsharded there and rely on the
    # FSDP-sharded params (the original multi-host path).
    inference_mesh = _sharding.make_mesh(resolved_fsdp_devices) if jax.process_count() == 1 else None
    action_conditioned = _get_critic_network(model).action_conditioned
    logger.info(f"Loaded model from {eval_config.checkpoint_path}, action_conditioned={action_conditioned}")

    data_config = config.data.create(config.assets_dirs, config.model)
    if eval_config.allow_unannotated_episodes:
        data_config = dataclasses.replace(
            data_config,
            rlds_kwargs = {
                **data_config.rlds_kwargs,
                "allow_unannotated_episodes": True,
                "filter_n": None,
                "td_n": None,
            },
        )
    action_horizon = config.action_horizon or config.model.action_horizon
    val_tokenizer = config.data._get_critic_tokenizer(config.model)

    val_input_transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
        *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
        *(
            transform for transform in data_config.model_transforms.inputs
            if not (eval_config.allow_unannotated_episodes and isinstance(transform, _transforms.SubtaskTextToId))
        ),
        _config.AddValidationVariants(
            val_tokenizer,
            use_quantile_norm = data_config.use_quantile_norm,
            include_negative = not eval_config.allow_unannotated_episodes,
        ),
    ])

    cache_dir = _resolve_eval_cache_dir(eval_config)
    split = data_config.val_split if eval_config.split == "val" else eval_config.split

    cache_complete = False
    # epath.iterdir handles gs:// cache dirs (virtual prefix) as well as local/NFS.
    from etils import epath

    existing_pkls: set[str] = set()
    try:
        existing_pkls = {p.name for p in epath.Path(cache_dir).iterdir() if p.name.endswith(".pkl")}
    except (FileNotFoundError, NotADirectoryError, OSError):
        existing_pkls = set()
    if existing_pkls:
        required_pkls = {f"{repo.replace('/', '__')}.pkl" for repo in config.include_repos}
        if required_pkls.issubset(existing_pkls) and len(existing_pkls) >= eval_config.num_trajectories:
            cache_complete = True
            logger.info(f"Cache complete at {cache_dir} ({len(existing_pkls)} pkls); skipping create_rlds_dataset")

    # cache_val_episodes is multi-worker by design: process 0 fills the
    # full-coverage slots, workers > 0 cache the include_repos. Run it on ALL
    # workers (matching train_value_function.py) — gating to process 0 drops
    # the per-worker include_repos coverage.
    if not cache_complete:
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            data_config,
            action_horizon,
            config.batch_size,
            split = split,
            shuffle = False,
            return_trajectories = True,
        )
        cache_val_episodes(
            val_trajectory_dataset,
            eval_config.num_trajectories,
            cache_dir,
            include_repos = config.include_repos,
            save_only = True,
            input_transform = val_input_transform,
            # Mirrors train_value_function.py's FT path: a fine-tune config targets a
            # single-repo dataset, so without this every val episode collapses onto one
            # repo_key and only one trajectory is ever cached.
            allow_duplicate_repos = eval_config.fine_tune is not None or eval_config.allow_duplicate_repos,
            episode_indices = eval_config.val_episode_indices or None,
            episode_keys = _parse_val_episodes(eval_config.val_episodes) or None,
        )
        del val_trajectory_dataset

    if jax.process_count() > 1:
        jax.experimental.multihost_utils.sync_global_devices("eval_cache_write")

    if eval_config.cache_only:
        logger.info("--cache-only set: validation cache populated; exiting before plotting.")
        return

    override_prompt = None
    if eval_config.override_task_prompt is not None:
        critic_prompt_mode = getattr(config.data, "prompt_mode", None)
        if critic_prompt_mode == "task_description_predict_current_subtask":
            # Match the serving/eval path's critic tokenization (best_of_n_policy.py):
            # tokenize the task-description prefix only — empty subtask suffix, no
            # trailing newline. The subtask indices are discarded (cached val frames
            # carry none, so the critic forward runs with subtask_start_index=None).
            override_tokens, override_mask, _, _ = _transforms._tokenize_subtask_prompt(
                val_tokenizer, eval_config.override_task_prompt, "", append_newline = False,
            )
            override_prompt = (override_tokens, override_mask)
        else:
            override_prompt = val_tokenizer.tokenize(eval_config.override_task_prompt)
        logger.info(
            f"Overriding tokenized_prompt with {eval_config.override_task_prompt!r} "
            f"(prompt_mode={critic_prompt_mode!r})"
        )

    if eval_config.wandb_logging and eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.init(
            project = eval_config.project_name,
            name = eval_config.wandb_run_name or f"eval_{eval_config.config_name}",
            config = dataclasses.asdict(eval_config),
        )

    # A critic trained with prompt_mode="task_description_predict_current_subtask" predicts
    # its own subtask, which adds the decode step and the subtask video to the evaluation;
    # every other output (standard plots, npz, gradients) is produced the same way either way.
    critic_prompt_mode = (
        getattr(config.data, "prompt_mode", None)
        or getattr(config.data, "subtask_prompt_mode", None)
    )
    subtask_mode = (
        critic_prompt_mode == "task_description_predict_current_subtask"
        and not eval_config.disable_subtask_decoding
    )
    image_size = data_config.rlds_kwargs.get("image_size", (224, 224))
    _run_trajectory_evaluation(
        model = model,
        train_module = train_module,
        val_tokenizer = val_tokenizer,
        cache_dir = cache_dir,
        image_size = tuple(image_size),
        eval_config = eval_config,
        action_conditioned = action_conditioned,
        subtask_mode = subtask_mode,
        override_prompt = override_prompt,
        mesh = inference_mesh,
    )

    # Keep all hosts alive until rank 0 has fully completed async rendering/logging.
    if (
        jax.process_index() == 0
        and train_module._render_thread is not None
        and train_module._render_thread.is_alive()
    ):
        logger.info("Waiting for render thread to finish")
        train_module._render_thread.join()
        train_module._render_thread = None

    if jax.process_count() > 1:
        logger.info("Waiting at post-render multihost barrier")
        jax.experimental.multihost_utils.sync_global_devices("evaluate_value_function_post_render_join")

    if eval_config.wandb_logging and eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.finish()

    logger.info("Evaluation complete")


if __name__ == "__main__":
    import tyro

    eval_config = tyro.cli(EvalConfig)
    main(eval_config)
