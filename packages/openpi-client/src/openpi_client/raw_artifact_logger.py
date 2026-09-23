"""Raw per-episode artifact capture shared across example eval clients.

Companion to ``eval_image_helper`` (eval-time image prep). Gated behind a
``--save-raw`` flag in the example clients, ``RawArtifactLogger`` writes
website/demo-ready raw artifacts per episode, bucketed by outcome:

  * one mp4 per top camera, at the env control-freq fps, with the env's frames
    written unresized and unmodified;
  * a ``q_values.npz`` mapping each infer timestep to the N candidate Q-values.

This is distinct from each client's ``VideoLogger`` (which renders the
downsampled annotated mosaic at the replan cadence); this captures the raw
control-freq camera streams instead.

``imageio`` is imported lazily inside ``finish_episode`` so importing this module
does not add a hard dependency to the minimal ``openpi-client`` package (which
depends only on numpy).
"""

from __future__ import annotations

import logging
import os

import numpy as np


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Return a uint8 H x W x 3 copy of an env camera frame without resizing.

    uint8 frames pass through verbatim (copied, so later in-place reuse of the
    env's render buffer cannot corrupt a recorded frame). Float frames are
    scaled from [0, 1] when needed and clipped to [0, 255]."""
    f = np.asarray(frame)
    if f.dtype == np.uint8:
        return np.ascontiguousarray(f).copy()
    f = f.astype(np.float32)
    if float(f.max(initial = 0.0)) <= 1.0 + 1e-6:
        f = f * 255.0
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))


class RawArtifactLogger:
    """Per-episode RAW capture for website/demo assets, gated behind --save-raw.

    Writes, per episode and bucketed by outcome, under
    ``<output_dir>/<successes|failures>/episode_<idx>/``:

      ``<cam>.mp4``    one video per top camera, at the env control-freq fps,
                       with the env's frames written unresized and unmodified
                       (``macro_block_size=2`` so even-dimensioned env renders
                       pass through byte-for-byte; only odd dims get a 1px nudge,
                       which libx264 requires).
      ``q_values.npz`` ``steps`` (K,), ``q_values`` (K, N), ``subtasks`` (K,) —
                       the N candidate Q-values at each infer timestep.

    Complements VideoLogger, which writes the downsampled annotated mosaic at the
    replan cadence; this captures the raw control-freq camera streams instead.
    """

    def __init__(self, output_dir: str, fps: float, cam_names: list[str]) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.cam_names = list(cam_names)
        os.makedirs(output_dir, exist_ok = True)
        self._reset()

    def _reset(self) -> None:
        self._frames: dict[str, list[np.ndarray]] = {c: [] for c in self.cam_names}
        self._val_steps: list[int] = []
        self._q_values: list[np.ndarray] = []
        self._subtasks: list[str] = []
        self._episode_idx: int | None = None

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx

    def record_frame(self, images: dict[str, np.ndarray]) -> None:
        for cam in self.cam_names:
            frame = images.get(cam)
            if frame is not None:
                self._frames[cam].append(_frame_to_uint8(frame))

    def record_values(self, q_values: np.ndarray | None, t: int, subtask: str = "") -> None:
        self._val_steps.append(int(t))
        if q_values is None:
            self._q_values.append(np.zeros((0,), dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._subtasks.append(subtask)

    def finish_episode(self, success: bool = False) -> None:
        if self._episode_idx is None:
            return
        # Lazy import: keeps `openpi-client` importable without imageio (the
        # minimal client depends only on numpy); eval clients always have it.
        import imageio

        sub = "successes" if success else "failures"
        ep_dir = os.path.join(self.output_dir, sub, f"episode_{self._episode_idx}")
        os.makedirs(ep_dir, exist_ok = True)
        for cam in self.cam_names:
            frames = self._frames[cam]
            if not frames:
                continue
            out_path = os.path.join(ep_dir, f"{cam.replace('/', '_')}.mp4")
            # macro_block_size=2 avoids libx264's default divisible-by-16 resize:
            # even-dimensioned frames (all standard env renders, e.g. 224x224) are
            # written unmodified; only odd dimensions get nudged by 1px (libx264
            # requires even W/H). imageio bundles libx264 (the system ffmpeg on
            # this cluster lacks it — see CLAUDE.md > "Saving Videos").
            imageio.mimsave(
                out_path, frames, format = "mp4", fps = self.fps,
                codec = "libx264", quality = 8, macro_block_size = 2,
            )
            logging.info(f"Saved raw camera video: {out_path}")
        # Q-values are produced once per infer call (the replan ticks); align
        # them to their env timestep. N can vary across calls (variable BoN);
        # pad short rows with NaN so the matrix is rectangular.
        steps = np.asarray(self._val_steps, dtype = np.int64)
        widths = [v.shape[0] for v in self._q_values]
        num_samples = max(widths) if widths else 0
        if num_samples > 0:
            q_matrix = np.full((len(self._q_values), num_samples), np.nan, dtype = np.float32)
            for i, v in enumerate(self._q_values):
                q_matrix[i, : v.shape[0]] = v
        else:
            q_matrix = np.zeros((len(self._q_values), 0), dtype = np.float32)
        np.savez(
            os.path.join(ep_dir, "q_values.npz"),
            steps = steps, q_values = q_matrix, subtasks = np.asarray(self._subtasks),
        )
        logging.info(f"Saved q_values.npz ({steps.shape[0]} infer steps x {num_samples} values): {ep_dir}")
