import dataclasses
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Task instruction sent as the policy prompt and as the critic's task description.
    prompt: str = "hang the shirt on the rod"


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    def obs_fn() -> dict:
        return _random_observation(args.prompt)

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


def _random_observation(prompt: str) -> dict:
    """A random observation in the format the bimanual EEF policies and critics consume.

    Three 224x224 RGB cameras keyed by the config's ``image_keys``, the 14-D EEF
    proprioceptive state, the policy prompt and the task description the critic
    decodes its subtask from.
    """
    return {
        "image": {
            "base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "state": np.random.rand(14).astype(np.float32),
        "prompt": prompt,
        "task_description": prompt,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
