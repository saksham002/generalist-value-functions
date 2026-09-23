"""Eval-time image preparation shared across example clients.

Wraps the per-call resize step so clients stay model-agnostic when the policy
and critic disagree on input shape (e.g. Gemma 4's 480×480 vs the standard
224×224). Reads the server's metadata on construction to decide:

  * `policy_image_size` — target shape for `element["image"]`.
  * `critic_image_size` — target shape for `element["critic_image"]` when the
    server consumes a separate critic image stream; `None` when it doesn't.
  * `expect_critic_images` — whether the server is wired to consume
    `element["critic_image"]`.

Callers pass a `{camera_key: HWC uint8 ndarray}` dict per inference. The helper
returns `{"image": ..., "critic_image": ...}` ready to merge into the websocket
`element` dict.
"""

import dataclasses
from typing import Any

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


_DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)


@dataclasses.dataclass
class EvalImageHelper:
    policy_image_size: tuple[int, int]
    critic_image_size: tuple[int, int] | None
    expect_critic_images: bool

    @classmethod
    def from_client(cls, client: _websocket_client_policy.WebsocketClientPolicy) -> "EvalImageHelper":
        """Auto-detect sizes + expect_critic_images from server metadata."""
        meta: dict[str, Any] = client.get_server_metadata() or {}
        policy_size = _as_size(meta.get("policy_image_size", _DEFAULT_IMAGE_SIZE))
        critic_size_raw = meta.get("critic_image_size")
        critic_size = _as_size(critic_size_raw) if critic_size_raw else None
        expect = bool(meta.get("expect_critic_images", False))
        return cls(
            policy_image_size = policy_size,
            critic_image_size = critic_size,
            expect_critic_images = expect,
        )

    def process_images(self, raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """Resize `raw` (model-side camera key → HWC uint8 image) for the policy
        and, when the server expects it, for the critic.

        Returns a dict with `"image"` (always) and `"critic_image"` (when
        `expect_critic_images=True`). When the critic size matches the policy
        size, the same dict is reused to avoid duplicate resize work.
        """
        out: dict[str, dict[str, np.ndarray]] = {
            "image": _resize_dict(raw, self.policy_image_size),
        }
        if self.expect_critic_images:
            target = self.critic_image_size or self.policy_image_size
            if target == self.policy_image_size:
                out["critic_image"] = out["image"]
            else:
                out["critic_image"] = _resize_dict(raw, target)
        return out


def _resize_dict(raw: dict[str, np.ndarray], size: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = size
    return {
        k: image_tools.convert_to_uint8(image_tools.resize_stretch(v, h, w))
        for k, v in raw.items()
    }


def _as_size(value: Any) -> tuple[int, int]:
    """Normalize a metadata image-size value into a (h, w) tuple of ints."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return (value, value)
    raise TypeError(f"Cannot interpret image size {value!r}")
