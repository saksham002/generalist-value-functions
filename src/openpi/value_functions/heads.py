"""Value heads for converting network features to value predictions."""

import dataclasses

import flax.nnx as nnx

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RegressionHeadConfig:
    """Configuration for regression output head."""

    orthogonal_init_scale: float | None = None

    def create(self, feature_dim: int, rng: at.KeyArrayLike) -> "RegressionHead":
        return RegressionHead(feature_dim, self.orthogonal_init_scale, rngs=nnx.Rngs(rng))


class RegressionHead(nnx.Module):
    """Linear projection from features to scalar value."""

    def __init__(
        self,
        feature_dim: int,
        orthogonal_init_scale: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if orthogonal_init_scale is not None:
            kernel_init = nnx.initializers.orthogonal(scale=orthogonal_init_scale)
            self.linear = nnx.Linear(feature_dim, 1, kernel_init=kernel_init, rngs=rngs)
        else:
            self.linear = nnx.Linear(feature_dim, 1, rngs=rngs)

    def __call__(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b"]:
        """Compute scalar value from features."""
        return self.linear(features).squeeze(-1)


# Type aliases kept for the objective signatures; the regression head is the only head.
ValueHead = RegressionHead
HeadConfig = RegressionHeadConfig
