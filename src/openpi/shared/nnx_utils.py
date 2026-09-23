from collections.abc import Callable
import dataclasses
import functools
import inspect
import math
import re
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax

import openpi.shared.array_typing as at

P = ParamSpec("P")
R = TypeVar("R")


def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """A higher-order function to JIT-compile `nnx.Module` methods, freezing the module's state in the process.

    Why not `nnx.jit`? For some reason, naively applying `nnx.jit` to `nnx.Module` methods, bound or unbound, uses much
    more memory than necessary. I'm guessing it has something to do with the fact that it must keep track of module
    mutations. Also, `nnx.jit` has some inherent overhead compared to a standard `jax.jit`, since every call must
    traverse the NNX module graph. See https://github.com/google/flax/discussions/4224 for details.

    `module_jit` is an alternative that avoids these issues by freezing the module's state. The function returned by
    `module_jit` acts exactly like the original method, except that the state of the module is frozen to whatever it was
    when `module_jit` was called. Mutations to the module within `meth` are still allowed, but they will be discarded
    after the method call completes.
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


@dataclasses.dataclass(frozen=True)
class PathRegex:
    """NNX Filter that matches paths using a regex.

    By default, paths are joined with a `/` separator. This can be overridden by setting the `sep` argument.
    """

    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """Apply a function to the leaves of the state that match the filter."""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)


def _normalize_key_part(key_part: Any) -> Any:
    """Normalize key parts for fallback matching across int/string numeric keys."""
    if isinstance(key_part, str):
        try:
            return int(key_part)
        except ValueError:
            return key_part
    return key_part


def _normalize_key_path(key_path: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_normalize_key_part(part) for part in key_path)


def replace_state_from_pure_dict_numeric_key_compat(state: nnx.State, pure_dict: at.Params) -> None:
    """Replace NNX state values while working around Flax numeric-key coercion.

    Flax NNX's `state.replace_by_pure_dict` converts every path part with `int(...)`
    before lookup. That breaks models whose state legitimately contains numeric string
    keys.
    """
    current_flat = state.flat_state()
    incoming_flat = traverse_util.flatten_dict(pure_dict)

    updates: dict[tuple[Any, ...], Any] = {}
    unresolved: dict[tuple[Any, ...], Any] = {}

    for key_path, value in incoming_flat.items():
        if key_path in current_flat:
            updates[key_path] = value
        else:
            unresolved[key_path] = value

    if unresolved:
        norm_to_state_key: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        ambiguous_state_norm_keys: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}

        for state_key in current_flat:
            norm_key = _normalize_key_path(state_key)
            existing = norm_to_state_key.get(norm_key)
            if existing is None:
                norm_to_state_key[norm_key] = state_key
            elif existing != state_key:
                ambiguous_state_norm_keys.setdefault(norm_key, {existing}).add(state_key)

        if ambiguous_state_norm_keys:
            ambiguous_samples = sorted(
                (norm_key, tuple(sorted(paths))) for norm_key, paths in ambiguous_state_norm_keys.items()
            )[:3]
            raise ValueError(
                f"Ambiguous normalized state keys prevent safe replacement: {ambiguous_samples}. "
                "Resolve conflicting key types in the model state."
            )

        unresolved_after: dict[tuple[Any, ...], Any] = {}
        normalized_target_sources: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        for unresolved_key, value in unresolved.items():
            norm_key = _normalize_key_path(unresolved_key)
            state_key = norm_to_state_key.get(norm_key)
            if state_key is None:
                unresolved_after[unresolved_key] = value
                continue

            existing_source = normalized_target_sources.get(state_key)
            if existing_source is not None and existing_source != unresolved_key:
                raise ValueError(
                    f"Ambiguous normalized incoming keys map to the same state key {state_key}: "
                    f"{existing_source} and {unresolved_key}."
                )

            normalized_target_sources[state_key] = unresolved_key
            updates[state_key] = value

        unresolved = unresolved_after

    if unresolved:
        missing_samples = sorted(unresolved.keys())[:3]
        raise ValueError(f"Keys in pure_dict not available in state: {missing_samples}")

    for key_path, value in updates.items():
        leaf = current_flat[key_path]
        current_flat[key_path] = leaf.replace(value) if hasattr(leaf, "replace") else value

    state.update(nnx.State.from_flat_path(current_flat))


def count_parameters(params: nnx.State | at.Params) -> int:
    """Count the total number of parameters in an nnx.State or param dict."""
    total_params = 0
    param_dict = params.to_pure_dict() if isinstance(params, nnx.State) else params
    for leaf in jax.tree_util.tree_leaves(param_dict):
        if not hasattr(leaf, "shape"):
            raise TypeError(f"Expected parameter leaf to have shape, got {type(leaf)}")
        total_params += math.prod(leaf.shape)
    return total_params
