# Normalization statistics

States, action chunks and the cached candidate actions are normalized with quantile
normalization: every dimension is mapped through its 1st/99th percentiles,
`x_norm = 2 * (x - q01) / (q99 - q01) - 1`, and then clipped to `[-1.25, 1.25]`
(`clip_normalized_bounds` on the data configs). The statistics are computed once per
dataset and stored as `norm_stats.json` under the config's assets directory; training
copies them into `<checkpoint>/assets/<asset_id>/` so inference always uses the statistics
the model was trained with.

## Computing statistics for a dataset

```bash
uv run scripts/compute_norm_stats.py --config-name realworld_xarm_packing_pi05_subtask
```

writes `<DATA_ROOT>/datasets/realworld_xarm_packing/norm_stats/norm_stats.json`. For a critic
fine-tune, pass the fine-tune config so the statistics describe the dataset the fine-tune
trains on:

```bash
uv run scripts/compute_norm_stats.py \
    --config-name robocoin_bimanual_paligemma_cql_rlds_subtask_ar \
    --fine-tune realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar
```

Three sets of statistics are written: `state`, `actions` (the first action of each chunk)
and `action_diff` (per-timestep chunk-wise deltas relative to the current state, which is
the representation the models are trained on). Use `--dataset-name <name>:<version>` to
compute statistics for a new RLDS dataset without registering a config, and
`--output-dir` to choose where the file goes.

## Action space

Policies and critics share a 14-D bimanual end-effector layout per arm:
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` for the left arm followed by the same for
the right arm. Deltas are chunk-wise (every action in a chunk is relative to the current
state); the two gripper channels are absolute. The proprioceptive state uses the same
14-D layout. A joint-space policy is converted to this layout with
forward kinematics before the critic scores its candidates.
