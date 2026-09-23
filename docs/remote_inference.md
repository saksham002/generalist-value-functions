# Serving a policy (with or without a critic) remotely

The policy server runs the base policy, and optionally a SeeQ critic that scores N
sampled action chunks and returns the best one, on a GPU or TPU host. Robot code talks
to it over a websocket, which keeps the robot environment free of JAX dependencies.

## Starting the policy server

Policy only (behaviour cloning):

```bash
uv run scripts/serve_policy.py --policy.config realworld_xarm_packing_pi05_subtask --policy.dir <policy checkpoint dir>
```

Policy steered by a fine-tuned critic (best-of-N with N=8):

```bash
uv run scripts/serve_policy.py \
    --policy.config realworld_xarm_packing_pi05_subtask --policy.dir <policy checkpoint dir> \
    critic:critic-args \
    --critic.config robocoin_bimanual_paligemma_cql_rlds_subtask_ar \
    --critic.dir <pretrained critic checkpoint dir> \
    --critic.fine-tune-config realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar \
    --critic.num-samples 8 --critic.subtask-decode-every 4 --critic.sample-parallel
```

`--critic.dir` is the checkpoint root of the pretrained critic; with `--critic.fine-tune-config`
the fine-tuned weights are read from `<dir>/<fine-tune name>/`. On a multi-chip host pass
`--critic.fsdp-devices <n>` so the policy and critic are sharded onto the same mesh. The
server listens on port 8000 by default.

## Querying the server from robot code

Install the lightweight client in the robot environment:

```bash
cd packages/openpi-client
pip install -e .
```

Then send one observation per control step. The dictionary layout below is the one every
SeeQ policy and critic consumes: three RGB cameras, the 14-D bimanual end-effector state,
the policy prompt, and the task description the critic decodes its subtask from.

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    observation = {
        "image": {
            "base_0_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, 224, 224)),
            "left_wrist_0_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(left_img, 224, 224)),
            "right_wrist_0_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(right_img, 224, 224)),
        },
        "state": eef_state,  # (14,) unnormalized; normalization happens on the server
        "prompt": task_instruction,
        "task_description": task_instruction,
    }
    result = client.infer(observation)
    action_chunk = result["actions"]  # (action_horizon, action_dim)
    # With a critic, result["q_values"] holds the N candidate values and
    # result["predicted_subtask"] the subtask the critic decoded (on decode steps).
    ...
```

`examples/simple_client/main.py` sends random observations in this format and reports
inference latency, which is a quick way to check a served checkpoint end to end.
