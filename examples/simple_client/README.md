# Simple client

A minimal client that sends random observations to a policy server and prints the
inference rate. It uses the observation format the SeeQ policies and critics consume
(three RGB cameras, a 14-D EEF state, a prompt and a task description), so it also
exercises the best-of-N path when the server was started with `--critic.*` flags.

Terminal window 1 (server, policy only or policy + critic):

```bash
uv run scripts/serve_policy.py --policy.config real_shirt_hang_pi05 --policy.dir <policy checkpoint dir>
```

Terminal window 2 (client):

```bash
uv run examples/simple_client/main.py --num-steps 20
```
