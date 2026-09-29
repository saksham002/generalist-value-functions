import dataclasses
import logging
import os
import socket
import time

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "real_shirt_hang_pi05").
    config: str
    # Checkpoint directory (e.g., "checkpoints/real_shirt_hang_pi05/exp").
    dir: str
    # Optional explicit checkpoint step. Only consulted by the BestOfN
    # critic-aware loading path (when --critic.* is set); the default policy
    # loader points `dir` at a step subdirectory directly.
    step: int | None = None
    # Optional FineTuneConfig name. Only consulted by the BestOfN loading path.
    fine_tune_config: str | None = None


@dataclasses.dataclass
class CriticArgs:
    """Optional critic / value-function args for BestOfN action selection.

    When set, scripts/serve_policy.py loads a critic checkpoint alongside the
    policy and wraps both in BestOfNPolicy. When unset, behavior is byte-
    identical to the policy-only serving path.
    """

    # Critic train config name (e.g., "robocoin_bimanual_paligemma_q_sarsa").
    config: str
    # Critic checkpoint directory.
    dir: str
    # Optional explicit critic step. None → use the latest step subdir.
    step: int | None = None
    # Optional FineTuneConfig name (e.g., "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final").
    fine_tune_config: str | None = None
    # Number of policy samples to draw per inference call (1 ≤ N ≤ ~16).
    num_samples: int = 8
    # If True, switch BestOfNPolicy to sample-parallel mode: build a
    # (num_samples, device_count // num_samples) mesh so each candidate runs on
    # its own batch position with a small per-sample FSDP group, instead of
    # sampling device_count candidates and discarding the surplus. Requires
    # device_count divisible by num_samples.
    sample_parallel: bool = False
    # Override the inference mesh's fsdp axis size when sample_parallel=True.
    # None → auto-pick (device_count // num_samples). Use this when the
    # auto-picked value would make per-chip param storage too large
    # (e.g. on v5e-32 keep fsdp=16 to match the saved sharding).
    fsdp_devices: int | None = None
    # For a predict_subtask_ar critic only: re-decode the current subtask once
    # every N infer calls and reuse the cached subtask string in between (the
    # subtask is a slow-changing phase label). Ignored for non-subtask_ar critics.
    subtask_decode_every: int = 20


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Specifies how to load the policy.
    policy: Checkpoint

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Override obs["prompt"] for the policy only, leaving the critic on the client-sent
    # prompt — required when the policy was trained with prompt_mode="task_description"
    # but the critic was trained with prompt_mode="subtask". BestOfN path only.
    task_description: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


    # Optional critic config for BestOfN action selection. Default behavior
    # (no critic) is unchanged. Only valid in combination with `policy:checkpoint`.
    critic: CriticArgs | None = None

    # Force routing through BestOfNPolicy even when no critic args are given.
    # Required for bimanual-EEF policies whose 32-D model action is padded down to
    # 14-D: the BestOfNPolicy infer path applies the action-dim slice before
    # Unnormalize, which the default Policy.infer path does not. Implied (no need
    # to set) when --critic.* is set.
    use_bestofn_loader: bool = False
    # Build a (num_samples, device_count // num_samples) mesh for the BC +
    # use_bestofn_loader path (the critic path has its own --critic.sample-parallel).
    # Default True — matches the critic path's typical setting.
    sample_parallel: bool = True
    # Override the inference mesh's fsdp axis size for the BC + use_bestofn_loader
    # path (the critic path has its own --critic.fsdp-devices). Match the value
    # the policy was trained with when the saved sharding doesn't load on the
    # auto-picked mesh (e.g. fsdp=16 for a v5e-32 / target-composite policy).
    fsdp_devices: int | None = None

    # Number of flow-matching integration (Euler) steps for the policy's
    # sample_actions. None → use the model default (10). Forwarded into the
    # policy's sample_kwargs; only affects the policy-only (BC) sampling path.
    num_steps: int | None = None

    # Sampling-counter continuity across a server restart. The flow-matching x_T for
    # inference call n is fold_in(rng, num_processes * n), so a restarted server would
    # otherwise replay a resumed eval with the noise of the calls it already made.
    # None → read the counter the previous process persisted beside the checkpoint
    # (0 when absent); an explicit value overrides it.
    start_inference_iter: int | None = None
    # Where that counter lives. None → beside the critic checkpoint root, or the policy
    # checkpoint dir when serving without a critic.
    inference_iter_path: str | None = None


def create_policy(args: Args) -> _policy.BasePolicy:
    """Create a policy from the given arguments."""
    if args.critic is not None or args.use_bestofn_loader:
        # Lazy import: BestOfNPolicy pulls in JAX value-function modules that
        # the policy-only default path doesn't need.
        from openpi.policies.best_of_n_policy import create_bestofn_policy
        from openpi.policies.best_of_n_policy import inference_iter_uri

        if args.critic is not None:
            return create_bestofn_policy(
                policy_config_name = args.policy.config,
                policy_checkpoint_dir = args.policy.dir,
                policy_step = args.policy.step,
                policy_fine_tune_config = args.policy.fine_tune_config,
                policy_task_description = args.task_description,
                critic_config_name = args.critic.config,
                critic_checkpoint_dir = args.critic.dir,
                critic_step = args.critic.step,
                critic_fine_tune_config = args.critic.fine_tune_config,
                num_samples = args.critic.num_samples,
                default_prompt = args.default_prompt,
                sample_parallel = args.critic.sample_parallel,
                fsdp_devices = args.critic.fsdp_devices,
                subtask_decode_every = args.critic.subtask_decode_every,
                num_steps = args.num_steps,
                start_inference_iter = args.start_inference_iter,
                inference_iter_path = (
                    args.inference_iter_path
                    or inference_iter_uri(args.critic.dir)
                ),
            )
        # use_bestofn_loader=True without critic: load policy through
        # BestOfNPolicy (which applies the bimanual-EEF action-dim slice)
        # but skip the critic + BestOfN sampling — infer() falls back to the
        # plain policy sampling path internally.
        return create_bestofn_policy(
            policy_config_name = args.policy.config,
            policy_checkpoint_dir = args.policy.dir,
            policy_step = args.policy.step,
            policy_fine_tune_config = args.policy.fine_tune_config,
            policy_task_description = args.task_description,
            default_prompt = args.default_prompt,
            sample_parallel = args.sample_parallel,
            fsdp_devices = args.fsdp_devices,
            num_steps = args.num_steps,
        )

    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps} if args.num_steps is not None else None,
    )


def main(args: Args) -> None:
    # Multi-host JAX init for TPU pods. Must run before any other JAX call;
    # load_policy reads jax.local_device_count() and would otherwise hang on
    # multi-host pods waiting for peers. The mapping from JAX process_index
    # to TPU worker IP is deterministic per pod — log it on every worker so
    # you can read off which IP corresponds to rank 0 and point the robot
    # client at it.
    if os.environ.get("PLATFORM", "gpu") == "tpu":
        import jax

        jax.distributed.initialize()
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(
            "JAX distributed: process_index=%d/%d, devices=%d (local=%d), host=%s, ip=%s",
            jax.process_index(), jax.process_count(),
            jax.device_count(), jax.local_device_count(),
            hostname, local_ip,
        )

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Multi-host serving split. JAX rank 0 binds the websocket; every other
    # rank enters BestOfNPolicy.participate_loop so the JIT'd inference
    # function (compiled across all hosts) doesn't deadlock on rank 0.
    import jax

    if jax.process_count() > 1 and jax.process_index() != 0:
        from openpi.policies.best_of_n_policy import BestOfNPolicy

        if isinstance(policy, BestOfNPolicy):
            logging.info(
                "JAX rank %d/%d: entering inference participation loop "
                "(rank 0 binds the websocket).",
                jax.process_index(), jax.process_count(),
            )
            policy.participate_loop()
        else:
            logging.warning(
                "JAX rank %d/%d: standard Policy doesn't support multi-host "
                "inference; sleeping. Rank 0 will hang waiting for collectives "
                "unless you launch with --use-bestofn-loader.",
                jax.process_index(), jax.process_count(),
            )
            while True:
                time.sleep(3600)
        return

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
