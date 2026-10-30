"""Start and supervise the Megatron server job step for one VLM evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from nvbench.orchestration import (
    eval_run_path_resolution as preparation_module,
)
from nvbench.orchestration.megatron_server_process_management import (
    megatron_dynamic_server_launch as launch_module,
    multinode_megatron_server_step as multinode_server_step_module,
    split_megatron_server_step as server_step_module,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--coordinator-port", type=int, default=8001)
    parser.add_argument("--master-addr")
    parser.add_argument("--rendezvous-port", type=int)
    parser.add_argument("--benchmark-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--render-server-command", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render or run one array-selected Megatron server step and print its artifacts."""
    args = _parser().parse_args(argv)
    node_rank = None
    if args.master_addr is not None or args.rendezvous_port is not None:
        try:
            node_rank = int(os.environ["SLURM_NODEID"])
        except (KeyError, ValueError) as error:
            raise ValueError("multi-node launch requires an integer SLURM_NODEID") from error
    with tempfile.TemporaryDirectory(prefix="nvbench-server-", dir="/tmp") as scratch:
        # Resolve configuration, bind the array index, validate paths, and build
        # the immutable launch description before starting torchrun.
        prepared = preparation_module.prepare_megatron_dynamic_server_vlm_fewshot_run(
            args.run_config,
            args.workspace,
            benchmark_index=args.benchmark_index,
            scored_limit=args.limit,
            host=args.server_host,
            port=args.server_port,
            coordinator_port=args.coordinator_port,
            python_executable=str(Path(sys.executable).resolve(strict=True)),
            compiler_cache_directory=scratch,
            node_rank=node_rank,
            master_addr=args.master_addr,
            rendezvous_port=args.rendezvous_port,
            validate_destination=not args.render_server_command
            and node_rank in (None, 0),
        )

        # Render mode exposes the exact command without starting the server.
        # Normal mode supervises torchrun through clean terminal-state recording.
        if args.render_server_command:
            payload = {
                "mode": "render_server_command",
                "launch": launch_module.render_megatron_dynamic_server_launch(
                    prepared.launch
                ),
            }
        else:
            if prepared.resolved.configured.run.launch_mode == "local_torchrun":
                result = server_step_module.execute_split_megatron_server_step(prepared)
            else:
                result = multinode_server_step_module.execute_multinode_megatron_server_step(
                    prepared
                )
            payload = {
                "server_step_status": "clean",
                "prepared_manifest": str(result.prepared_manifest),
                "prepared_control": str(result.prepared_control_artifact),
                "terminal_control": str(result.terminal_control_artifact),
                "server_log": str(result.server_log_artifact),
                "process_returncode": result.process_returncode,
            }

    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"nvbench server step failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
