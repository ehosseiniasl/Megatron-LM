"""Run the model-free native VLMEvalKit suite preflight and write its record."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nvbench.orchestration import (
    eval_run_path_resolution as preparation_module,
    vlmeval_native_dataset_preflight as preflight_module,
)
from nvbench.orchestration.utils import write_json_once


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _existing_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{label} must be a nonsymlink regular file")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Validate every configured native dataset/prompt and write one immutable report."""
    args = _parser().parse_args(argv)
    run_config = _existing_file(args.run_config, "run config")
    output = args.output
    if not output.is_absolute() or output.parent != output.parent.resolve(strict=True):
        raise ValueError("output parent must be an existing, unambiguous absolute directory")
    resolved = preparation_module.resolve_megatron_dynamic_server_vlm_fewshot_run(
        run_config,
        args.workspace,
        validate_destination=False,
    )
    suite = resolved.configured.benchmark_suite
    if suite is None:
        raise ValueError("native dataset preflight requires a benchmark suite")
    result = preflight_module.preflight_datasets(
        suite,
        Path(resolved.paths.cache_root),
    )
    repository_root = Path(__file__).resolve().parents[3]
    relative_config = run_config.relative_to(repository_root).as_posix()
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "completed",
        "run_config": relative_config,
        "suite": {
            "name": suite.name,
            "source": suite.source_path,
        },
        "cache_root": resolved.paths.cache_root,
        "records": [asdict(record) for record in result.records],
    }
    write_json_once(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"native dataset preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
