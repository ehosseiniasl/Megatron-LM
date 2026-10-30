"""Validate a Megatron PretokenizedConfig YAML before training."""

import argparse
from pathlib import Path

from mistral_adapter.pretokenized_config_utils import load_pretokenized_config
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    get_datasource_metadata_v2_path,
)

MAX_ERRORS_TO_PRINT = 20


def _metadata_path_for_prefix(prefix_path: str) -> Path:
    prefix_paths = PrefixPaths.from_prefix(prefix_path)
    merged_chunks_folder = prefix_paths.prefix.parent
    source_folder = merged_chunks_folder.parent
    metadata_path = get_datasource_metadata_v2_path(merged_chunks_folder)
    if metadata_path.exists():
        return metadata_path
    return get_datasource_metadata_v2_path(source_folder)


def validate_pretokenized_config(config_path: Path) -> int:
    config = load_pretokenized_config(config_path)
    errors: list[str] = []
    for source in config.sources:
        metadata_path = _metadata_path_for_prefix(source.prefix_path)
        if not metadata_path.exists():
            prefix_paths = PrefixPaths.from_prefix(source.prefix_path)
            merged_chunks_folder = prefix_paths.prefix.parent
            source_folder = merged_chunks_folder.parent
            errors.append(
                f"{source.prefix_path}: missing pretokenization_metadata_v2.json "
                f"in {merged_chunks_folder} or {source_folder}"
            )
            continue
        try:
            PretokenizedDatasourceMetadataV2.model_validate_json(
                metadata_path.read_text()
            )
        except Exception as exc:
            errors.append(f"{source.prefix_path}: {type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError(
            "Megatron cannot reload pretokenized source metadata. "
            "Fix the embedded mistral_fork metadata schema or the selected data before launching.\n"
            + "\n".join(errors[:MAX_ERRORS_TO_PRINT])
            + (
                f"\n... and {len(errors) - MAX_ERRORS_TO_PRINT} more error(s)"
                if len(errors) > MAX_ERRORS_TO_PRINT
                else ""
            )
        )
    return len(config.sources)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Megatron PretokenizedConfig YAML and its source metadata."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source_count = validate_pretokenized_config(args.config)
    print(
        f"Validated {source_count} pretokenized source metadata file(s) with Megatron."
    )


if __name__ == "__main__":
    main()
