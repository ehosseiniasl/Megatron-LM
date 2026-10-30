import contextlib
import cProfile
import json
import logging
import time
from pathlib import Path
from pstats import SortKey
from typing import Iterator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def log_profile_if_slow(
    name: str,
    threshold: float,
    add_log: dict[str, str | int | float],
    output_path: Path | None = None,
) -> Iterator[None]:
    start = time.time()
    try:
        with cProfile.Profile() as pr:
            yield
    finally:
        duration = time.time() - start
        logs = dict(add_log)
        logs["duration"] = duration
        if duration > threshold:
            logger.warning(
                f"{name}:{json.dumps(logs, ensure_ascii=False, indent=None)}"
            )
            msg = " - ".join([f"{k}:{v}" for k, v in logs.items()])
            logger.info(f"================ START PROFILE {name} {msg} ==============")
            pr.print_stats(sort=SortKey.CUMULATIVE)
            logger.info(f"================ END PROFILE {name} {msg} ==============")

            if output_path:
                assert not output_path.exists(), output_path
                logger.info(f"Dumping profile to {output_path}")
                output_path.parent.mkdir(exist_ok=True, parents=True)
                pr.dump_stats(str(output_path))
