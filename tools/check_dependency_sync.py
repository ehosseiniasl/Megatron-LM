# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the training installer's Docker wrapper and shared base image.

The default Dockerfile invokes install.sh directly; there is no duplicated recipe.
Legacy inline Dockerfiles remain supported by the detailed comparison below.

``dependencies/install.sh`` and ``dependencies/Dockerfile.training``
install the same dependency set by two different routes. Neither is generated
from the other, so this script is what stops them from drifting apart. It runs
is not wired into CI: run it by hand after touching either file. It needs no
GPU and no container, and imports nothing outside the standard library.

Three things are compared:

1.  The pin block delimited by ``# BEGIN DEPENDENCY PINS`` and
    ``# END DEPENDENCY PINS``. The Dockerfile writes each pin as ``ARG NAME=...``
    and the script as ``NAME=...``; after dropping the ``ARG`` prefix the two
    blocks must match line for line, in order. This catches the common drift of
    bumping a version in one file and forgetting the other.

2.  The install steps, split into the sections marked by ``# ==== name ====``.
    The Dockerfile writes its steps as shell inside ``RUN bash -ex <<"EOF"``
    heredocs, so both files reduce to a stream of shell commands. Indentation,
    comments, blank lines and line continuations are normalized away, ``ENV`` is
    rewritten to ``export``, and the resulting command streams must match line
    for line within each section.

3.  The base image, which ``dependencies/install_sbatch.sh`` names a third way:
    the Dockerfile as a registry reference (``ARG FROM_IMAGE_NAME``) and the
    sbatch script as the path of an already-imported squashfs
    (``BASE_IMAGE``). The former is converted into the filename enroot would
    have written for it, and the two must name the same image.

Anything before the first section marker is per-file preamble, and is not
compared: the Dockerfile has to say ``FROM`` and the script has to say
``set -euo pipefail``. Keep dependency work inside the sections and the check
stays meaningful.

The check reasons about text, not about what the commands do. It will not
notice a ``sed`` invocation, a symlink or a patch applied in only one of the two
files, and it cannot prove the two routes produce the same environment; only
building both against the same base image and diffing ``pip freeze`` would show
that, which is far too slow to run per pull request.

Usage:

    python tools/check_dependency_sync.py
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCRIPT = REPO_ROOT / "dependencies" / "install.sh"
DEFAULT_DOCKERFILE = REPO_ROOT / "dependencies" / "Dockerfile.training"
DEFAULT_SBATCH = REPO_ROOT / "dependencies" / "install_sbatch.sh"

PIN_BLOCK_START = "# BEGIN DEPENDENCY PINS"
PIN_BLOCK_END = "# END DEPENDENCY PINS"

SECTION_MARKER = re.compile(r"^#\s*====\s*(?P<name>[\w.-]+)\s*====\s*$")

# Dockerfile lines that carry no shell of their own and drop out of the
# normalized command stream.
HEREDOC_OPEN = re.compile(r'^RUN\s+bash\s+-ex\s+<<"EOF"$')
HEREDOC_CLOSE = "EOF"

# The base image, named as a registry reference in the Dockerfile and as an
# imported squashfs path in the sbatch script.
FROM_IMAGE_NAME = re.compile(r"^ARG\s+FROM_IMAGE_NAME=(\S+)$")
BASE_IMAGE = re.compile(r'^BASE_IMAGE="(.+)"$')


def read_lines(path: Path) -> list[str]:
    """Return the file's lines with trailing whitespace removed."""
    try:
        text = path.read_text()
    except OSError as error:
        sys.exit(f"error: cannot read {path}: {error}")
    return [line.rstrip() for line in text.splitlines()]


def extract_pin_block(lines: list[str], path: Path) -> list[str]:
    """Return the pin block's lines, with any leading ``ARG`` dropped."""
    try:
        start = lines.index(PIN_BLOCK_START)
        end = lines.index(PIN_BLOCK_END)
    except ValueError:
        sys.exit(
            f"error: {path} is missing the {PIN_BLOCK_START!r} / {PIN_BLOCK_END!r} "
            f"delimiters that tools/check_dependency_sync.py compares."
        )
    if end <= start:
        sys.exit(f"error: {path} has {PIN_BLOCK_END!r} before {PIN_BLOCK_START!r}.")

    pins = []
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pins.append(re.sub(r"^ARG\s+", "", stripped))
    return pins


def normalize_commands(lines: list[str]) -> list[str]:
    """Reduce a run of file lines to a comparable stream of shell commands.

    Comments, blank lines and indentation go away, the Dockerfile's heredoc
    delimiters and ``ENV`` directives are rewritten to their shell equivalent,
    and continuation lines are folded into the command they continue.
    """
    commands: list[str] = []
    pending: str | None = None

    for line in lines:
        stripped = line.strip()

        if pending is None:
            if not stripped or stripped.startswith("#"):
                continue
            if HEREDOC_OPEN.match(stripped) or stripped == HEREDOC_CLOSE:
                continue
            # A Dockerfile ENV and a shell export state the same thing.
            stripped = re.sub(r"^ENV\s+", "export ", stripped)

        if stripped.endswith("\\"):
            fragment = stripped[:-1].strip()
            pending = fragment if pending is None else f"{pending} {fragment}"
            continue

        commands.append(stripped if pending is None else f"{pending} {stripped}")
        pending = None

    if pending is not None:
        commands.append(pending)
    return commands


def extract_sections(lines: list[str], path: Path) -> dict[str, list[str]]:
    """Return the normalized command stream of each ``# ==== name ====`` section.

    The pin block and everything ahead of the first section marker are dropped:
    the preamble is where the two files are allowed to differ.
    """
    try:
        pin_end = lines.index(PIN_BLOCK_END)
    except ValueError:
        pin_end = -1

    sections: dict[str, list[str]] = {}
    current: str | None = None
    body: list[str] = []

    for line in lines[pin_end + 1 :]:
        match = SECTION_MARKER.match(line.strip())
        if match:
            if current is not None:
                sections[current] = normalize_commands(body)
            current = match.group("name")
            if current in sections:
                sys.exit(f"error: {path} declares section {current!r} more than once.")
            body = []
            continue
        if current is not None:
            body.append(line)

    if current is not None:
        sections[current] = normalize_commands(body)

    if not sections:
        sys.exit(
            f"error: {path} declares no '# ==== name ====' sections, so there is "
            f"nothing for tools/check_dependency_sync.py to compare."
        )
    return sections


def report_diff(title: str, script_lines: list[str], dockerfile_lines: list[str]) -> bool:
    """Print a unified diff of the two sides. Return True when they differ."""
    if script_lines == dockerfile_lines:
        return False

    print(f"\n{title}")
    print("-" * len(title))
    diff = difflib.unified_diff(
        script_lines,
        dockerfile_lines,
        fromfile="install.sh",
        tofile="Dockerfile.training",
        lineterm="",
    )
    for line in diff:
        print(line)
    return True


def extract_one(pattern: re.Pattern, lines: list[str], path: Path, what: str) -> str:
    """Return the single capture group matching ``pattern``, or exit."""
    matches = [m.group(1) for m in (pattern.match(line) for line in lines) if m]
    if len(matches) != 1:
        sys.exit(f"error: expected exactly one {what} line in {path}, found {len(matches)}.")
    return matches[0]


def enroot_basename(image_reference: str) -> str:
    """Return the squashfs filename enroot produces for a container image.

    enroot drops the registry host and replaces the path and tag separators with
    ``+``, so ``nvcr.io/nvidia/pytorch:26.07-py3`` imports as
    ``nvidia+pytorch+26.07-py3.sqsh``.
    """
    host, separator, remainder = image_reference.partition("/")
    if separator and ("." in host or ":" in host):
        image_reference = remainder
    return image_reference.replace("/", "+").replace(":", "+") + ".sqsh"


def check_base_image(dockerfile_lines: list[str], args: argparse.Namespace) -> bool:
    """Check the sbatch script builds on the Dockerfile's base image.

    The two name it differently: the Dockerfile as a registry reference, the
    sbatch script as the path of an already-imported squashfs. Compare them by
    converting the former into the filename enroot would have written.
    """
    image_reference = extract_one(
        FROM_IMAGE_NAME, dockerfile_lines, args.dockerfile, "ARG FROM_IMAGE_NAME"
    )
    base_image = extract_one(BASE_IMAGE, read_lines(args.sbatch), args.sbatch, "BASE_IMAGE")

    expected = enroot_basename(image_reference)
    actual = PurePosixPath(base_image).name
    if expected == actual:
        return False

    print("\nBase images differ")
    print("-" * len("Base images differ"))
    print(f"{args.dockerfile.name} builds on {image_reference}, i.e. {expected}")
    print(f"{args.sbatch.name} builds on {actual}")
    return True


def check_installer_wrapper(lines: list[str], args: argparse.Namespace) -> int:
    """Validate the thin Docker wrapper and shared base without duplicating installer commands."""
    commands = normalize_commands(lines)
    runs = [line for line in commands if line.startswith("RUN ")]
    expected_run = r"RUN(?: --mount=\S+)* bash /opt/dependencies/install\.sh"
    valid = (
        args.script.name == "install.sh"
        and "COPY dependencies/ /opt/dependencies/" in commands
        and len(runs) == 1
        and re.fullmatch(expected_run, runs[0]) is not None
        and not check_base_image(lines, args)
    )
    if not valid:
        print("Training Dockerfile must run install.sh directly with the shared base image.")
        return 1
    print("Training Dockerfile delegates to install.sh; base image matches install_sbatch.sh.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    parser.add_argument("--sbatch", type=Path, default=DEFAULT_SBATCH)
    args = parser.parse_args()

    script_lines = read_lines(args.script)
    dockerfile_lines = read_lines(args.dockerfile)

    if any("/opt/dependencies/install.sh" in line for line in dockerfile_lines):
        return check_installer_wrapper(dockerfile_lines, args)

    failed = report_diff(
        "Dependency pins differ",
        extract_pin_block(script_lines, args.script),
        extract_pin_block(dockerfile_lines, args.dockerfile),
    )

    script_sections = extract_sections(script_lines, args.script)
    dockerfile_sections = extract_sections(dockerfile_lines, args.dockerfile)

    failed |= report_diff(
        "Install sections differ", list(script_sections), list(dockerfile_sections)
    )

    for name in script_sections:
        if name not in dockerfile_sections:
            continue
        failed |= report_diff(
            f"Section {name!r} differs", script_sections[name], dockerfile_sections[name]
        )

    failed |= check_base_image(dockerfile_lines, args)

    if failed:
        print(
            f"\nThe files under {args.script.parent} have drifted apart. "
            f"Apply the same dependency change to each of them."
        )
        return 1

    print(
        f"{args.script.name} and {args.dockerfile.name} agree on "
        f"{len(extract_pin_block(script_lines, args.script))} pins across "
        f"{len(script_sections)} sections, on the same base image as "
        f"{args.sbatch.name}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
