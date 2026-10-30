# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify frozen path modules retain Ubuntu's safe archive-extraction behavior."""

import io
import os
import posixpath
import tarfile
import tempfile
from pathlib import Path


def verify_frozen_paths() -> None:
    """Exercise ALLOW_MISSING and safe extraction without disabling frozen modules or filtering."""
    if posixpath.__loader__.__name__ != "FrozenImporter":
        raise RuntimeError("posixpath is not frozen; this does not validate the rebuilt runtime")
    if not hasattr(posixpath, "ALLOW_MISSING"):
        raise RuntimeError("frozen posixpath is missing Ubuntu's ALLOW_MISSING backport")
    from pip._internal.utils.unpacking import untar_file

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        posixpath.realpath(os.path.join(directory, "missing"), strict=posixpath.ALLOW_MISSING)
        archive = root / "test.tar"
        with tarfile.open(archive, "w") as stream:
            member = tarfile.TarInfo("pkg/setup.py")
            member.size = 3
            stream.addfile(member, io.BytesIO(b"nm4"))
        with tarfile.open(archive) as stream:
            stream.extractall(root / "out", filter="data")
        if (root / "out/pkg/setup.py").read_bytes() != b"nm4":
            raise RuntimeError("normal archive extraction did not produce the expected bytes")
        untar_file(str(archive), str(root / "pip-out"))
        pip_files = list((root / "pip-out").rglob("setup.py"))
        if len(pip_files) != 1 or pip_files[0].read_bytes() != b"nm4":
            raise RuntimeError("pip archive extraction did not produce the expected bytes")
        for name, kind, link in (
            ("../escape", tarfile.REGTYPE, ""),
            ("link", tarfile.SYMTYPE, "../escape"),
            ("hardlink", tarfile.LNKTYPE, "../escape"),
        ):
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = link
            try:
                tarfile.data_filter(member, str(root / "out"))
            except (tarfile.OutsideDestinationError, tarfile.LinkOutsideDestinationError):
                continue
            raise RuntimeError(f"archive security filter accepted an escaping member: {name}")
    print("FROZEN_STDLIB_CHECK_OK", flush=True)


if __name__ == "__main__":
    verify_frozen_paths()
