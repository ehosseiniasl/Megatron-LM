import os
import subprocess
from pathlib import Path


def eval_script(path: Path):
    basename = ".".join(str(path).split(".")[:-1])
    try:
        build = subprocess.run(
            ["ghc", "-o", basename, str(path)], capture_output=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "Timeout",
            "exit_code": -1,
            "stdout": "Compiler timeout",
            "stderr": "Compiler timeout",
        }

    if build.returncode != 0:
        return {
            "status": "SyntaxError",
            "exit_code": build.returncode,
            "stdout": build.stdout.decode("utf-8", errors="ignore") if build.stdout else "",
            "stderr": build.stderr.decode("utf-8", errors="ignore") if build.stderr else "",
        }

    try:
        output = subprocess.run([basename], capture_output=True, timeout=5)
        status = "OK" if output.returncode == 0 else "Exception"
        returncode = output.returncode
    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        output = exc
        returncode = -1
    finally:
        for ext in ["", ".hi", ".o"]:
            try:
                os.remove(basename + ext)
            except FileNotFoundError:
                pass

    return {
        "status": status,
        "exit_code": returncode,
        "stdout": output.stdout.decode("utf-8", errors="ignore") if hasattr(output, 'stdout') and output.stdout else "",
        "stderr": output.stderr.decode("utf-8", errors="ignore") if hasattr(output, 'stderr') and output.stderr else "",
    }
