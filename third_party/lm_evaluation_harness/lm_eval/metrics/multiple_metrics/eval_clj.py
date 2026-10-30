import subprocess
from pathlib import Path


def eval_script(path: Path):
    try:
        output = subprocess.run(
            ["clj", "-M", str(path)], capture_output=True, timeout=30
        )
        if output.returncode == 0:
            status = "OK"
        else:
            stderr = output.stderr.decode("utf-8", errors="ignore") if output.stderr else ""
            if "CompilerException" in stderr or "Syntax error" in stderr:
                status = "SyntaxError"
            else:
                status = "Exception"
        returncode = output.returncode
    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        output = exc
        returncode = -1

    return {
        "status": status,
        "exit_code": returncode,
        "stdout": output.stdout.decode("utf-8", errors="ignore") if hasattr(output, 'stdout') and output.stdout else "",
        "stderr": output.stderr.decode("utf-8", errors="ignore") if hasattr(output, 'stderr') and output.stderr else "",
    }
