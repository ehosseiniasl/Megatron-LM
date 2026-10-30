import shutil
import logging

from lm_eval.metrics.code_eval import evaluate_candidates
from lm_eval.metrics.multiple_metrics.containerized_eval import eval_string_script

logger = logging.getLogger(__name__)

# Mapping of language to compiler/interpreter command
LANGUAGE_COMPILERS = {
    "js": "node",
    "ts": "tsc",
    "cpp": "g++",
    "rs": "rustc",
    "go": "go",
    "php": "php",
    "rb": "ruby",
    "r": "Rscript",
    "sh": "bash",
    "py": "python",
    "python": "python",
    "java": "javac",
    "lua": "lua",
    "pl": "perl",
    "jl": "julia",
    "julia": "julia",
    "swift": "swift",
    "d": "dmd",
    "scala": "scala",
    "cs": "dotnet",
    "hs": "ghc",
    "haskell": "ghc",
    "clj": "clojure",
    "clojure": "clojure",
    "rkt": "racket",
    "racket": "racket",
}

# Track which languages have already warned about missing compilers
_warned_languages: set[str] = set()


def check_compiler_available(language: str) -> bool:
    """Check if the compiler/interpreter for a language is available."""
    compiler = LANGUAGE_COMPILERS.get(language)
    if compiler is None:
        # Unknown language, let the evaluator handle it
        return True
    return shutil.which(compiler) is not None


def doc_to_text(doc: dict) -> str:
    return doc["prompt"]


def doc_to_target(doc: dict) -> str:
    return doc["tests"]


def build_program(prompt: str, completion: str, tests: str) -> str:
    # Some languages (cpp, rs, sh) have tests that start with "}" to close the function
    # Strip trailing "}" from completion to avoid double braces
    if tests.lstrip().startswith("}"):
        completion = completion.rstrip()
        if completion.endswith("}"):
            completion = completion[:-1].rstrip()
    return prompt + completion + "\n" + tests


def process_resp(repeat_idx: int, resp: str, doc: dict) -> dict:
    language = doc.get("language", "py")
    program = build_program(doc["prompt"], resp, doc["tests"])

    # Check if compiler is available
    if not check_compiler_available(language):
        compiler = LANGUAGE_COMPILERS.get(language, "unknown")
        if language not in _warned_languages:
            logger.warning(
                f"Compiler for '{language}' not found ({compiler}). "
                f"Samples will be marked as CompilerNotFound. "
                f"Install the compiler and use rescore_multiple.py to re-evaluate."
            )
            _warned_languages.add(language)
        return {
            "program": program,
            "stdout": "",
            "stderr": f"Compiler '{compiler}' not found for language '{language}'",
            "exit_code": -1,
            "status": "CompilerNotFound",
            "passed": False,
            "task_id": doc["name"],
            "completion_id": repeat_idx,
        }

    result = eval_string_script(language, program)
    result["passed"] = result["status"] == "OK"
    result["task_id"] = doc["name"]
    result["completion_id"] = repeat_idx
    return result


def score_predictions(candidates: list[list[str]], docs: list[dict]) -> list[list[dict]]:
    return evaluate_candidates(candidates, docs, process_resp)
