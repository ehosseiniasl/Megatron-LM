import hashlib
import os
import requests
import inspect
from functools import wraps
from typing import Any, Callable

import dill

from lm_eval.utils import eval_logger


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.pickle"


def load_from_cache(file_name):
    try:
        path = f"{PATH}/{file_name}{FILE_SUFFIX}"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception:
        eval_logger.debug(f"{file_name} is not cached, generating...")
        pass


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}{FILE_SUFFIX}"

    eval_logger.debug(f"Saving {file_path} to cache...")
    with open(file_path, "wb") as file:
        file.write(dill.dumps(obj))


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)


def fetch_to_cache(key: str, url: str, headers: dict):
    try:

        file_name, file_extension = os.path.splitext(key)
        file_name = f"{file_name}.{HASH_PREFIX}{file_extension}"
        file_path = f"{PATH}/{file_name}"

        if not os.path.exists(file_path):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            content = response.content

            eval_logger.debug(f"Saving {file_path} to cache...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(content)

        return os.path.dirname(file_path)
    except Exception as e:
        eval_logger.error(f"Failed to fetch {key} from {url}: {e}")
        raise e


def cache_parquet(source_file_path: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import pandas as pd
            except ImportError:
                eval_logger.warning("pandas not available, skipping cache")
                return func(*args, **kwargs)

            args_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]

            try:
                with open(source_file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                file_hash = "unknown"

            cache_key = f"{func.__name__}_{args_hash}_{file_hash}"
            cache_path = f"{PATH}/{cache_key}.parquet"

            if os.path.exists(cache_path):
                try:
                    eval_logger.debug(f"Loading {cache_key} from cache...")
                    df = pd.read_parquet(cache_path)
                    return df.to_dict("records")
                except Exception as e:
                    eval_logger.warning(f"Failed to load cache {cache_key}: {e}")

            result = func(*args, **kwargs)

            try:
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                eval_logger.debug(f"Saving {cache_key} to cache...")
                df = pd.DataFrame(result)
                df.to_parquet(cache_path, index=False)
            except Exception as e:
                eval_logger.warning(f"Failed to save cache {cache_key}: {e}")

            return result

        return wrapper

    return decorator


def _get_file_hash(file_path: str) -> str:
    """Get MD5 hash of a file's content."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    except Exception:
        return "no_file"


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict, file_hash: str) -> str:
    """Generate cache key from function arguments and file hash."""
    key_data = {
        "func_name": func_name,
        "args": str(args),
        "kwargs": str(sorted(kwargs.items())),
        "file_hash": file_hash,
    }
    key_string = str(key_data)
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


def parquet_cache(func: Callable) -> Callable:
    """
    Decorator to cache function results using parquet format.
    Caches based on function arguments and source file hash.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        import pandas as pd

        source_file_path = inspect.getfile(func)
        file_hash = _get_file_hash(source_file_path)
        cache_key = _generate_cache_key(func.__name__, args, kwargs, file_hash)
        cache_path = f"{PATH}/{cache_key}.parquet"

        if os.path.exists(cache_path):
            try:
                eval_logger.debug(f"Loading {cache_key} from cache...")
                df = pd.read_parquet(cache_path)
                return df.to_dict("records")
            except Exception as e:
                eval_logger.warning(f"Failed to load cache {cache_key}: {e}")

        result = func(*args, **kwargs)

        try:
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            eval_logger.debug(f"Saving {cache_key} to cache...")
            df = pd.DataFrame(result)
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            eval_logger.warning(f"Failed to save cache {cache_key}: {e}")

        return result

    return wrapper
