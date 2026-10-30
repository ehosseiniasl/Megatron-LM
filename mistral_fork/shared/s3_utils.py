import json
import logging
import os
import random
import threading
import time
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.parse import urlsplit

try:
    import boto3
    import botocore
except ImportError:
    boto3 = None  # type: ignore[assignment]
    botocore = None  # type: ignore[assignment]

logger = logging.getLogger("s3_utils")

Backend = str
DEFAULT_BACKEND = "coreweave"

_CLIENTS: dict[str, object] = {}
_CLIENTS_LOCK = threading.Lock()
_NAMESPACE_CONFIG_ENV = "MISTRAL_S3_NAMESPACE_CONFIG"
_ENDPOINT_URL_ENV = "MISTRAL_S3_ENDPOINT_URL"
_LOCAL_IMAGE_ROOT_ENV = "MISTRAL_LOCAL_IMAGE_ROOT"


@lru_cache(maxsize=1)
def _load_namespace_config() -> dict[str, object] | None:
    config_path = os.environ.get(_NAMESPACE_CONFIG_ENV)
    if config_path is None:
        return None

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError(f"Unsupported S3 namespace config version: {config.get('version')}")
    for field, expected_type in (
        ("bucket_mappings", dict),
        ("dropped_buckets", list),
        ("dropped_path_prefixes", list),
        ("passthrough_buckets", list),
    ):
        if not isinstance(config.get(field), expected_type):
            raise TypeError(f"S3 namespace config field {field!r} must be {expected_type.__name__}")
    default_bucket_prefix = config.get("default_bucket_prefix")
    if default_bucket_prefix is not None and not isinstance(default_bucket_prefix, str):
        raise TypeError("S3 namespace config field 'default_bucket_prefix' must be str")
    allowed_buckets = config.get("allowed_buckets")
    if allowed_buckets is not None and not isinstance(allowed_buckets, list):
        raise TypeError("S3 namespace config field 'allowed_buckets' must be list")
    return config


def _is_bucket_dropped(config: dict[str, object], bucket_name: str) -> bool:
    if bucket_name in set(config["dropped_buckets"]):
        return True

    allowed_buckets = config.get("allowed_buckets")
    if allowed_buckets is None:
        return False
    bucket_mappings = config["bucket_mappings"]
    passthrough_buckets = config["passthrough_buckets"]
    assert isinstance(allowed_buckets, list)
    assert isinstance(bucket_mappings, dict)
    assert isinstance(passthrough_buckets, list)
    return (
        bucket_name not in set(allowed_buckets)
        and bucket_name not in bucket_mappings
        and bucket_name not in set(passthrough_buckets)
    )


def resolve_s3_location(bucket_name: str, object_name: str) -> tuple[str, str]:
    config = _load_namespace_config()
    if config is None:
        return bucket_name, object_name

    if _is_bucket_dropped(config, bucket_name):
        raise FileNotFoundError(f"S3 bucket is configured as dropped: {bucket_name}")

    bucket_mappings = config["bucket_mappings"]
    assert isinstance(bucket_mappings, dict)
    destination = bucket_mappings.get(bucket_name)
    if destination is not None:
        parsed = urlsplit(destination)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 namespace destination for {bucket_name}: {destination}")
        prefix = parsed.path.strip("/")
        mapped_key = "/".join(part for part in (prefix, object_name.lstrip("/")) if part)
        return parsed.netloc, mapped_key

    if bucket_name in set(config["passthrough_buckets"]):
        return bucket_name, object_name

    default_bucket_prefix = config.get("default_bucket_prefix")
    if default_bucket_prefix is not None:
        parsed = urlsplit(default_bucket_prefix)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 namespace default bucket prefix: {default_bucket_prefix}")
        prefix = parsed.path.strip("/")
        mapped_key = "/".join(
            part for part in (prefix, bucket_name, object_name.lstrip("/")) if part
        )
        return parsed.netloc, mapped_key

    raise KeyError(f"S3 bucket is not present in the namespace config: {bucket_name}")


def is_object_path_dropped(path: str) -> bool:
    config = _load_namespace_config()
    if config is None:
        return False
    if path.startswith("s3://"):
        bucket_name = path[5:].split("/", 1)[0]
        return _is_bucket_dropped(config, bucket_name)
    return any(path.startswith(prefix) for prefix in config["dropped_path_prefixes"])


def validate_s3_path(path: str) -> None:
    if not path.startswith("s3://"):
        return
    bucket_name, object_name = path[5:].split("/", 1)
    resolve_s3_location(bucket_name, object_name)


def _local_object_path(bucket_name: str, object_name: str) -> Path | None:
    local_root = os.environ.get(_LOCAL_IMAGE_ROOT_ENV)
    if local_root is None:
        return None

    object_path = PurePosixPath(object_name)
    if object_path.is_absolute() or ".." in object_path.parts:
        raise ValueError(f"Invalid S3 object name for local image lookup: {object_name}")

    config = _load_namespace_config()
    if config is not None and _is_bucket_dropped(config, bucket_name):
        raise FileNotFoundError(f"S3 bucket is configured as dropped: {bucket_name}")

    return Path(local_root) / bucket_name / Path(*object_path.parts)


_MAX_READ_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 0.25
_TRANSIENT_CLIENT_ERROR_CODES = frozenset(
    {
        "InternalError",
        "InternalServerError",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
    }
)
_TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class S3ObjectReadError(OSError):
    pass


class _ShortReadError(OSError):
    pass


def _is_retryable_client_error(error: object) -> bool:
    response = error.response  # type: ignore[attr-defined]
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _TRANSIENT_CLIENT_ERROR_CODES or status in _TRANSIENT_HTTP_STATUS_CODES


def get_client(backend: Backend = DEFAULT_BACKEND) -> object:
    if boto3 is None:
        raise ImportError("boto3 is required for S3 operations")
    if backend in _CLIENTS:
        return _CLIENTS[backend]
    with _CLIENTS_LOCK:
        if backend not in _CLIENTS:
            from botocore.config import Config

            endpoint = os.environ.get(_ENDPOINT_URL_ENV)
            if endpoint is not None:
                _CLIENTS[backend] = boto3.client(
                    "s3", endpoint_url=endpoint, config=Config(max_pool_connections=16)
                )
            elif os.environ.get("COREWEAVE_ENDPOINT") is not None:
                endpoint = os.environ["COREWEAVE_ENDPOINT"]
                _CLIENTS[backend] = boto3.client(
                    "s3",
                    aws_access_key_id=os.environ["COREWEAVE_ACCESS_KEY"],
                    aws_secret_access_key=os.environ["COREWEAVE_SECRET_KEY"],
                    endpoint_url=endpoint,
                    region_name="default",
                    config=Config(s3={"addressing_style": "virtual"}),
                )
            else:
                _CLIENTS[backend] = boto3.client("s3")
    return _CLIENTS[backend]


def get_object_content_from_bucket(
    object_name: str,
    bucket_name: str,
    offset: int = 0,
    size: int = 0,
    backend: Backend = DEFAULT_BACKEND,
) -> bytes:
    local_path = _local_object_path(bucket_name, object_name)
    if local_path is not None:
        with local_path.open("rb") as file:
            file.seek(offset)
            result = file.read(size) if size > 0 else file.read()
        if size > 0 and len(result) != size:
            raise _ShortReadError(
                f"Expected {size} bytes from {local_path} at offset {offset}, "
                f"received {len(result)}"
            )
        return result

    bucket_name, object_name = resolve_s3_location(bucket_name, object_name)
    range_header = f"bytes={offset}-{offset + size - 1}" if size > 0 else ""
    kwargs: dict = {"Bucket": bucket_name, "Key": object_name}
    if range_header:
        kwargs["Range"] = range_header
    client = get_client(backend)
    from botocore import exceptions

    retryable_errors = (
        exceptions.ChecksumError,
        exceptions.ConnectionError,
        exceptions.HTTPClientError,
        exceptions.IncompleteReadError,
        OSError,
    )

    for attempt in range(1, _MAX_READ_ATTEMPTS + 1):
        try:
            response = client.get_object(**kwargs)  # type: ignore[union-attr]
            body = response["Body"]
            try:
                result = body.read()
            finally:
                body.close()
            if size > 0 and len(result) != size:
                raise _ShortReadError(
                    f"Expected {size} bytes from s3://{bucket_name}/{object_name} "
                    f"at offset {offset}, received {len(result)}"
                )
            return result
        except exceptions.ClientError as caught_error:
            error = caught_error
            retryable = _is_retryable_client_error(caught_error)
        except retryable_errors as caught_error:
            error = caught_error
            retryable = True

        if not retryable or attempt == _MAX_READ_ATTEMPTS:
            raise S3ObjectReadError(
                f"Failed to read s3://{bucket_name}/{object_name} after {attempt} attempt(s)"
            ) from error

        base_delay = _RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1)
        delay = base_delay + random.uniform(0.0, base_delay)
        logger.warning(
            "Recoverable error reading s3://%s/%s; retrying in %.2fs (%d/%d): %s",
            bucket_name,
            object_name,
            delay,
            attempt,
            _MAX_READ_ATTEMPTS - 1,
            error,
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


def check_bucket_exists(
    bucket_name: str,
    create: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    backend: Backend = DEFAULT_BACKEND,
) -> bool:
    try:
        get_client(backend).head_bucket(Bucket=bucket_name)  # type: ignore[union-attr]
        if verbose:
            print(f"Bucket exists: {bucket_name}")
        return True
    except Exception:
        if verbose:
            print(f"Bucket does not exist: {bucket_name}")
        if create:
            if verbose:
                print("Creating bucket:", bucket_name)
            get_client(backend).create_bucket(Bucket=bucket_name)  # type: ignore[union-attr]
            return True
        return False
