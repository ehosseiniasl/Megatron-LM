import asyncio
import hashlib
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx
import openai
import requests
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)
from tqdm import tqdm


# Silence OpenAI client logging
logging.getLogger("openai._client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
eval_logger = logging.getLogger("lm-eval")


def parse_node_list(node_list_str: str):
    nodes = []
    parts = []
    current = ""
    in_brackets = False

    for char in node_list_str:
        if char == "[":
            in_brackets = True
            current += char
        elif char == "]":
            in_brackets = False
            current += char
        elif char == "," and not in_brackets:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:
        parts.append(current)

    for part in parts:
        if "[" in part and "]" in part:
            prefix, bracket_content = part.split("[", 1)
            bracket_content = bracket_content.rstrip("]")

            for item in bracket_content.split(","):
                if "-" in item:
                    start, end = item.split("-")
                    for i in range(int(start), int(end) + 1):
                        node_id = str(i).zfill(len(start))
                        nodes.append(f"{prefix}{node_id}")
                else:
                    nodes.append(f"{prefix}{item}")
        else:
            nodes.append(part)
    return nodes


def wait_and_check_server(base_url: str) -> bool:
    @retry(
        stop=stop_after_delay(1800),  # 30 minutes
        wait=wait_exponential(min=1, max=10),  # exponential backoff up to 10s
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
        reraise=True,
    )
    def _check_server():
        try:
            # Any response means server is up, even errors
            resp = requests.post(f"{base_url}/completions", timeout=5)
            return resp.status_code == 415  # completions endpoint is supported
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            eval_logger.debug(f"Server @ {base_url}/completions not ready: {str(e)}")
            raise

    return _check_server()


def discover_endpoints_from_slurm(base_url: str) -> list[str]:
    nodelist = parse_node_list(os.environ.get("SLURM_NODELIST", ""))
    if not nodelist:
        return [base_url]

    # Parse base_url to get the port
    parsed_url = urlparse(base_url)
    host, base_port = parsed_url.hostname, parsed_url.port
    assert (
        host is not None and base_port is not None
    ), "Hostname or port not found in base_url"
    base_port = int(base_port)

    eval_logger.info("Discovering other servers ...")
    wait_and_check_server(base_url)

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(min=5, max=30),
        reraise=True,
    )
    def _discover_with_validation():
        # Test each node with ports base_port to base_port+99
        node_endpoints = {}
        for node in nodelist:
            node_endpoints[node] = []
            for port_offset in range(100):  # 0 to 99 inclusive
                test_port = base_port + port_offset
                test_url = base_url.replace(host, node).replace(
                    str(base_port), str(test_port)
                )

                try:
                    requests.get(test_url, timeout=1)
                    eval_logger.info(f"Discovered {test_url}")
                    node_endpoints[node].append(test_port)
                except (requests.exceptions.ConnectionError, Exception):
                    continue  # Try next port

        # Validate that all nodes have the same number of ports
        if not node_endpoints:
            raise Exception("No endpoints discovered on any node")

        port_counts = [len(ports) for ports in node_endpoints.values()]
        if len(set(port_counts)) > 1:
            raise Exception(
                f"Nodes have different numbers of ports: {dict(zip(nodelist, port_counts))}"
            )

        # Validate that ports are sequential
        ports = sorted(itertools.chain.from_iterable(node_endpoints.values()))
        min_port, max_port = min(ports), max(ports)
        if len(set(ports)) != len(ports):
            raise Exception(f"Ports are not unique: {ports}")

        # Validate that ports are sequential
        if ports != list(range(min_port, max_port + 1)):
            raise Exception(f"Ports are not sequential: {ports}")

        # Build discovered endpoints list
        discovered_endpoints = []
        for node in nodelist:
            for port in node_endpoints[node]:
                test_url = base_url.replace(host, node).replace(
                    str(base_port), str(port)
                )
                discovered_endpoints.append(test_url)

        return discovered_endpoints

    discovered_endpoints = _discover_with_validation()
    return discovered_endpoints or [base_url]


class LoadBalancedEndpoints:
    def __init__(self, base_urls: str, max_concurrent_per_client: int = 1):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            api_key = "DO_NOT_USE"
            eval_logger.warning(f"Using a dummy OpenAI API key: {api_key}")

        # Create httpx client with 30 minute timeout (1800 seconds)
        long_timeout_async_client = httpx.AsyncClient(timeout=1800)

        if base_urls:
            if os.environ.get("NVBENCH_DP_DISCOVER") == "1":
                self.base_urls = discover_endpoints_from_slurm(base_urls)
            elif base_urls.count("_") == 0:
                self.base_urls = [base_urls]
            else:
                self.base_urls = base_urls.split("_")
            self.clients = [
                AsyncOpenAI(
                    api_key=api_key, base_url=url, http_client=long_timeout_async_client
                )
                for url in self.base_urls
            ]
        else:
            self.base_urls = [None]
            self.clients = [
                AsyncOpenAI(api_key=api_key, http_client=long_timeout_async_client)
            ]
        eval_logger.info(
            "LoadBalancer configured: endpoints=%d, max_concurrent_per_client=%d",
            len(self.clients),
            max_concurrent_per_client,
        )
        self.semaphores = [
            asyncio.Semaphore(max_concurrent_per_client) for _ in self.clients
        ]

    @retry(
        retry=retry_if_exception_type(openai.OpenAIError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
    )
    async def _process_request(
        self,
        client: AsyncOpenAI,
        idx: int,
        inps: List[Any],
        req_kwargs: Dict,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
        cache_hook=None,
        cache_key: str = None,
        **kwargs,
    ) -> Tuple[int, Any]:
        """Process a single request with the given client."""
        async with semaphore:
            if isinstance(inps[0], dict):  # Chat completion
                response = await client.chat.completions.create(
                    messages=inps, **kwargs, **req_kwargs
                )
            else:  # Regular completion
                response = await client.completions.create(
                    prompt=inps, **kwargs, **req_kwargs
                )

            # Cache the result immediately after receiving it
            if cache_hook and cache_hook.has_cache() and cache_key:
                cache_hook.set_cache(cache_key, response)

            pbar.update(len(inps))
            return idx, response

    async def _process_batches(
        self,
        requests: List[Tuple[List[Any], Dict, str]],
        pbar: tqdm,
        cache_hook=None,
        **kwargs,
    ) -> List[Any]:
        """Process all requests in parallel, preserving original order."""
        # Create tasks for all requests
        tasks = []
        for idx, (inps, req_kwargs, cache_key) in enumerate(requests):
            client_idx = idx % len(self.clients)
            client = self.clients[client_idx]
            semaphore = self.semaphores[client_idx]
            task = self._process_request(
                client,
                idx,
                inps,
                req_kwargs,
                semaphore,
                pbar,
                cache_hook,
                cache_key,
                **kwargs,
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and sort by original index
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            processed_results.append(result)

        # Sort by original index and return just the responses
        processed_results.sort(key=lambda x: x[0])
        return [result[1] for result in processed_results]

    def _generate_batch_cache_key(
        self, batch_inputs: List[Any], batch_kwargs: Dict, **kwargs
    ) -> str:
        """Generate a cache key for a single batch."""
        # Create a deterministic hash of the batch content
        batch_content = {
            "inputs": batch_inputs,
            "batch_kwargs": batch_kwargs,
            "kwargs": kwargs,
        }
        batch_str = json.dumps(batch_content, sort_keys=True, default=str)
        return hashlib.md5(batch_str.encode()).hexdigest()

    def submit_requests(
        self,
        requests: List[Tuple[List[Any], Dict]],
        total_requests: int,
        disable_tqdm: bool = False,
        cache_hook=None,
        **kwargs,
    ) -> List[Any]:
        """Submit all requests and return responses in the original order."""
        cached_results = []
        remaining_requests = []
        cache_indices = []
        pending_requests = 0

        # Check cache for each individual batch
        for i, (batch_inputs, batch_kwargs) in enumerate(requests):
            if cache_hook and cache_hook.has_cache():
                cache_key = self._generate_batch_cache_key(
                    batch_inputs, batch_kwargs, **kwargs
                )
                cached_result = cache_hook.get_cache(cache_key)
                if cached_result is not None:
                    cached_results.append((i, cached_result))
                    continue

            pending_requests += len(batch_inputs)
            remaining_requests.append((i, batch_inputs, batch_kwargs))
            cache_indices.append(i)

        # Process only non-cached requests
        if remaining_requests:
            remaining_request_tuples = [
                (
                    inputs,
                    kwargs_dict,
                    self._generate_batch_cache_key(inputs, kwargs_dict, **kwargs),
                )
                for _, inputs, kwargs_dict in remaining_requests
            ]
            with tqdm(total=pending_requests, disable=disable_tqdm) as pbar:
                remaining_results = asyncio.run(
                    self._process_batches(
                        remaining_request_tuples, pbar, cache_hook, **kwargs
                    )
                )
        else:
            remaining_results = []

        # Combine cached and new results in original order
        all_results = [None] * len(requests)

        # Fill in cached results
        for idx, result in cached_results:
            all_results[idx] = result

        # Fill in new results
        result_idx = 0
        for orig_idx, _, _ in remaining_requests:
            all_results[orig_idx] = remaining_results[result_idx]
            result_idx += 1

        return all_results
