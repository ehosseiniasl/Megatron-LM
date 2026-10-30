import random
import requests
import threading
import os
import concurrent.futures

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.utils import eval_logger
from lm_eval.api.registry import register_model

eval_logger = eval_logger


@register_model("megatron_eval_api")
class MegatronEvalAPIModel(LM):
    def __init__(
        self, endpoints: str, name: str, batch_size: int = 4, add_BOS: bool = False
    ) -> None:
        self.endpoints = [ep.strip() for ep in endpoints.split("|")]
        self.name = name
        self.batch_size = int(batch_size)
        self.add_BOS = add_BOS
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs = dict([arg.split("=") for arg in arg_string.split(",")])
        eval_logger.info(f"Creating model {cls.__name__} with kwargs {kwargs}")
        return cls(**kwargs)
    
    def _make_single_request(self, url, data, request_id):
        response = requests.post(url, json=data, timeout=1000)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        response_body = response.json()
        response_body["args"] = data["args"]
        response_body["request_id"] = request_id
        return response_body
    
    def _make_concurrent_requests(self, args, uri, process_response, disable_tqdm: bool = False):
        response_results = []

        if os.path.exists("exception.txt"):
            os.remove("exception.txt")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.endpoints)
        ) as executor:
            futures = []
            request_id = 0
            try:
                for i in range(0, len(args), self.batch_size):
                    batch_args = args[i : i + self.batch_size]
                    endpoint = random.choice(self.endpoints)
                    url = f"http://{endpoint}/{uri}"
                    data = {"args": batch_args, "add_BOS": self.add_BOS}
                    futures.append(executor.submit(self._make_single_request, url, data, request_id))
                    request_id += 1

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    disable=disable_tqdm,
                ):
                    request_id, result = process_response(future.result())
                    response_results.append((request_id, result))
            except Exception as e:
                with open("exception.txt", "w") as f:
                    f.write('exception happens when making requests to servers.\n')
                    f.write(f"{request_id}/{(len(args)+self.batch_size-1)//self.batch_size} done.\n")
                    f.write(str(e))
                raise e
        
        sorted_response_results = sorted(response_results, key=lambda x: x[0])

        res = []
        for _, v in sorted_response_results:
            res.extend(v)

        return res

    def _process_loglikelihood_response(self, response):
        res = []
        request_id = response["request_id"]
        for req, ll, is_greedy in zip(
            response["args"], response["ll"], response["is_greedy"]):
            self.cache_hook.add_partial("loglikelihood", req, (ll, is_greedy))
            res.append((ll, is_greedy))
        return request_id, res

    def loglikelihood(self, reqs, disable_tqdm: bool = False):
        args = [req.args for req in reqs]
        return self._make_concurrent_requests(
            args, "likelihood", self._process_loglikelihood_response, disable_tqdm)

    def _process_generate_until_response(self, response):
        request_id = response["request_id"]
        for req, resp in zip(response['args'], response["text"]):
            self.cache_hook.add_partial("generate_until", req, resp)

        eval_logger.debug("Generation response: %s", response)
        return request_id, response["text"]

    def generate_until(self, reqs, disable_tqdm: bool = False):
        args = [req.args for req in reqs]
        return self._make_concurrent_requests(
            args, "generate_until", self._process_generate_until_response, disable_tqdm)
        
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling is not implemented for MegatronEvalAPIModel"
        )
