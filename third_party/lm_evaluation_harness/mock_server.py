"""Mock OpenAI-compatible server for notok-completions debugging.

Supports:
  - POST /completions  (text completions with echo/logprobs)
  - GET  /encode       (tokenizer API)
  - GET  /decode       (tokenizer API)

Usage:
  python mock_server.py [--port PORT]

Then run lm_eval against it, e.g.:
  lm_eval --model notok-completions \
      --model_args base_url=http://127.0.0.1:54568,model=mock \
      --tasks niah_single_1 --batch_size 8 --limit 10 \
      --override "dataset_kwargs.max_seq_lengths=[4096]"
"""
import argparse
import json
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler


class MockHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── POST ────────────────────────────────────────────────────────
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path.rstrip("/").endswith("/completions"):
            self._handle_completions(body)
        else:
            self._send_json(404, {"error": f"Unknown POST endpoint: {self.path}"})

    def _handle_completions(self, body):
        prompt = body.get("prompt", [""])
        if isinstance(prompt, str):
            prompt = [prompt]

        model = body.get("model", "mock")
        echo = body.get("echo", False)
        logprobs_n = body.get("logprobs")
        stop = body.get("stop")

        # ── request logging ──
        prompt_summary = [f"len={len(p)}" for p in prompt]
        print(
            f"[completions] batch={len(prompt)}  echo={echo}  "
            f"logprobs={logprobs_n}  stop={stop!r}  "
            f"prompts={prompt_summary}",
            flush=True,
        )

        choices = []
        for i, p in enumerate(prompt):
            if echo:
                generated = "42"
                text = p + " " + generated
                # Build token-level logprobs spanning the full echoed text
                words = text.split()
                offsets = []
                pos = 0
                for w in words:
                    idx = text.find(w, pos)
                    offsets.append(idx)
                    pos = idx + len(w)
                choices.append({
                    "index": i,
                    "text": text,
                    "logprobs": {
                        "tokens": words,
                        "token_logprobs": [-0.01] * len(words),
                        "top_logprobs": [{w: -0.01} for w in words],
                        "text_offset": offsets,
                    },
                    "finish_reason": "stop",
                })
            else:
                text = "42"
                choices.append({
                    "index": i,
                    "text": text,
                    "logprobs": {
                        "tokens": ["42"],
                        "token_logprobs": [-0.01],
                        "top_logprobs": [{"42": -0.01}],
                        "text_offset": [0],
                    } if logprobs_n is not None else None,
                    "finish_reason": "stop",
                })

        self._send_json(200, {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {"prompt_tokens": sum(len(p) for p in prompt),
                      "completion_tokens": len(prompt) * 2,
                      "total_tokens": sum(len(p) for p in prompt) + len(prompt) * 2},
        })

    # ── GET ─────────────────────────────────────────────────────────
    def do_GET(self):
        # Read body for GET requests (tokenizer API sends JSON via GET)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        path = self.path.rstrip("/").split("/")[-1]

        if path == "encode":
            text = body.get("text", "")
            # Fake tokenizer: 1 token per 4 chars
            tokens = list(range(len(text) // 4 + 1))
            self._send_json(200, {"tokens": tokens})
        elif path == "decode":
            tokens = body.get("tokens", [])
            self._send_json(200, {"text": " ".join(str(t) for t in tokens)})
        else:
            # Health / discovery
            self._send_json(200, {"status": "ok"})

    def log_message(self, format, *args):
        pass  # Suppress default access log; we log in _handle_completions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=54568)
    args = ap.parse_args()
    print(f"Mock server on http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), MockHandler).serve_forever()
