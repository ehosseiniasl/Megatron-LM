import json

from nvbench.assemble_results import load_results


def _write_result(path, limit):
    path.write_text(
        json.dumps(
            {
                "config": {"limit": limit},
                "results": {
                    "llm_compression_2026_arxiv_interleaved": {
                        "alias": "llm_compression_2026_arxiv_interleaved",
                        "bits_per_byte": 1.23,
                    }
                },
                "n-shot": {"llm_compression_2026_arxiv_interleaved": 0},
                "versions": {"llm_compression_2026_arxiv_interleaved": 1},
            }
        )
    )


def test_load_results_accepts_configured_limit(tmp_path):
    _write_result(tmp_path / "results_configured.json", 1500)
    config = {
        "tasks": [
            {
                "task_name": "llm_compression_2026_arxiv_interleaved",
                "limit": 1500,
                "report_metric": [
                    {"alias": "llm_compression_2026_arxiv_interleaved", "metric": "bits_per_byte"}
                ],
            }
        ]
    }

    results = load_results(tmp_path, config)

    assert len(results) == 1
    assert next(iter(results.values())) == 1.23


def test_load_results_rejects_unconfigured_limit(tmp_path):
    _write_result(tmp_path / "results_canary.json", 10)
    config = {
        "tasks": [
            {
                "task_name": "llm_compression_2026_arxiv_interleaved",
                "limit": 1500,
                "report_metric": [
                    {"alias": "llm_compression_2026_arxiv_interleaved", "metric": "bits_per_byte"}
                ],
            }
        ]
    }

    assert load_results(tmp_path, config) == {}
