from __future__ import annotations

import re
import unittest
from pathlib import Path


NVBENCH_ROOT = Path(__file__).resolve().parents[1]
SBATCH = NVBENCH_ROOT / "run_vlm_eval.sbatch"
JUDGE_SBATCH = NVBENCH_ROOT / "run_judge_eval.sbatch"
SERVER_ENTRYPOINT = NVBENCH_ROOT / "run_server.py"
CLIENT_ENTRYPOINT = NVBENCH_ROOT / "run_client.py"


class SbatchLaunchContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SBATCH.read_text(encoding="utf-8")
        self.active = [
            line.strip()
            for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_planner_owns_allocation_and_wrapper_owns_two_exact_steps(self) -> None:
        directives = {
            line.strip() for line in self.text.splitlines() if line.startswith("#SBATCH")
        }
        self.assertEqual(directives, {"#SBATCH --exclusive", "#SBATCH --job-name=vlmeval-fewshot"})
        self.assertEqual(len(re.findall(r"^srun ", self.text, re.MULTILINE)), 2)
        self.assertEqual(self.text.count("--exact"), 2)
        self.assertEqual(self.text.count("--overlap"), 2)
        self.assertEqual(self.text.count("--relative=0"), 1)
        self.assertIn('--cpus-per-task="${SERVER_CPUS}"', self.text)
        self.assertIn('--cpus-per-task="${CLIENT_CPUS}"', self.text)
        self.assertIn('--gpus-per-task="${GPUS_PER_NODE}"', self.text)
        self.assertEqual(self.text.count("--gpus=0"), 1)
        self.assertIn('--nodes="${NODES}" --ntasks="${NODES}"', self.text)
        self.assertIn('SERVER_ARGS=(--master-addr "${SERVER_HOST}"', self.text)

    def test_array_index_is_scheduler_owned_and_forwarded_after_user_flags(self) -> None:
        self.assertNotIn("#SBATCH --output=", self.text)
        self.assertNotIn("/users/", self.text)
        self.assertIn(
            "ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}",
            self.text,
        )
        self.assertEqual(
            self.text.count('--benchmark-index "${ARRAY_INDEX}"'),
            2,
        )
        self.assertNotIn("OCRBench", self.text)
        self.assertNotIn("DocVQA", self.text)

    def test_both_entrypoints_require_and_bind_the_same_array_index(self) -> None:
        for path in (SERVER_ENTRYPOINT, CLIENT_ENTRYPOINT):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn(
                    'parser.add_argument("--benchmark-index", type=int, required=True)',
                    text,
                )
                self.assertEqual(text.count("benchmark_index=args.benchmark_index"), 1)

    def test_each_step_requires_explicit_images_without_stale_defaults(self) -> None:
        self.assertIn(
            "SERVER_IMAGE=${NVBENCH_SERVER_IMAGE:?Export NVBENCH_SERVER_IMAGE before calling sbatch}",
            self.text,
        )
        self.assertIn(
            "CLIENT_IMAGE=${NVBENCH_CLIENT_IMAGE:?Export NVBENCH_CLIENT_IMAGE before calling sbatch}",
            self.text,
        )
        self.assertEqual(self.text.count('--container-image="${SERVER_IMAGE}"'), 1)
        self.assertEqual(self.text.count('--container-image="${CLIENT_IMAGE}"'), 1)
        self.assertNotIn("CONTAINER_IMAGE_SHA256", self.text)
        self.assertEqual(
            self.text.count(
                "CLIENT_LD_LIBRARY_PATH=${NVBENCH_CLIENT_LD_LIBRARY_PATH:-/opt/hpcx/ucx/lib}"
            ),
            1,
        )
        client_library_binding = 'LD_LIBRARY_PATH="${CLIENT_LD_LIBRARY_PATH}"'
        self.assertEqual(self.text.count(client_library_binding), 1)
        self.assertGreater(
            self.text.index(client_library_binding), self.text.index("--gpus=0")
        )
        for speculative_path in ("mpi/lib", "torch_tensorrt", "cuda/compat"):
            self.assertNotIn(speculative_path, self.text)
        self.assertNotIn("/usr/bin/python", self.text)
        self.assertIn(
            "VENDORED_VLMEVALKIT_ROOT=${SLURM_SUBMIT_DIR}/third_party/VLMEvalKit",
            self.text,
        )
        self.assertEqual(self.text.count('PYTHONPATH="${VENDORED_VLMEVALKIT_ROOT}'), 2)
        self.assertEqual(
            len(re.findall(r"^  python3 -m nvbench\.", self.text, re.MULTILINE)),
            2,
        )
        self.assertNotIn("python3 third_party/", self.text)
        self.assertEqual(self.text.count("PYTHONDONTWRITEBYTECODE=1"), 2)
        self.assertEqual(self.text.count("PYTHONUNBUFFERED=1"), 2)

    def test_server_is_backgrounded_and_wrapper_owns_last_resort_cleanup(self) -> None:
        self.assertIn(
            '--benchmark-index "${ARRAY_INDEX}" &',
            self.text,
        )
        self.assertIn("SERVER_STEP_PID=$!", self.text)
        self.assertIn("trap cleanup_server_step EXIT INT TERM", self.text)
        self.assertIn(
            '--benchmark-index "${ARRAY_INDEX}"',
            self.text,
        )
        self.assertIn('wait "${SERVER_STEP_PID}"', self.text)
        self.assertIn("trap - EXIT INT TERM", self.text)

    def test_wrapper_has_only_the_required_topology_branch(self) -> None:
        forbidden = (
            r"\bcase\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\b(?:pip|conda)\s+install\b",
            r"\bbash\s+-c\b",
        )
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, "\n".join(self.active)))
        self.assertEqual(len(re.findall(r"^if ", self.text, re.MULTILINE)), 1)
        self.assertLessEqual(len(self.active), 60)

    def test_judge_wrapper_is_one_cpu_only_step(self) -> None:
        text = JUDGE_SBATCH.read_text(encoding="utf-8")
        directives = {
            line.strip() for line in text.splitlines() if line.startswith("#SBATCH")
        }
        self.assertEqual(directives, {"#SBATCH --job-name=vlmeval-judge"})
        self.assertEqual(len(re.findall(r"^srun ", text, re.MULTILINE)), 1)
        self.assertIn("--nodes=1 --ntasks=1", text)
        self.assertNotIn("--gpus", text)
        self.assertNotIn("NVBENCH_SERVER_IMAGE", text)
        self.assertIn("python3 -m nvbench.run_judge", text)
        self.assertIn('--benchmark-index "${ARRAY_INDEX}"', text)


if __name__ == "__main__":
    unittest.main()
