# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""No-Slurm tests for nemotron4/scripts/kernel_cache/seed_from_run_script.sbatch.

The launcher derives a bounded seed run from a production run.sh by sourcing it
in a restricted shell. These tests use SEED_DRY_RUN=1 (nothing is submitted)
against a synthetic production script shaped like the derisking run.sh files.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = (
    Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
).resolve()
TOOLS = REPO_ROOT / "tools" / "persistent_cache"
LAUNCHER = REPO_ROOT / "nemotron4" / "scripts" / "kernel_cache" / "seed_from_run_script.sbatch"


def _production_tree(tmp_path: Path) -> dict:
    prod = tmp_path / "prod"
    (prod / "derisking" / "fake" / "phase1").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"sqsh")
    (tmp_path / "bf16.yaml").write_text("x: 1\n")
    (tmp_path / "source.yaml").write_text("version: 9\n")
    run_sh = prod / "derisking" / "fake" / "phase1" / "run.sh"
    run_sh.write_text(f"""#!/bin/bash
#SBATCH --nodes=400
export NVTE_FAKE_FLAG=40
export TRITON_CACHE_DIR="/tmp/triton_cache"
ROOT_DIR="{prod}"
REPO_DIR="{repo}"
IMAGE_PATH="{image}"
NAME="derisking/fake/phase1"
RUN_DIR="${{ROOT_DIR}}/${{NAME}}"
LOGS_DIR="${{RUN_DIR}}/logs"
mkdir -p ${{LOGS_DIR}}
QUANT_CONFIG="{tmp_path}/bf16.yaml"
test -f "${{IMAGE_PATH}}"
PRETOKENIZED_CONFIG="${{RUN_DIR}}/pretokenized.yaml"
awk '{{print}}' "{tmp_path}/source.yaml" > "${{PRETOKENIZED_CONFIG}}.tmp"
cat "{tmp_path}/source.yaml" >> "${{PRETOKENIZED_CONFIG}}.tmp" || exit 1
mv "${{PRETOKENIZED_CONFIG}}.tmp" "${{PRETOKENIZED_CONFIG}}"
if [ -z "${{SLURM_JOB_ID:-}}" ]; then
    echo "ERROR: submit this with sbatch" >&2
    exit 1
fi
VISION_CHECKPOINT_DIR="{tmp_path}/vision"
SEQ_LEN=8192
MAX_BATCH_SIZE=4608
if [[ -f "${{RUN_DIR}}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    LOAD_OPTIONS="--load ${{RUN_DIR}}/checkpoints"
else
    LOAD_OPTIONS="--load-vision-from ${{VISION_CHECKPOINT_DIR}}"
fi
OPTIONS=" \\
    --use-mcore-models \\
    --micro-batch-size 3 \\
    --seq-length ${{SEQ_LEN}} \\
    --num-experts 512 \\
    --fp8-recipe mxfp8 \\
    --recompute-modules residual_stream moe_act \\
    --offload-modules fused_group_mlp \\
    --moe-flex-dispatcher-backend hybridep \\
    --te-precision-config-file ${{QUANT_CONFIG}} \\
    --step-batch-size-schedule 0:1536,100:4608 \\
    --dataloader-type external \\
    --pretokenized-config ${{PRETOKENIZED_CONFIG}} \\
    --dataset-provider mistral \\
    --mimo-encoder-dp 64 \\
    --mimo-llm-offset 64 \\
    --mimo-llm-dp 24 \\
    --mimo-llm-ep 64 \\
    --tensor-parallel-num-weight-shards 64 \\
    --ddp-param-name-patterns-for-fp32-local-accumulation *output_layer* *router* \\
    ${{LOAD_OPTIONS}} \\
    --save ${{RUN_DIR}}/checkpoints \\
    --save-interval 200 \\
    --async-save \\
    --tensorboard-dir ${{RUN_DIR}}/tensorboard \\
    --tensor-metrics layer-a:500 layer-b:500 \\
    --wandb-project nemotron4 \\
    --wandb-exp-name ${{NAME}} \\
    --log-interval 10 \\
    --exit-duration-in-mins 1430 \\
        "
RUN_CMD="set -f; cd ${{REPO_DIR}}; exec python -u -m examples.mimo.pretrain_mimo ${{OPTIONS}}"
srun -l --container-image=${{IMAGE_PATH}} bash -c "${{RUN_CMD}}"
""")
    run_sh.chmod(0o755)
    return {
        "run_sh": run_sh,
        "prod_run_dir": prod / "derisking" / "fake" / "phase1",
        "repo": repo,
        "image": image,
    }


def _env(tmp_path: Path, **extra) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("SLURM_") or k.startswith("SEED_") or k.startswith("MCORE_"))
    }
    env.update(
        {
            "SEED_DRY_RUN": "1",
            "SEED_MODE": "native",
            "SEED_OUT_ROOT": str(tmp_path / "out"),
            "SEED_TOOLS": str(TOOLS),
        }
    )
    env.update(extra)
    return env


def _run(launcher: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=300
    )


def test_dry_run_derives_the_recipe_without_touching_production(tmp_path):
    tree = _production_tree(tmp_path)
    env = _env(tmp_path, RUN_SH=str(tree["run_sh"]))
    result = _run(LAUNCHER, env)
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = tmp_path / "out" / "native-dryrun"
    options = (run_dir / "derived_options.txt").read_text().split()
    joined = " ".join(options)
    # geometry, batch, run length rewritten
    assert "--mimo-encoder-dp 64 --mimo-llm-offset 64 --mimo-llm-dp 1" in joined
    assert (
        "--global-batch-size 384" in joined
    )  # mbs 3 x 64 LLM data-parallel ranks x SEED_MICROBATCHES=2
    assert "--dataset-provider mistral" in joined and "--pretokenized-config" in joined
    assert "--exit-interval 20 --log-interval 1" in joined
    assert f"--load-vision-from {tmp_path}/vision" in joined
    # production kernels' flags and literal globs survive verbatim
    assert "--fp8-recipe mxfp8" in joined and "*output_layer* *router*" in joined
    # checkpoint, schedule and sinks dropped (multi-valued flags fully)
    for gone in (
        "--save",
        "--async-save",
        "--step-batch-size-schedule",
        "--tensor-metrics",
        "layer-a:500",
        "--exit-duration-in-mins",
        "--wandb-project",
        "--wandb-exp-name",
        "derisking/fake/phase1",
    ):
        assert gone not in options, gone
    # the production script's environment travels, its side effects do not
    assert 'NVTE_FAKE_FLAG="40"' in (run_dir / "recipe_env.sh").read_text()
    assert not (tree["prod_run_dir"] / "pretokenized.yaml.tmp").exists()
    assert not (tree["prod_run_dir"] / "logs").exists()
    assert "exec python -u -m examples.mimo.pretrain_mimo" in (run_dir / "run_cmd.txt").read_text()
    assert f"SEED_CONTAINER_REPO_DIR={tree['repo']}" in (run_dir / "resolved_config.sh").read_text()


def test_recipe_without_optional_features_still_derives(tmp_path):
    tree = _production_tree(tmp_path)
    recipe = tree["run_sh"].read_text()
    for line in (
        "    --num-experts 512 \\\n",
        "    --fp8-recipe mxfp8 \\\n",
        "    --recompute-modules residual_stream moe_act \\\n",
        "    --offload-modules fused_group_mlp \\\n",
        "    --moe-flex-dispatcher-backend hybridep \\\n",
    ):
        assert line in recipe
        recipe = recipe.replace(line, "")
    tree["run_sh"].write_text(recipe)
    result = _run(LAUNCHER, _env(tmp_path, RUN_SH=str(tree["run_sh"])))
    assert result.returncode == 0, result.stdout + result.stderr
    options = (tmp_path / "out" / "native-dryrun" / "derived_options.txt").read_text().split()
    for gone in ("--fp8-recipe", "--recompute-modules", "--offload-modules", "--num-experts"):
        assert gone not in options


def test_recipe_missing_shape_flag_fails_derivation(tmp_path):
    tree = _production_tree(tmp_path)
    recipe = tree["run_sh"].read_text()
    line = "    --micro-batch-size 3 \\\n"
    assert line in recipe
    tree["run_sh"].write_text(recipe.replace(line, ""))
    result = _run(LAUNCHER, _env(tmp_path, RUN_SH=str(tree["run_sh"])))
    assert result.returncode == 12
    assert "--micro-batch-size" in result.stderr


def test_container_alias_preserves_host_checkout_and_rewrites_script_entrypoint(tmp_path):
    tree = _production_tree(tmp_path)
    visible = "/scratch/production/code"
    env = _env(
        tmp_path,
        RUN_SH=str(tree["run_sh"]),
        SEED_CONTAINER_REPO_DIR=visible,
        SEED_ENTRYPOINT=f"python -u {tree['repo']}/examples/mimo/pretrain_mimo.py",
    )
    result = _run(LAUNCHER, env)
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = tmp_path / "out" / "native-dryrun"
    command = (run_dir / "run_cmd.txt").read_text()
    assert f"{tree['repo']}:{visible}" in result.stdout
    assert f"cd {visible};" in command
    assert f"export PYTHONPATH={visible}${{PYTHONPATH:+:${{PYTHONPATH}}}};" in command
    assert f"exec python -u {visible}/examples/mimo/pretrain_mimo.py" in command
    assert str(tree["repo"]) not in command
    resolved = (run_dir / "resolved_config.sh").read_text()
    assert f"SEED_REPO_DIR={tree['repo']}" in resolved
    assert f"SEED_CONTAINER_REPO_DIR={visible}" in resolved


@pytest.mark.parametrize("visible", [None, "/scratch/production/code"])
def test_image_verification_sources_training_checkout_tools(tmp_path, visible):
    tree = _production_tree(tmp_path)
    visible = visible or str(tree["repo"])
    result = _run(
        LAUNCHER,
        _env(
            tmp_path,
            RUN_SH=str(tree["run_sh"]),
            SEED_MODE="image",
            SEED_IMAGE=str(tree["image"]),
            SEED_CONTAINER_REPO_DIR=visible,
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = (tmp_path / "out" / "image-dryrun" / "run_cmd.txt").read_text()
    assert f"source {visible}/tools/persistent_cache/enable.sh;" in command
    assert "/opt/mcore_cache/tools" not in command


@pytest.mark.parametrize(
    "visible",
    ["relative/path", "/", "//", "/scratch/a/../code", "/scratch/code;bad", "/scratch/a b"],
)
def test_container_alias_rejects_unsafe_paths(tmp_path, visible):
    tree = _production_tree(tmp_path)
    result = _run(
        LAUNCHER, _env(tmp_path, RUN_SH=str(tree["run_sh"]), SEED_CONTAINER_REPO_DIR=visible)
    )
    assert result.returncode == 2
    assert "safe absolute paths" in result.stderr


@pytest.mark.parametrize(
    "prewarm,equivalent,expected", [("1", "1", ("1", "1")), ("true", "0", ("0", "0"))]
)
def test_successful_seed_records_host_alias_and_normalized_optins(
    tmp_path, prewarm, equivalent, expected
):
    tree = _production_tree(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('''#!/bin/bash
for arg in "$@"; do
    case "$arg" in --output=*) output="${arg#--output=}";; esac
done
case "$*" in
    *--container-image=*) printf ' iteration 1/60\n iteration 60/60\n' > "$output";;
    *package_node_cache.sh*)
        for node in {0..31}; do mkdir -p "$SEED_NAMESPACE/cache_write/_node/node-$node"; done;;
esac
''')
    fake_srun.chmod(0o755)
    namespace = tmp_path / "cache"
    visible = "/scratch/production/code"
    env = _env(
        tmp_path,
        RUN_SH=str(tree["run_sh"]),
        SEED_DRY_RUN="0",
        SEED_MODE="write",
        SEED_ITERS="60",
        SEED_CONSOLIDATE="0",
        SEED_NAMESPACE=str(namespace),
        SEED_CONTAINER_REPO_DIR=visible,
        SLURM_JOB_ID="4242",
        SLURM_NNODES="32",
        MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU=prewarm,
        MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE=equivalent,
    )
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = _run(LAUNCHER, env)
    assert result.returncode == 0, result.stdout + result.stderr
    info = dict(line.split("=", 1) for line in (namespace / "SEED_INFO").read_text().splitlines())
    assert info["repo_dir"] == str(tree["repo"])
    assert info["repo_container_dir"] == visible
    assert info["inductor_prewarm_weighted_relu"] == expected[0]
    assert info["inductor_autotune_equivalent_cache"] == expected[1]


def test_spool_copy_resolves_tools_through_scontrol(tmp_path):
    """sbatch runs a spool copy of the script; SEED_TOOLS must resolve via the
    submitted path (scontrol Command=), not BASH_SOURCE."""
    tree = _production_tree(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    copied = spool / "slurm_script"
    shutil.copy(LAUNCHER, copied)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "scontrol").write_text(f"#!/bin/sh\necho '   Command={LAUNCHER} extra args'\n")
    (fake_bin / "scontrol").chmod(0o755)
    ns = tmp_path / "ns"
    (ns / "consolidated").mkdir(parents=True)
    env = _env(
        tmp_path,
        RUN_SH=str(tree["run_sh"]),
        SEED_MODE="read",
        SEED_NAMESPACE=str(ns),
        SLURM_JOB_ID="4242",
    )
    env.pop("SEED_TOOLS")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = _run(copied, env)
    assert result.returncode == 0, result.stdout + result.stderr
    run_cmd = (tmp_path / "out" / "read-4242" / "run_cmd.txt").read_text()
    assert f"source {TOOLS}/enable.sh;" in run_cmd


def test_spool_copy_without_resolvable_path_fails_with_the_override_hint(tmp_path):
    tree = _production_tree(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    copied = spool / "slurm_script"
    shutil.copy(LAUNCHER, copied)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "scontrol").write_text("#!/bin/sh\nexit 1\n")
    (fake_bin / "scontrol").chmod(0o755)
    env = _env(tmp_path, RUN_SH=str(tree["run_sh"]), SLURM_JOB_ID="4242")
    env.pop("SEED_TOOLS")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = _run(copied, env)
    assert result.returncode == 2
    assert "export SEED_TOOLS=" in result.stderr
