# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exercise the seed/bake submission pipeline without Slurm or GPUs.

Recipe derivation uses the real seed helper. Only sbatch is replaced, so these
tests cover the environment and dependencies that queued jobs will receive.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO_ROOT = (
    Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
).resolve()
TOOLS = REPO_ROOT / "tools" / "persistent_cache"
EXAMPLE = REPO_ROOT / "nemotron4" / "scripts" / "kernel_cache"
PIPELINE = EXAMPLE / "seed_and_bake.sh"
SEED_HELPER = EXAMPLE / "seed_from_run_script.sbatch"


@pytest.fixture
def pipeline(tmp_path):
    prod = tmp_path / "production"
    prod.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    image = tmp_path / "base.sqsh"
    image.write_bytes(b"test image")
    recipe = prod / "run.sh"
    recipe.write_text(f'''#!/bin/bash
#SBATCH --nodes=400
if [ -z "${{SLURM_JOB_ID:-}}" ]; then exit 1; fi
export NVTE_PIPELINE_TEST=original
unset AWS_PROFILE MISTRAL_S3_ENDPOINT_URL
REPO_DIR="{repo}"
IMAGE_PATH="{image}"
NAME=nano
VISION_CHECKPOINT_DIR="{tmp_path}/vision"
mkdir -p "{prod}/logs"
printf 'production side effect' > "{prod}/should-not-exist"
OPTIONS="--micro-batch-size 3 --seq-length 8192 --num-experts 512
--fp8-recipe mxfp8 --recompute-modules residual_stream moe_act
--offload-modules fused_group_mlp --moe-flex-dispatcher-backend hybridep
--mimo-encoder-dp 64 --mimo-llm-offset 64 --mimo-llm-dp 24
--mimo-llm-ep 64 --tensor-parallel-num-weight-shards 64
--dataset-provider mistral --save {prod}/checkpoints"
RUN_CMD="set -f; cd ${{REPO_DIR}}; exec python -u -m examples.mimo.pretrain_mimo ${{OPTIONS}}"
srun --container-image="${{IMAGE_PATH}}" bash -c "${{RUN_CMD}}"
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "submissions.jsonl"
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "record = pathlib.Path(os.environ['FAKE_SBATCH_RECORD'])\n"
        "index = len(record.read_text().splitlines()) + 1 if record.exists() else 1\n"
        "with record.open('a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
        "if index == int(os.environ.get('FAKE_SBATCH_FAIL_AT', '0')):\n"
        "    print('synthetic submission failure', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print(str(7100 + index) + ';aga')\n"
    )
    fake_sbatch.chmod(0o755)
    fake_srun = bin_dir / "srun"
    fake_srun.write_text("#!/bin/sh\necho 'unexpected srun invocation' >&2\nexit 97\n")
    fake_srun.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SEED_", "MCORE_", "SLURM_", "SBATCH_", "FAKE_SBATCH_"))
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        FAKE_SBATCH_RECORD=str(calls),
        AWS_PROFILE="inherited-profile",
        MISTRAL_S3_ENDPOINT_URL="https://inherited.invalid",
    )
    return {
        "root": tmp_path,
        "prod": prod,
        "recipe": recipe,
        "repo": repo,
        "image": image,
        "output": tmp_path / "pipeline",
        "calls": calls,
        "env": env,
    }


def _run(pipeline, *args, **env_overrides):
    env = dict(pipeline["env"], **env_overrides)
    return subprocess.run(
        [
            "bash",
            str(PIPELINE),
            "--run-script",
            str(pipeline["recipe"]),
            "--output-dir",
            str(pipeline["output"]),
            "--account",
            "test-account",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _calls(pipeline):
    path = pipeline["calls"]
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _option(call, name):
    args = call["argv"]
    for index, arg in enumerate(args):
        if arg == name:
            return args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_submits_seed_then_bake_with_one_frozen_recipe(pipeline):
    result = _run(
        pipeline,
        MCORE_CACHE_IMAGE_DIR="/tmp/another-image-cache",
        MCORE_CACHE_PREFER_IMAGE="1",
        SEED_DRY_RUN="1",
        SEED_CONSOLIDATE="0",
    )
    _assert_success(result)
    write, bake = _calls(pipeline)
    assert write["argv"][-1] == str(SEED_HELPER)
    assert bake["argv"][-1] == str(TOOLS / "bake_cache_image.sh")
    assert _option(write, "--dependency") is None
    assert _option(bake, "--dependency") == "afterok:7101"
    assert _option(bake, "--kill-on-invalid-dep") == "yes"
    for call, nodes, tasks in ((write, "32", "4"), (bake, "1", "1")):
        assert "--parsable" in call["argv"]
        assert _option(call, "--nodes") == nodes
        assert _option(call, "--ntasks-per-node") == tasks
        assert _option(call, "--gpus-per-node") == "4"
        assert _option(call, "--account") == "test-account"
        assert _option(call, "--partition") == "batch"
        assert call["env"]["SEED_DRY_RUN"] == "0"
        assert "MCORE_CACHE_IMAGE_DIR" not in call["env"]
        assert "MCORE_CACHE_PREFER_IMAGE" not in call["env"]
        assert "SLURM_JOB_ID" not in call["env"]
    frozen = pipeline["output"] / "preflight" / "write-dryrun" / "resolved_recipe.sh"
    assert frozen.is_file()
    assert write["env"]["RUN_SH"] == str(frozen)
    assert write["env"]["SEED_MODE"] == "write"
    assert write["env"]["SEED_IMAGE"] == str(pipeline["image"])
    assert write["env"]["SEED_MICROBATCHES"] == "1"
    assert write["env"]["SEED_CONSOLIDATE"] == "1"
    assert write["env"]["SEED_NAMESPACE"] == str(pipeline["output"] / "cache")
    assert bake["env"]["BASE_IMAGE"] == write["env"]["SEED_IMAGE"]
    assert bake["env"]["MCORE_CACHE_DIR"] == write["env"]["SEED_NAMESPACE"]
    assert bake["env"]["OUTPUT_IMAGE"] == str(pipeline["output"] / "cached.sqsh")
    assert bake["env"]["TOOLS_DIR"] == str(TOOLS)
    assert bake["env"]["CACHE_BUILD_REPO"] == write["env"]["SEED_REPO_DIR"]
    assert (
        bake["env"]["CACHE_CONTAINER_REPO_DIR"]
        == write["env"]["SEED_CONTAINER_REPO_DIR"]
        == str(pipeline["repo"])
    )
    recorded_jobs = (pipeline["output"] / "jobs.tsv").read_text()
    assert "write\t7101" in recorded_jobs
    assert "bake\t7102" in recorded_jobs


def test_verify_chains_all_stages_and_checks_the_baked_image(pipeline):
    result = _run(
        pipeline, "--verify", "--partition", "gpu", "--qos", "short", "--reservation", "reserved"
    )
    _assert_success(result)
    calls = _calls(pipeline)
    assert len(calls) == 4
    write, read, bake, image = calls
    assert [call["env"]["SEED_MODE"] for call in (write, read, image)] == ["write", "read", "image"]
    assert read["env"]["SEED_IMAGE"] == write["env"]["SEED_IMAGE"]
    assert image["env"]["SEED_IMAGE"] == bake["env"]["OUTPUT_IMAGE"]
    for index, call in enumerate(calls):
        assert _option(call, "--partition") == "gpu"
        assert _option(call, "--qos") == "short"
        assert _option(call, "--reservation") == "reserved"
        assert call["env"]["SEED_DRY_RUN"] == "0"
        assert "SLURM_JOB_ID" not in call["env"]
        if index:
            assert _option(call, "--dependency") == f"afterok:{7100 + index}"
            assert _option(call, "--kill-on-invalid-dep") == "yes"
    assert {call["env"]["RUN_SH"] for call in (write, read, image)} == {write["env"]["RUN_SH"]}


def test_container_alias_is_frozen_and_forwarded_to_every_stage(pipeline):
    visible = "/scratch/production/code"
    _assert_success(_run(pipeline, "--verify", SEED_CONTAINER_REPO_DIR=visible))
    write, read, bake, image = _calls(pipeline)
    for call in (write, read, image):
        assert call["env"]["SEED_REPO_DIR"] == str(pipeline["repo"])
        assert call["env"]["SEED_CONTAINER_REPO_DIR"] == visible
    assert bake["env"]["CACHE_BUILD_REPO"] == str(pipeline["repo"])
    assert bake["env"]["CACHE_CONTAINER_REPO_DIR"] == visible
    env = dict(
        write["env"], SEED_DRY_RUN="1", SEED_OUT_ROOT=str(pipeline["root"] / "queued-preflight")
    )
    _assert_success(
        subprocess.run(
            ["bash", str(SEED_HELPER)], env=env, capture_output=True, text=True, timeout=60
        )
    )
    command = (pipeline["root"] / "queued-preflight/write-dryrun/run_cmd.txt").read_text()
    assert f"cd {visible};" in command


def test_dry_run_derives_real_options_without_submitting_or_touching_production(pipeline):
    result = _run(pipeline, "--dry-run", "--verify")
    _assert_success(result)
    assert _calls(pipeline) == []
    assert not pipeline["output"].exists()
    previews = list(pipeline["root"].glob("pipeline.preview.*"))
    assert len(previews) == 1
    preflight = previews[0] / "preflight" / "write-dryrun"
    options = " ".join((preflight / "derived_options.txt").read_text().split())
    assert "--global-batch-size 192" in options
    assert "--exit-interval 20" in options
    assert "--dataset-provider mistral" in options
    assert "--save " not in options
    assert (preflight / "resolved_recipe.sh").is_file()
    assert not (pipeline["prod"] / "logs").exists()
    assert not (pipeline["prod"] / "should-not-exist").exists()


def test_explicit_seed_image_is_also_used_as_bake_base(pipeline):
    image = pipeline["root"] / "override.sqsh"
    image.write_bytes(b"override image")
    result = _run(pipeline, SEED_IMAGE=str(image), SEED_MICROBATCHES="2", SEED_ITERS="30")
    _assert_success(result)
    write, bake = _calls(pipeline)
    assert write["env"]["SEED_IMAGE"] == bake["env"]["BASE_IMAGE"] == str(image)
    assert write["env"]["SEED_MICROBATCHES"] == "2"
    assert write["env"]["SEED_ITERS"] == "30"


def test_seed_geometry_override_changes_all_training_allocations(pipeline):
    result = _run(pipeline, "--verify", SEED_LLM_DP="2")
    _assert_success(result)
    write, read, bake, image = _calls(pipeline)
    for call in (write, read, image):
        assert _option(call, "--nodes") == "48"
        assert _option(call, "--ntasks") == "192"
        assert call["env"]["SEED_LLM_DP"] == "2"
    assert _option(bake, "--nodes") == "1"
    preflight = pipeline["output"] / "preflight" / "write-dryrun"
    options = " ".join((preflight / "derived_options.txt").read_text().split())
    assert "--global-batch-size 384" in options


def test_geometry_options_override_environment_for_every_stage(pipeline):
    result = _run(
        pipeline,
        "--verify",
        "--llm-dp",
        "4",
        "--encoder-dp",
        "32",
        "--gpus-per-node",
        "4",
        SEED_LLM_DP="2",
    )
    _assert_success(result)
    write, read, bake, image = _calls(pipeline)
    for call in (write, read, image):
        assert _option(call, "--nodes") == "72"
        assert _option(call, "--ntasks") == "288"
        assert call["env"]["SEED_LLM_DP"] == "4"
        assert call["env"]["SEED_ENCODER_DP"] == "32"
    assert _option(bake, "--nodes") == "1"
    preflight = pipeline["output"] / "preflight" / "write-dryrun"
    options = " ".join((preflight / "derived_options.txt").read_text().split())
    assert "--mimo-encoder-dp 32 --mimo-llm-offset 32 --mimo-llm-dp 4" in options
    assert "--global-batch-size 768" in options


@pytest.mark.parametrize(
    "flag,value", [("--llm-dp", "0"), ("--encoder-dp", "x"), ("--gpus-per-node", "-4")]
)
def test_invalid_geometry_option_fails_before_any_submission(pipeline, flag, value):
    result = _run(pipeline, flag, value)
    assert result.returncode != 0
    assert _calls(pipeline) == []
    assert not pipeline["output"].exists()


def test_invalid_seed_geometry_fails_before_any_submission(pipeline):
    result = _run(pipeline, SEED_ENCODER_DP="65")
    assert result.returncode != 0
    assert _calls(pipeline) == []
    assert "geometry" in result.stdout + result.stderr


def test_frozen_recipe_survives_source_changes_and_preserves_environment_unsets(pipeline):
    _assert_success(_run(pipeline))
    write, _ = _calls(pipeline)
    pipeline["recipe"].write_text('echo "recipe changed" >&2\nexit 37\n')
    env = dict(
        write["env"], SEED_DRY_RUN="1", SEED_OUT_ROOT=str(pipeline["root"] / "batch-preflight")
    )
    result = subprocess.run(
        ["bash", str(SEED_HELPER)], env=env, capture_output=True, text=True, timeout=60
    )
    _assert_success(result)
    recipe_env = pipeline["root"] / "batch-preflight" / "write-dryrun" / "recipe_env.sh"
    probe = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; [[ ! ${AWS_PROFILE+x} && ! ${MISTRAL_S3_ENDPOINT_URL+x} && "$NVTE_PIPELINE_TEST" == original ]]',
            "_",
            str(recipe_env),
        ],
        env=pipeline["env"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    _assert_success(probe)


def test_invalid_recipe_fails_before_any_submission(pipeline):
    pipeline["recipe"].write_text('IMAGE_PATH=missing\nOPTIONS=""\n')
    result = _run(pipeline)
    assert result.returncode != 0
    assert _calls(pipeline) == []


def test_submission_failure_stops_later_stages_and_retains_prior_job_id(pipeline):
    result = _run(pipeline, "--verify", FAKE_SBATCH_FAIL_AT="2")
    assert result.returncode != 0
    assert len(_calls(pipeline)) == 2
    recorded_jobs = (pipeline["output"] / "jobs.tsv").read_text()
    assert "write\t7101" in recorded_jobs
    assert "7102" not in recorded_jobs
    assert "synthetic submission failure" in result.stdout + result.stderr


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_output_is_rejected_without_overwriting_or_submitting(pipeline, nonempty):
    pipeline["output"].mkdir()
    marker = pipeline["output"] / "preserve-me"
    if nonempty:
        marker.write_text("existing run")
    result = _run(pipeline)
    assert result.returncode != 0
    assert _calls(pipeline) == []
    if nonempty:
        assert marker.read_text() == "existing run"
        assert list(pipeline["output"].iterdir()) == [marker]
    else:
        assert list(pipeline["output"].iterdir()) == []


def test_output_image_override_is_baked_and_verified(pipeline):
    images = pipeline["root"] / "images"
    images.mkdir()
    target = images / "base+supercache.sqsh"
    result = _run(pipeline, "--verify", "--output-image", str(target))
    _assert_success(result)
    write, read, bake, image = _calls(pipeline)
    assert bake["env"]["OUTPUT_IMAGE"] == str(target)
    assert image["env"]["SEED_IMAGE"] == str(target)
    assert bake["env"]["MCORE_CACHE_DIR"] == str(pipeline["output"] / "cache")
    assert not (pipeline["output"] / "cached.sqsh").exists()


@pytest.mark.parametrize("name", ["existing.sqsh", "not-an-image.tar", "missing-dir/new.sqsh"])
def test_invalid_output_image_fails_before_any_submission(pipeline, name):
    images = pipeline["root"] / "images"
    images.mkdir()
    (images / "existing.sqsh").write_text("in use")
    result = _run(pipeline, "--output-image", str(images / name))
    assert result.returncode != 0
    assert _calls(pipeline) == []
    assert not pipeline["output"].exists()
    assert (images / "existing.sqsh").read_text() == "in use"
