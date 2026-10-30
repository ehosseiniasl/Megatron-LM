# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run the bake orchestration with simulated container execution, never a GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_root_override = os.environ.get('MCORE_REPO_ROOT')
REPO_ROOT = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
BAKE = REPO_ROOT / 'tools/persistent_cache/bake_cache_image.sh'


@pytest.fixture
def bake(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    tools_dir = tmp_path / 'tools'
    tools_dir.mkdir()
    (tools_dir / 'install_image_cache.py').touch()
    (tools_dir / 'prepare_nccl_topology_template.py').touch()
    repo = tmp_path / 'checkout'
    repo.mkdir()
    namespace = tmp_path / 'cache'
    (namespace / 'consolidated').mkdir(parents=True)
    base = tmp_path / 'base.sqsh'
    base.write_bytes(b'original')
    record = tmp_path / 'record.json'
    # The actual nested bash command runs. Only the installer and srun are
    # replaced; we observe the final installer arguments and mount request.
    (bin_dir / 'python3').write_text(
        f'#!{sys.executable}\n'
        'import json,os,pathlib,sys\n'
        'pathlib.Path(os.environ["INSTALL_RECORD"]).write_text(json.dumps(sys.argv[1:]))\n'
        'dest=pathlib.Path(sys.argv[sys.argv.index("--dest")+1])\n'
        'dest.mkdir(parents=True)\n'
        '(dest/"MANIFEST").write_text("test=1\\n")\n'
    )
    (bin_dir / 'srun').write_text(
        f'#!{sys.executable}\n'
        'import json,os,pathlib,subprocess,sys\n'
        'args=sys.argv[1:]\n'
        'pathlib.Path(os.environ["SRUN_RECORD"]).write_text(json.dumps(args))\n'
        'rc=subprocess.run(args[args.index("bash"):]).returncode\n'
        'output=next(a.split("=",1)[1] for a in args if a.startswith("--container-save="))\n'
        'pathlib.Path(output).write_bytes(b"derived")\n'
        'sys.exit(rc)\n'
    )
    for name in ('python3', 'srun'):
        (bin_dir / name).chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('SLURM_', 'MCORE_', 'CACHE_', 'SBATCH_'))
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        BASE_IMAGE=str(base),
        MCORE_CACHE_DIR=str(namespace),
        OUTPUT_IMAGE=str(tmp_path / 'derived.sqsh'),
        TOOLS_DIR=str(tools_dir),
        CACHE_BUILD_REPO=str(repo),
        INSTALL_DEST=str(tmp_path / 'installed'),
        INSTALL_RECORD=str(record),
        SRUN_RECORD=str(tmp_path / 'srun.json'),
    )
    return tmp_path, env


def run_bake(bake, metadata='', **overrides):
    root, env = bake
    (root / 'cache/SEED_INFO').write_text(metadata)
    return subprocess.run(
        ['bash', str(BAKE)], env=dict(env, **overrides), capture_output=True, text=True, timeout=30
    )


def test_bake_preserves_visible_checkout_and_seeded_image_defaults(bake):
    root, env = bake
    target = '/production/nemotron4/code'
    result = run_bake(
        bake,
        'inductor_prewarm_weighted_relu=1\ninductor_autotune_equivalent_cache=1\n',
        CACHE_CONTAINER_REPO_DIR=target,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    installer = json.loads((root / 'record.json').read_text())
    assert installer[installer.index('--training-repo') + 1] == target
    assert '--prewarm-weighted-relu' in installer
    assert '--equivalent-autotune-cache' in installer
    srun = json.loads((root / 'srun.json').read_text())
    mounts = next(a for a in srun if a.startswith('--container-mounts='))
    assert f",{env['CACHE_BUILD_REPO']}:{target}" in mounts
    assert (root / 'base.sqsh').read_bytes() == b'original'
    assert (root / 'derived.sqsh').read_bytes() == b'derived'
    assert not list(root.glob('*.partial-*.sqsh'))


def test_old_seed_stays_default_off_despite_inherited_feature_flags(bake):
    root, env = bake
    result = run_bake(
        bake, MCORE_INDUCTOR_PREWARM_WEIGHTED_RELU='1', MCORE_INDUCTOR_AUTOTUNE_EQUIVALENT_CACHE='1'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    installer = json.loads((root / 'record.json').read_text())
    assert installer[installer.index('--training-repo') + 1] == env['CACHE_BUILD_REPO']
    assert '--prewarm-weighted-relu' not in installer
    assert '--equivalent-autotune-cache' not in installer
    assert '--nccl-topo-file' not in installer
    assert '--nccl-topo-namespace' not in installer


@pytest.mark.parametrize(
    'value', ['', 'yes', '1\ninductor_prewarm_weighted_relu=1', '$(touch malicious)']
)
def test_invalid_seed_settings_fail_before_container_start(bake, value):
    root, _env = bake
    result = run_bake(bake, f'inductor_prewarm_weighted_relu={value}\n')
    assert result.returncode != 0
    assert not (root / 'srun.json').exists()
    assert not (root / 'derived.sqsh').exists()


@pytest.mark.parametrize(
    'setting', ['inductor_prewarm_weighted_relu', 'inductor_autotune_equivalent_cache']
)
@pytest.mark.parametrize('values', [('1', ''), ('0', ''), ('', '1'), ('', '')])
def test_duplicate_empty_seed_settings_fail_before_container_start(bake, setting, values):
    root, _env = bake
    metadata = ''.join(f'{setting}={value}\n' for value in values)
    result = run_bake(bake, metadata)
    assert result.returncode != 0
    assert not (root / 'srun.json').exists()
    assert not (root / 'derived.sqsh').exists()


def test_unsafe_checkout_alias_fails_before_container_start(bake):
    root, _env = bake
    result = run_bake(bake, CACHE_CONTAINER_REPO_DIR='/production/code,other:/mount')
    assert result.returncode != 0
    assert not (root / 'srun.json').exists()


def test_bake_passes_optional_topology_separately_from_kernel_namespace(bake):
    root, _env = bake
    topology = root / 'previous-cmh-topo.xml'
    topology.write_text('<system version="1"/>')
    result = run_bake(
        bake, CACHE_NCCL_TOPO_FILE=str(topology), CACHE_NCCL_TOPO_NAMESPACE='cmh-nvl72.v1'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    installer = json.loads((root / 'record.json').read_text())
    assert installer[installer.index('--nccl-topo-file') + 1] == str(topology)
    assert installer[installer.index('--nccl-topo-namespace') + 1] == 'cmh-nvl72.v1'
    assert installer[installer.index('--namespace') + 1] == str(root / 'cache')
    assert '--skip-pci-validation' not in installer
    srun = json.loads((root / 'srun.json').read_text())
    mounts = next(value for value in srun if value.startswith('--container-mounts='))
    assert f',{topology}:{topology}:ro' in mounts
    assert (root / 'base.sqsh').read_bytes() == b'original'
    assert topology.read_text() == '<system version="1"/>'


@pytest.mark.parametrize('missing', ['file', 'namespace'])
def test_bake_rejects_incomplete_topology_pair_before_container_start(bake, missing):
    root, _env = bake
    topology = root / 'topo.xml'
    topology.write_text('<system/>')
    options = {'CACHE_NCCL_TOPO_FILE': str(topology), 'CACHE_NCCL_TOPO_NAMESPACE': 'cmh-v1'}
    del options['CACHE_NCCL_TOPO_FILE' if missing == 'file' else 'CACHE_NCCL_TOPO_NAMESPACE']
    result = run_bake(bake, **options)
    assert result.returncode != 0
    assert 'must be supplied together' in result.stderr
    assert not (root / 'srun.json').exists()


@pytest.mark.parametrize(
    'path', ['relative.xml', '/scratch/a b.xml', '/scratch/a;touch.xml', '/scratch/a/../b.xml']
)
def test_bake_rejects_unsafe_topology_path_before_container_start(bake, path):
    root, _env = bake
    result = run_bake(bake, CACHE_NCCL_TOPO_FILE=path, CACHE_NCCL_TOPO_NAMESPACE='cmh-v1')
    assert result.returncode != 0
    assert 'safe absolute path' in result.stderr
    assert not (root / 'srun.json').exists()


@pytest.mark.parametrize('label', ['cmh v1', '../cmh', 'cmh;touch', 'cmh\nother=1', '-cmh'])
def test_bake_rejects_unsafe_topology_label_before_container_start(bake, label):
    root, _env = bake
    topology = root / 'topo.xml'
    topology.write_text('<system/>')
    result = run_bake(bake, CACHE_NCCL_TOPO_FILE=str(topology), CACHE_NCCL_TOPO_NAMESPACE=label)
    assert result.returncode != 0
    assert 'alphanumeric label' in result.stderr
    assert not (root / 'srun.json').exists()


def test_bake_rejects_missing_topology_file_and_never_clobbers_output(bake):
    root, _env = bake
    (root / 'derived.sqsh').write_bytes(b'existing-image')
    result = run_bake(
        bake, CACHE_NCCL_TOPO_FILE=str(root / 'missing.xml'), CACHE_NCCL_TOPO_NAMESPACE='cmh-v1'
    )
    assert result.returncode != 0
    assert 'not a readable file' in result.stderr
    assert not (root / 'srun.json').exists()
    assert (root / 'derived.sqsh').read_bytes() == b'existing-image'


def test_bake_optional_topology_preserves_existing_output(bake):
    root, _env = bake
    topology = root / 'topo.xml'
    topology.write_text('<system/>')
    (root / 'derived.sqsh').write_bytes(b'existing-image')
    result = run_bake(bake, CACHE_NCCL_TOPO_FILE=str(topology), CACHE_NCCL_TOPO_NAMESPACE='cmh-v1')
    assert result.returncode != 0
    assert 'already exists' in result.stderr
    assert not (root / 'srun.json').exists()
    assert (root / 'derived.sqsh').read_bytes() == b'existing-image'
