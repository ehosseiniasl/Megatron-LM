# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Validate optional topology templates against a small, realistic sysfs tree."""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_root_override = os.environ.get("MCORE_REPO_ROOT")
REPO = Path(_root_override) if _root_override else Path(__file__).resolve().parents[4]
HELPER = REPO / "tools/persistent_cache/prepare_nccl_topology_template.py"
_spec = importlib.util.spec_from_file_location("nccl_topology_under_test", HELPER)
topology = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(topology)
HOST_ARCH = platform.machine()
BRIDGE, GPU, NIC = "0000:00:01.0", "0000:01:00.0", "0000:02:00.0"


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(topology.platform, "machine", lambda: HOST_ARCH)
    sysfs = tmp_path / "sys/bus/pci/devices"
    sysfs.mkdir(parents=True)
    root = tmp_path / "sys/devices/pci0000:00"
    devices = {BRIDGE: root / BRIDGE, GPU: root / BRIDGE / GPU, NIC: root / BRIDGE / NIC}
    for busid, device in devices.items():
        device.mkdir(parents=True, exist_ok=True)
        values = {
            "vendor": "0x15b3" if busid == NIC else "0x10de",
            "device": "0x1234",
            "class": "0x060400" if busid == BRIDGE else "0x030200",
            "subsystem_vendor": "0x10de",
            "subsystem_device": "0x5678",
        }
        for field, value in values.items():
            (device / field).write_text(value + "\n")
        (sysfs / busid).symlink_to(device, target_is_directory=True)
    source = tmp_path / "dump.xml"
    source.write_text(
        f'<system version="1">'
        f'<cpu numaid="0" arch="{HOST_ARCH}" host_hash="host-a" affinity="ff">'
        f'<pci busid="{BRIDGE}" link_width="16">'
        f'<pci busid="{GPU}"><gpu dev="0" sm="100" rank="17" gdr="1" mlopart="-1">'
        f'<nvlink target="{GPU}" count="18"/></gpu></pci>'
        f'<pci busid="{NIC}"><nic><net name="mlx5_0" dev="3" guid="private" port="1" '
        'speed="800000" latency="0.5" gdr="1" maxconn="32"/></nic></pci></pci></cpu>'
        f'<cpu numaid="1" arch="{HOST_ARCH}" host_hash="host-a"/>'
        f'<cpu numaid="0" arch="{HOST_ARCH}" host_hash="host-b">'
        '<pci busid="0001:ff:00.0"><gpu rank="99"/></pci></cpu></system>'
    )
    output, manifest = tmp_path / "topo_template.xml", tmp_path / "topo_template.manifest.json"
    return source, output, manifest, sysfs, devices


def prepare(case, **kwargs):
    source, output, manifest, sysfs, _devices = case
    return topology.prepare_template(source, output, manifest, "cmh-producer-v1", sysfs, **kwargs)


def validate(case, **kwargs):
    _source, output, manifest, sysfs, _devices = case
    return topology.validate_template(output, manifest, sysfs_root=sysfs, **kwargs)


def test_single_host_static_template_and_partial_gpu_coverage(case):
    source, output, _manifest, _sysfs, _devices = case
    original = source.read_bytes()
    metadata = prepare(case)
    assert metadata == validate(case, namespace="cmh-producer-v1")
    assert metadata["gpu_count"] == 1
    assert metadata["gpu_busids"] == [GPU]
    assert metadata["cpu_roots"] == 2
    assert metadata["pci_busids"] == [BRIDGE, GPU, NIC]
    assert metadata["net_names"] == ["mlx5_0"]
    assert metadata["pci_validated"] is True
    assert "nccl_version" not in metadata
    assert source.read_bytes() == original
    root = ET.parse(output).getroot()
    assert len(root.findall("cpu")) == 2
    assert all("host_hash" not in cpu.attrib for cpu in root.iter("cpu"))
    gpu = next(root.iter("gpu"))
    assert not set(gpu.attrib) & {"dev", "sm", "rank", "gdr", "mlopart"}
    assert gpu.find("nvlink").get("count") == "18"
    assert next(root.iter("net")).attrib == {"name": "mlx5_0"}
    assert next(root.iter("pci")).get("link_width") == "16"
    assert metadata["pci_fingerprint"][GPU]["parent_path"].endswith(f"pci0000:00/{BRIDGE}")


@pytest.mark.parametrize(
    "field", ["vendor", "device", "subsystem_vendor", "subsystem_device", "class"]
)
def test_same_busid_with_changed_hardware_is_rejected(case, field):
    prepare(case)
    (case[4][GPU] / field).write_text("0x0001\n")
    with pytest.raises(ValueError, match="hardware or parent layout"):
        validate(case)


def test_changed_bridge_parent_is_rejected_without_changing_busid(case):
    prepare(case)
    device = case[4][GPU]
    moved = device.parent.parent / "0000:00:02.0" / GPU
    moved.parent.mkdir()
    device.rename(moved)
    (case[3] / GPU).unlink()
    (case[3] / GPU).symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="hardware or parent layout"):
        validate(case)


def test_sysfs_link_to_another_busid_is_rejected(case):
    prepare(case)
    (case[3] / GPU).unlink()
    (case[3] / GPU).symlink_to(case[4][NIC], target_is_directory=True)
    with pytest.raises(ValueError, match="path does not identify"):
        validate(case)


@pytest.mark.parametrize(
    "field", ["vendor", "device", "subsystem_vendor", "subsystem_device", "class"]
)
def test_preparation_checks_static_ids_in_the_source_xml(case, field):
    root = ET.parse(case[0]).getroot()
    pci = next(pci for pci in root.iter("pci") if pci.get("busid") == GPU)
    pci.set(field, "0x0001")
    ET.ElementTree(root).write(case[0])
    with pytest.raises(ValueError, match=f"XML PCI {field}"):
        prepare(case)
    assert not case[1].exists()


def test_preparation_rejects_a_different_xml_pci_parent(case):
    root = ET.parse(case[0]).getroot()
    bridge = next(pci for pci in root.iter("pci") if pci.get("busid") == BRIDGE)
    gpu = next(pci for pci in root.iter("pci") if pci.get("busid") == GPU)
    nic = next(pci for pci in root.iter("pci") if pci.get("busid") == NIC)
    bridge.remove(gpu)
    nic.append(gpu)
    ET.ElementTree(root).write(case[0])
    with pytest.raises(ValueError, match="XML PCI parent"):
        prepare(case)


def test_matching_xml_static_ids_are_normalized_and_missing_bridges_are_allowed(case):
    root = ET.parse(case[0]).getroot()
    cpu = root.find("cpu")
    bridge = next(pci for pci in root.iter("pci") if pci.get("busid") == BRIDGE)
    for pci in list(bridge):
        for field in ("vendor", "device", "subsystem_vendor", "subsystem_device", "class"):
            pci.set(field, (case[4][pci.get("busid")] / field).read_text().strip().upper())
        cpu.append(pci)
    cpu.remove(bridge)
    ET.ElementTree(root).write(case[0])
    assert prepare(case)["pci_busids"] == [GPU, NIC]
    validate(case)


def test_missing_optional_attributes_are_stable_but_new_values_reject(case):
    for device in case[4].values():
        (device / "subsystem_vendor").unlink()
        (device / "subsystem_device").unlink()
    metadata = prepare(case)
    assert metadata["pci_fingerprint"][GPU]["subsystem_vendor"] is None
    assert validate(case) == metadata
    (case[4][GPU] / "subsystem_vendor").write_text("0x10de\n")
    with pytest.raises(ValueError, match="hardware or parent layout"):
        validate(case)


@pytest.mark.parametrize("field", ["vendor", "device", "class"])
def test_missing_required_hardware_attribute_cannot_prepare(case, field):
    (case[4][GPU] / field).unlink()
    with pytest.raises(OSError):
        prepare(case)
    assert not case[1].exists() and not case[2].exists()


def test_missing_referenced_device_rejected_at_runtime(case):
    prepare(case)
    (case[3] / GPU).unlink()
    with pytest.raises(OSError):
        validate(case)


def test_template_digest_filename_namespace_and_metadata_are_checked(case):
    prepare(case)
    with pytest.raises(ValueError, match="namespace mismatch"):
        validate(case, namespace="different-cluster")
    renamed = case[1].with_name("other.xml")
    shutil.copyfile(case[1], renamed)
    with pytest.raises(ValueError, match="path does not match"):
        topology.validate_template(renamed, case[2], sysfs_root=case[3])
    case[1].write_bytes(case[1].read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        validate(case)


@pytest.mark.parametrize(
    "key,value", [("schema", 1), ("gpu_count", 4), ("pci_busids", []), ("pci_fingerprint", {})]
)
def test_modified_manifest_rejected(case, key, value):
    prepare(case)
    metadata = json.loads(case[2].read_text())
    metadata[key] = value
    case[2].write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate(case)


def test_template_and_sysfs_roots_can_relocate_without_changing_layout(case, tmp_path):
    metadata = prepare(case)
    moved = tmp_path / "image/nccl_topo"
    moved.mkdir(parents=True)
    for path in (case[1], case[2]):
        shutil.copyfile(path, moved / path.name)
    assert (
        topology.validate_template(moved / case[1].name, moved / case[2].name, sysfs_root=case[3])
        == metadata
    )


def test_arm_architecture_alias_and_runtime_architecture_mismatch(case, monkeypatch):
    source = case[0]
    source.write_text(source.read_text().replace(f'arch="{HOST_ARCH}"', 'arch="aarch64"'))
    monkeypatch.setattr(topology.platform, "machine", lambda: "arm64")
    assert prepare(case)["architecture"] == "arm64"
    validate(case)
    monkeypatch.setattr(topology.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="architecture"):
        validate(case)


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "<system>",
        '<system version="2"/>',
        '<system version="1"/>',
        '<system version="1"><cpu/></system>',
    ],
)
def test_malformed_or_incomplete_source_is_rejected(case, payload):
    case[0].write_text(payload)
    with pytest.raises(ValueError):
        prepare(case)


@pytest.mark.parametrize("remove", ["gpu", "net"])
def test_gpu_and_network_presence_are_required(case, remove):
    root = ET.parse(case[0]).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == remove:
                parent.remove(child)
    ET.ElementTree(root).write(case[0])
    with pytest.raises(ValueError, match="GPUs and named network"):
        prepare(case)


def test_unsafe_busid_does_not_escape_sysfs(case):
    case[0].write_text(case[0].read_text().replace(GPU, "../../foreign"))
    with pytest.raises(ValueError, match="PCI bus ID"):
        prepare(case)


def test_offline_skip_is_explicit_and_cannot_pass_runtime_validation(case):
    shutil.rmtree(case[3])
    metadata = prepare(case, validate_pci=False)
    assert metadata["pci_validated"] is False and metadata["pci_fingerprint"] == {}
    with pytest.raises(ValueError, match="without PCI validation"):
        validate(case)


def test_cli_preparation_and_validation_use_only_stdlib(case):
    source, output, manifest, sysfs, _devices = case
    common = ["--output", str(output), "--manifest", str(manifest), "--sysfs-root", str(sysfs)]
    command = [sys.executable, "-S", str(HELPER)]
    prepared = subprocess.run(
        [*command, *common, "--source", str(source), "--namespace", "producer-v1"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert prepared.returncode == 0, prepared.stderr
    checked = subprocess.run(
        [*command, *common, "--validate"], capture_output=True, text=True, timeout=20
    )
    assert checked.returncode == 0, checked.stderr
    output.write_text("corrupted")
    checked = subprocess.run(
        [*command, *common, "--validate"], capture_output=True, text=True, timeout=20
    )
    assert checked.returncode != 0
    checked = subprocess.run(
        [*command, *common, "--validate", "--skip-pci-validation"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode != 0
