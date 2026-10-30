#!/usr/bin/env python3
"""Prepare and validate an image-contained, single-host NCCL topology template.

NCCL_TOPO_DUMP_FILE can contain the full MNNVL clique, which is too large to
feed back through NCCL's fixed-size XML parser.  Homogeneous nodes only need
one host's static PCI/NVLink skeleton: NCCL refreshes the current rank, NIC
properties, host hash, and rank IDs while loading NCCL_TOPO_FILE. Partial
topologies are valid; the recorded GPU count does not imply full node coverage.
The namespace is a producer label, not an NCCL version. XML/architecture/PCI
checks do not establish complete NCCL, driver, plugin or firmware ABI parity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

_SCHEMA = 2
_PCI_ID = re.compile(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
_PCI_FIELDS = ("vendor", "device", "subsystem_vendor", "subsystem_device", "class")
_REQUIRED_PCI_FIELDS = {"vendor", "device", "class"}
_GPU_RUNTIME_ATTRIBUTES = ("dev", "sm", "rank", "gdr", "mlopart")


def _namespace(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(
            "topology namespace must be a non-empty producer label without control characters"
        )
    return value


def _xml(data: bytes) -> ET.Element:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError("malformed NCCL topology XML") from exc
    if root.tag != "system" or root.get("version") != "1":
        raise ValueError("expected NCCL <system version='1'> topology")
    return root


def _busid(value: str) -> str:
    if not _PCI_ID.fullmatch(value):
        raise ValueError("invalid topology PCI bus ID")
    domain, bus, device_function = value.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{device_function}"


def _details(root: ET.Element) -> dict[str, object]:
    cpus = root.findall("cpu")
    if not cpus:
        raise ValueError("topology has no CPU roots")
    architecture = _normalized_arch(platform.machine())
    if {_normalized_arch(cpu.get("arch", "")) for cpu in cpus} != {architecture}:
        raise ValueError("topology architecture does not match this host")
    pci_busids = [_busid(pci.get("busid", "")) for pci in root.iter("pci")]
    if not pci_busids or len(pci_busids) != len(set(pci_busids)):
        raise ValueError("topology must contain unique PCI bus IDs")
    gpu_busids = sorted(
        _busid(pci.get("busid", "")) for pci in root.iter("pci") if pci.find("gpu") is not None
    )
    net_names = sorted(net.get("name", "") for net in root.iter("net"))
    if not gpu_busids or not net_names or any(not name.strip() for name in net_names):
        raise ValueError("topology template must contain GPUs and named network devices")
    return {
        "xml_version": 1,
        "architecture": architecture,
        "cpu_roots": len(cpus),
        "gpu_count": len(gpu_busids),
        "gpu_busids": gpu_busids,
        "net_names": net_names,
        "pci_busids": sorted(pci_busids),
    }


def _pci_value(value: str, field: str) -> str:
    width = 6 if field == "class" else 4
    if not re.fullmatch(r"0[xX][0-9a-fA-F]{1," + str(width) + r"}", value):
        raise ValueError(f"invalid PCI {field} value")
    return f"0x{int(value, 16):0{width}x}"


def _pci_fingerprint(busids: list[str], sysfs_root: Path) -> dict[str, dict[str, str | None]]:
    result = {}
    for busid in busids:
        device = (sysfs_root / busid).resolve(strict=True)
        if device.name != busid or not device.is_dir():
            raise ValueError("PCI sysfs path does not identify the referenced device")
        fields = {}
        for field in _PCI_FIELDS:
            try:
                value = (device / field).read_text().strip()
            except FileNotFoundError:
                if field in _REQUIRED_PCI_FIELDS:
                    raise
                fields[field] = None
                continue
            fields[field] = _pci_value(value, field)
        # Relative to the sysfs PCI directory: preserves bridge ancestry while
        # allowing the entire sysfs tree to be mounted at another prefix.
        fields["parent_path"] = os.path.relpath(device.parent, sysfs_root.resolve(strict=True))
        result[busid] = fields
    return result


def _check_pci_xml(root: ET.Element, fingerprint: dict[str, dict[str, str | None]]) -> None:
    """Check provided PCI IDs and ancestry, allowing omitted intermediate bridges."""

    def visit(element: ET.Element, parent: str | None = None) -> None:
        if element.tag == "pci":
            busid = _busid(element.get("busid", ""))
            current = fingerprint[busid]
            for field in _PCI_FIELDS:
                if (
                    field in element.attrib
                    and _pci_value(element.attrib[field], field) != current[field]
                ):
                    raise ValueError(f"topology XML PCI {field} does not match this host")
            if parent is not None and parent not in Path(current["parent_path"]).parts:
                raise ValueError("topology XML PCI parent is not an ancestor on this host")
            parent = busid
        for child in element:
            visit(child, parent)

    visit(root)


def _check_static_template(root: ET.Element) -> None:
    if any("host_hash" in cpu.attrib for cpu in root.iter("cpu")):
        raise ValueError("template retains host identity")
    if any(set(gpu.attrib).intersection(_GPU_RUNTIME_ATTRIBUTES) for gpu in root.iter("gpu")):
        raise ValueError("template retains runtime GPU identity")
    if any(set(net.attrib) != {"name"} for net in root.iter("net")):
        raise ValueError("template retains runtime network identity")


def _normalized_arch(value: str) -> str:
    return {"aarch64": "arm64"}.get(value, value)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def prepare_template(
    source: Path,
    output: Path,
    manifest: Path,
    namespace: str,
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
    validate_pci: bool = True,
) -> dict[str, object]:
    """Strip runtime identity and record the current referenced PCI hardware."""
    _namespace(namespace)
    if len({path.resolve() for path in (source, output, manifest)}) != 3:
        raise ValueError("source, template and manifest paths must be distinct")

    source_bytes = source.read_bytes()
    source_root = _xml(source_bytes)

    source_cpus = source_root.findall("cpu")
    if not source_cpus:
        raise ValueError("topology has no CPU roots")
    first_host_hash = source_cpus[0].get("host_hash")
    if not first_host_hash:
        raise ValueError("topology CPU root has no host_hash")
    host_cpus = [cpu for cpu in source_cpus if cpu.get("host_hash") == first_host_hash]
    if not host_cpus:
        raise ValueError("topology has no complete host to template")

    template_root = ET.Element("system", {"version": "1"})
    for source_cpu in host_cpus:
        cpu = copy.deepcopy(source_cpu)
        cpu.attrib.pop("host_hash", None)
        template_root.append(cpu)

    for gpu in template_root.iter("gpu"):
        for attr in _GPU_RUNTIME_ATTRIBUTES:
            gpu.attrib.pop(attr, None)
    for net in template_root.iter("net"):
        name = net.get("name")
        if not name:
            raise ValueError("topology network node has no name")
        net.attrib.clear()
        net.set("name", name)

    details = _details(template_root)
    fingerprint = _pci_fingerprint(details["pci_busids"], sysfs_root) if validate_pci else {}
    if validate_pci:
        _check_pci_xml(template_root, fingerprint)

    ET.indent(template_root, space="  ")
    template_bytes = ET.tostring(template_root, encoding="utf-8", xml_declaration=False) + b"\n"
    template_sha256 = hashlib.sha256(template_bytes).hexdigest()
    metadata: dict[str, object] = {
        "schema": _SCHEMA,
        "namespace": namespace,
        "template_file": output.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "template_sha256": template_sha256,
        "pci_validated": bool(validate_pci),
        "pci_fingerprint": fingerprint,
        **details,
    }

    _atomic_write(output, template_bytes)
    _atomic_write(manifest, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    return metadata


def validate_template(
    template: Path,
    manifest: Path,
    namespace: str | None = None,
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
) -> dict[str, object]:
    """Reject changed templates or referenced hardware before setting NCCL_TOPO_FILE.

    The namespace is an optional expected producer label. This checks the
    recorded host architecture and PCI subset, not the complete NCCL ABI or
    whether the partial template covers every GPU/NIC on the host.
    """
    if template.resolve() == manifest.resolve():
        raise ValueError("template and manifest paths must be distinct")
    metadata = json.loads(manifest.read_text())
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema")) is not int
        or metadata["schema"] != _SCHEMA
    ):
        raise ValueError("unsupported NCCL topology manifest")
    _namespace(metadata.get("namespace"))
    if namespace is not None and _namespace(namespace) != metadata["namespace"]:
        raise ValueError("topology producer namespace mismatch")
    if metadata.get("template_file") != template.name:
        raise ValueError("topology template path does not match its manifest")
    template_bytes = template.read_bytes()
    if metadata.get("template_sha256") != hashlib.sha256(template_bytes).hexdigest():
        raise ValueError("topology template SHA256 mismatch")
    root = _xml(template_bytes)
    _check_static_template(root)
    details = _details(root)
    if any(
        type(metadata.get(key)) is not type(value) or metadata[key] != value
        for key, value in details.items()
    ):
        raise ValueError("topology structure does not match its manifest")
    if metadata.get("pci_validated") is not True:
        raise ValueError("topology was prepared without PCI validation")
    fingerprint = _pci_fingerprint(details["pci_busids"], sysfs_root)
    if metadata.get("pci_fingerprint") != fingerprint:
        raise ValueError("topology PCI hardware or parent layout does not match its manifest")
    _check_pci_xml(root, fingerprint)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys/bus/pci/devices"))
    parser.add_argument("--skip-pci-validation", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate and (args.source is not None or args.skip_pci_validation):
        parser.error("--validate does not accept --source or --skip-pci-validation")
    if not args.validate and (args.source is None or args.namespace is None):
        parser.error("preparation requires --source and --namespace")
    try:
        if args.validate:
            metadata = validate_template(
                args.output, args.manifest, args.namespace, args.sysfs_root
            )
        else:
            metadata = prepare_template(
                args.source,
                args.output,
                args.manifest,
                args.namespace,
                args.sysfs_root,
                validate_pci=not args.skip_pci_validation,
            )
    except (ValueError, OSError) as exc:
        print(f"[NCCL TOPO CACHE] validation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(
        f"[NCCL TOPO CACHE] {'validated' if args.validate else 'prepared'} "
        f"namespace={metadata['namespace']} cpu_roots={metadata['cpu_roots']} "
        f"gpus={len(metadata['gpu_busids'])} nets={len(metadata['net_names'])} "
        f"sha256={metadata['template_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
