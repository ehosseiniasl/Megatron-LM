# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Convert a window of Megatron OpenTelemetry trace archives into a Perfetto trace.

The per-node OTel Collector file exporter writes length-prefixed, zstd-compressed OTLP JSON
records to ``node-<host>.otlp[-<rotation time>-size].zstd``. This tool finds the restart
cycles in a telemetry root, resolves an iteration or checkpoint range of one cycle to a time
window, binary-searches every node archive for that window, and writes a gzipped Perfetto
protobuf trace with one track per rank. Input files are only ever opened read-only.

    python tools/telemetry.py ROOT --list-cycles
    python tools/telemetry.py ROOT --cycle 4103814.0.3 --iterations 20-50 -o out.pftrace.gz
    python tools/telemetry.py ROOT --cycle 4103814.0.3 --checkpoints 2-4 -o out.pftrace.gz
"""

import argparse
import collections
import ctypes
import ctypes.util
import datetime
import glob
import gzip
import multiprocessing
import os
import re
import struct
import sys
import time

try:
    from orjson import loads as _loads
except ImportError:
    from json import loads as _loads

try:
    import zstandard

    def _decompress(frame):
        return zstandard.ZstdDecompressor().decompress(frame)

except ImportError:  # Fall back to the system libzstd so the tool runs on bare login nodes.
    _libzstd = ctypes.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
    _libzstd.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    _libzstd.ZSTD_getFrameContentSize.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _libzstd.ZSTD_decompress.restype = ctypes.c_size_t
    _libzstd.ZSTD_decompress.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _libzstd.ZSTD_isError.argtypes = [ctypes.c_size_t]

    def _decompress(frame):
        size = _libzstd.ZSTD_getFrameContentSize(frame, len(frame))
        if size >= 1 << 63:
            raise ValueError('zstd frame without a content size')
        out = ctypes.create_string_buffer(size)
        n = _libzstd.ZSTD_decompress(out, size, frame, len(frame))
        if _libzstd.ZSTD_isError(n):
            raise ValueError('corrupt zstd frame')
        return out.raw[:n]


_MAGIC = b'\x28\xb5\x2f\xfd'  # zstd frame magic number.
_CHUNK = 1 << 16
_SLACK_NS = 60 * 10**9  # Tolerated out-of-order export between records of one archive.
_END_RE = re.compile(rb'"endTimeUnixNano":"(\d+)"')
_NODE_RE = re.compile(r'^node-(.+?)\.otlp')
_SCALARS = ('stringValue', 'boolValue', 'doubleValue')
_TRAINING = 'nv.dl.training'
_ITERATION = 'nv.dl.training.iteration'
_SAVE = 'nv.dl.training.checkpoint.save'
_STEP = 'nv.dl.training.step'
_RESOURCE_KEYS = frozenset(('nv.dl.run.uuid', 'nv.dl.rank', 'service.name'))
_CYCLE_KEYS = _RESOURCE_KEYS | {
    'slurm.array.job_id',
    'slurm.job.id',
    'slurm.restart_count',
    'nv.nvrx.cycle.index',
    'nv.dl.world_size',
}
# Perfetto clock ids: builtin REALTIME and BOOTTIME, and a sequence-scoped incremental clock
# so that event timestamps are encoded as small deltas.
_REALTIME, _BOOTTIME, _DELTA_CLOCK = 1, 6, 64
_SUMMARY_UUID, _RANKS_UUID = 1, 2
_SUMMARY_SEQ, _CLOCK_SEQ = 1, 2


# Minimal protobuf encoding of the Perfetto trace format (protos/perfetto/trace/).


def _varint(value):
    if value < 0x80:
        return bytes((value,))
    out = bytearray()
    while value > 0x7F:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _int(field, value):
    return _varint(field << 3) + _varint(value & 0xFFFFFFFFFFFFFFFF)


def _bytes(field, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _packet(payload):
    return _bytes(1, payload)  # Trace.packet


def _annotation(name_iid, value):
    if isinstance(value, bool):
        encoded = _int(2, value)
    elif isinstance(value, int):
        encoded = _int(4, value)
    elif isinstance(value, float):
        encoded = _varint(5 << 3 | 1) + struct.pack('<d', value)
    else:
        encoded = _bytes(6, str(value))
    return _bytes(4, _int(1, name_iid) + encoded)  # TrackEvent.debug_annotations


def _track(uuid, name, parent=None, order=None, explicit_children=False):
    payload = _int(1, uuid) + _bytes(2, name)
    if parent is not None:
        payload += _int(5, parent)
    if explicit_children:
        payload += _int(11, 3)  # ChildTracksOrdering.EXPLICIT
    if order is not None:
        payload += _int(12, order)
    return _packet(_bytes(60, payload))


def _clocks(timestamp, *clock_ids):
    return b''.join(_bytes(1, _int(1, clock) + _int(2, timestamp)) for clock in clock_ids)


def _encode_sequence(seq, spans, lane_uuid):
    """Encode ``(start, end, name, args)`` spans as nested slices on one packet sequence.

    Perfetto requires slices on a track to nest, so a span that does not nest in a lane moves
    to the next lane. Returns the encoded packets and the number of lanes used.
    """
    spans.sort(key=lambda span: (span[0], -span[1]))
    stacks, events = [], []
    for span in spans:
        start, end = span[0], span[1]
        for lane, stack in enumerate(stacks):
            while stack and stack[-1] <= start:
                events.append((stack.pop(), lane, None))
            if not stack or end <= stack[-1]:
                break
        else:
            lane = len(stacks)
            stacks.append([])
        stacks[lane].append(end)
        events.append((start, lane, span))
    for lane, stack in enumerate(stacks):
        while stack:
            events.append((stack.pop(), lane, None))
    if not events:
        return b'', 0
    # The sort is stable, so events of one lane keep their nesting order at equal timestamps.
    events.sort(key=lambda event: event[0])

    seq_field = _int(10, seq)
    previous = events[0][0]
    defaults = _int(58, _DELTA_CLOCK) + _bytes(11, _int(11, lane_uuid(0)))
    out = bytearray(
        _packet(
            seq_field
            + _int(13, 1)  # SEQ_INCREMENTAL_STATE_CLEARED
            + _bytes(
                6,
                _clocks(previous, _BOOTTIME)
                + _bytes(1, _int(1, _DELTA_CLOCK) + _int(2, previous) + _int(3, 1)),
            )
            + _bytes(59, defaults)
        )
    )
    names, keys = {}, {}
    for timestamp, lane, span in events:
        event = _int(9, 2 if span is None else 1)  # TYPE_SLICE_END or TYPE_SLICE_BEGIN
        if lane:
            event += _int(11, lane_uuid(lane))
        interned = b''
        if span is not None:
            iid = names.get(span[2])
            if iid is None:
                iid = names[span[2]] = len(names) + 1
                interned += _bytes(2, _int(1, iid) + _bytes(2, span[2]))
            event += _int(10, iid)
            for key, value in span[3]:
                kid = keys.get(key)
                if kid is None:
                    kid = keys[key] = len(keys) + 1
                    interned += _bytes(3, _int(1, kid) + _bytes(2, key))
                event += _annotation(kid, value)
        payload = _int(8, timestamp - previous) + seq_field + _bytes(11, event)
        if interned:
            payload += _bytes(12, interned)
        out += _packet(payload)
        previous = timestamp
    return bytes(out), len(stacks)


# Reading the collector archives.


def _frames(f):
    """Yield the zstd frames of consecutive records, starting at the current file offset."""
    while True:
        header = f.read(4)
        if len(header) < 4:
            return
        (length,) = struct.unpack('>I', header)
        frame = f.read(length)
        if len(frame) < length:  # Truncated record at the end of a file still being written.
            return
        yield frame


def _probe(f, offset, size):
    """Return (offset, frame) of the first whole record in the chunk at offset, or None.

    A zstd frame magic number is a record boundary if the length before it points at the
    next record's magic number or at the end of the file. One read per probe keeps the
    number of Lustre round trips down.
    """
    f.seek(offset)
    chunk = f.read(_CHUNK)
    pos = chunk.find(_MAGIC, 4)
    while pos != -1:
        end = pos + struct.unpack('>I', chunk[pos - 4 : pos])[0]
        if chunk[end + 4 : end + 8] == _MAGIC or (end == len(chunk) and offset + end == size):
            return offset + pos - 4, chunk[pos:end]
        pos = chunk.find(_MAGIC, pos + 1)
    return None


def _seek(f, size, target):
    """Return a record boundary at or before the first record holding a span ending after target.

    Records are appended in export order, so their latest span end time increases through the
    file and can be binary-searched without reading the whole archive.
    """
    lo, hi = 0, size
    while hi - lo > _CHUNK:
        mid = (lo + hi) // 2
        probe = _probe(f, mid, size)
        ends = _END_RE.findall(_decompress(probe[1])) if probe and probe[0] < hi else None
        if ends and max(map(int, ends)) < target:
            lo = probe[0]
        else:
            hi = mid
    return lo


def _attrs(attributes, keys=None):
    out = {}
    for attr in attributes:
        if keys is not None and attr['key'] not in keys:
            continue
        for kind, value in attr['value'].items():
            out[attr['key']] = (
                int(value) if kind == 'intValue' else value if kind in _SCALARS else str(value)
            )
    return out


def _discover(root):
    """Map each (archive directory, host) to its archives as (mtime_ns, size, path), oldest first."""
    units = collections.defaultdict(list)
    for path in glob.glob(
        os.path.join(glob.escape(root), '**', 'node-*.otlp*zstd'), recursive=True
    ):
        match = _NODE_RE.match(os.path.basename(path))
        if match:
            stat = os.stat(path)
            units[(os.path.dirname(path), match.group(1))].append(
                (stat.st_mtime_ns, stat.st_size, path)
            )
    return {key: sorted(files) for key, files in units.items()}


def _launch_start(archive_dir):
    """Return the job start time (ns) recorded by the launch that wrote an archive directory."""
    try:
        with open(os.path.join(archive_dir, '..', '..', 'launch-times.txt'), 'rb') as f:
            return int(float(f.readline().split(b'=')[1]) * 1e9)
    except (OSError, IndexError, ValueError):
        return None


def _index_unit(paths):
    """Scan one node's full history for per-cycle iteration and checkpoint-save times."""
    cycles = {}
    for path in paths:
        with open(path, 'rb') as f:
            for frame in _frames(f):
                raw = _decompress(frame)
                if (
                    b'"nv.dl.training.iteration"' not in raw
                    and b'"nv.dl.training.checkpoint.save"' not in raw
                ):
                    continue
                for resource_spans in _loads(raw).get('resourceSpans', ()):
                    attributes = resource_spans.get('resource', {}).get('attributes', ())
                    resource = _attrs(attributes, _CYCLE_KEYS)
                    run = resource.get('nv.dl.run.uuid')
                    if resource.get('service.name') != _TRAINING or run is None:
                        continue
                    cycle = cycles.setdefault(
                        run,
                        {
                            'label': '{}.{}.{}'.format(
                                resource.get('slurm.array.job_id') or resource.get('slurm.job.id'),
                                resource.get('slurm.restart_count', 0),
                                resource.get('nv.nvrx.cycle.index', 0),
                            ),
                            'world': resource.get('nv.dl.world_size'),
                            _ITERATION: {},
                            _SAVE: {},
                        },
                    )
                    for scope_spans in resource_spans.get('scopeSpans', ()):
                        for span in scope_spans.get('spans', ()):
                            table = cycle.get(span['name'])
                            if table is None:
                                continue
                            step = _attrs(span.get('attributes', ())).get(_STEP)
                            if step is None:
                                continue
                            start, end = int(span['startTimeUnixNano']), int(
                                span['endTimeUnixNano']
                            )
                            old = table.get(step, (start, end))
                            table[step] = (min(old[0], start), max(old[1], end))
    return cycles


def _collect(path, run, t0, t1, limit, with_args, tracks, iterations):
    """Collect spans of one cycle overlapping [t0, t1] from an archive into per-rank tracks."""
    # Lustre reports 4 MiB blocks, which would make every binary-search probe read 4 MiB.
    with open(path, 'rb', buffering=_CHUNK) as f:
        f.seek(_seek(f, os.fstat(f.fileno()).st_size, t0 - _SLACK_NS))
        for frame in _frames(f):
            min_end = None
            for resource_spans in _loads(_decompress(frame)).get('resourceSpans', ()):
                attributes = resource_spans.get('resource', {}).get('attributes', ())
                resource = _attrs(attributes, _RESOURCE_KEYS)
                keep = resource.get('nv.dl.run.uuid') == run and 'nv.dl.rank' in resource
                if keep:
                    rank, service = int(resource['nv.dl.rank']), resource.get('service.name', '')
                    spans = tracks[(rank, service)]
                for scope_spans in resource_spans.get('scopeSpans', ()):
                    for span in scope_spans.get('spans', ()):
                        start, end = int(span['startTimeUnixNano']), int(span['endTimeUnixNano'])
                        min_end = end if min_end is None else min(min_end, end)
                        if not keep or end < t0 or start > t1:
                            continue
                        attrs = _attrs(span.get('attributes', ()))
                        spans.append(
                            (start, end, span['name'], list(attrs.items()) if with_args else ())
                        )
                        if span['name'] == _ITERATION and service == _TRAINING:
                            iterations.append((attrs.get(_STEP), rank, start))
            if min_end is not None and min_end > limit:
                break


def _convert_unit(task):
    """Encode one node's spans in the window; returns (packets, iteration starts, slice count)."""
    unit_id, host, paths, run, t0, t1, limit, with_args = task
    tracks, iterations = collections.defaultdict(list), []
    for path in paths:
        try:
            _collect(path, run, t0, t1, limit, with_args, tracks, iterations)
        except (OSError, ValueError, KeyError) as e:
            print(f'warning: skipping the rest of {path}: {e!r}', file=sys.stderr)
    tracks = {key: spans for key, spans in tracks.items() if spans}
    if not tracks:
        return b'', [], 0
    ranks = sorted({rank for rank, _ in tracks})
    host_uuid = (1 << 40) + unit_id
    out = bytearray(
        _track(host_uuid, f'{host} (ranks {ranks[0]}-{ranks[-1]})', _RANKS_UUID, ranks[0], True)
    )
    services, slices = {}, 0
    for (rank, service), spans in sorted(tracks.items()):
        svc = services.setdefault(service, len(services))
        uuid = lambda lane: (rank + 1) << 16 | svc << 12 | lane
        body, lanes = _encode_sequence((rank + 1) << 4 | svc, spans, uuid)
        name = (
            f'rank {rank}' if service == _TRAINING else f'rank {rank} {service.rsplit(".", 1)[-1]}'
        )
        for lane in range(lanes):
            label = f'{name} #{lane}' if lane else name
            out += _track(uuid(lane), label, host_uuid, rank << 8 | svc << 4 | min(lane, 15))
        out += body
        slices += len(spans)
    return bytes(out), [(step, rank, start, host) for step, rank, start in iterations], slices


def _summary(iterations):
    """Per-iteration slices from the earliest to the latest iteration start across ranks."""
    by_step = collections.defaultdict(list)
    for step, rank, start, host in iterations:
        if step is not None:
            by_step[step].append((start, rank, host))
    spans = []
    for step, starts in by_step.items():
        starts.sort()
        (first, first_rank, _), (last, last_rank, last_host) = starts[0], starts[-1]
        args = [
            ('first_rank', first_rank),
            ('last_rank', last_rank),
            ('last_host', last_host),
            ('ranks', len(starts)),
        ]
        spans.append((first, last, f'iteration {step}', args))
    uuid = lambda lane: _SUMMARY_UUID if lane == 0 else (1 << 41) + lane
    body, lanes = _encode_sequence(_SUMMARY_SEQ, spans, uuid)
    tracks = b''.join(
        _track(uuid(lane), 'Iteration start skew' + (f' #{lane}' if lane else ''))
        for lane in range(lanes)
    )
    return tracks + body


def _merge_cycles(results):
    """Merge per-node cycle indexes into the list of cycles that trained, oldest first."""
    cycles = {}
    for result in results:
        for run, cycle in result.items():
            merged = cycles.setdefault(run, {**cycle, 'run': run, _ITERATION: {}, _SAVE: {}})
            for kind in (_ITERATION, _SAVE):
                for step, (start, end) in cycle[kind].items():
                    old = merged[kind].get(step, (start, end))
                    merged[kind][step] = (min(old[0], start), max(old[1], end))
    cycles = sorted(
        (c for c in cycles.values() if c[_ITERATION]),
        key=lambda c: min(start for start, _ in c[_ITERATION].values()),
    )
    labels = collections.Counter(c['label'] for c in cycles)
    for cycle in cycles:
        if labels[cycle['label']] > 1:
            cycle['label'] += '-' + cycle['run'][:8]
    return cycles


def _utc(ns):
    return datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S'
    )


def _print_cycles(cycles):
    print(
        f'{"cycle":<16} {"ranks":>6} {"iterations":>12} {"checkpoints":>18}  '
        f'{"start (UTC)":<20} {"end (UTC)":<20}'
    )
    for cycle in cycles:
        steps = sorted(cycle[_ITERATION])
        saves = sorted(cycle[_SAVE])
        ckpts = f'{len(saves)} ({saves[0]}-{saves[-1]})' if saves else '0'
        start = min(s for s, _ in cycle[_ITERATION].values())
        end = max(e for _, e in cycle[_ITERATION].values())
        print(
            f'{cycle["label"]:<16} {cycle["world"] or "?":>6} {f"{steps[0]}-{steps[-1]}":>12} '
            f'{ckpts:>18}  {_utc(start):<20} {_utc(end):<20}'
        )


def _window(cycle, args):
    """Resolve the requested iteration or checkpoint range to (t0, t1, description)."""
    steps = cycle[_ITERATION]
    if args.iterations:
        first, last = args.iterations
        missing = [step for step in (first, last) if step not in steps]
        if missing:
            sys.exit(
                f'error: iterations {missing} not in cycle {cycle["label"]} '
                f'({min(steps)}-{max(steps)})'
            )
        return steps[first][0], steps[last][1], f'iterations {first}-{last}'
    first, last = args.checkpoints
    if last == first:
        last = first + 1
    saves = sorted(cycle[_SAVE].items(), key=lambda item: item[1][0])
    if not 1 <= first < last <= len(saves):
        sys.exit(
            f'error: cycle {cycle["label"]} has {len(saves)} checkpoint saves; got {first}-{last}'
        )
    (first_step, (t0, _)), (last_step, (t1, _)) = saves[first - 1], saves[last - 1]
    return t0, t1, f'checkpoints {first}-{last} (saves at iterations {first_step} and {last_step})'


def _range(text):
    first, _, last = text.partition('-')
    return int(first), int(last or first)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('root', help='telemetry root, e.g. <run>/telemetry')
    parser.add_argument('-o', '--output', help='output path, e.g. window.pftrace.gz')
    parser.add_argument('--list-cycles', action='store_true', help='list restart cycles and exit')
    parser.add_argument(
        '--cycle', help='restart cycle from --list-cycles; required if there are several'
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument('--iterations', type=_range, metavar='A[-B]', help='iterations A through B')
    window.add_argument(
        '--checkpoints',
        type=_range,
        metavar='A[-B]',
        help='from the start of the cycle\'s A-th checkpoint save to the start of its B-th '
        '(default A+1)',
    )
    parser.add_argument(
        '--grace', type=float, default=120, help='seconds to read past the window for late spans'
    )
    parser.add_argument('--no-args', action='store_true', help='drop span attributes')
    parser.add_argument(
        '--no-summary', action='store_true', help='drop the iteration start-skew track'
    )
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    root = os.path.realpath(args.root)
    if not args.list_cycles:
        if not args.output or not (args.iterations or args.checkpoints):
            parser.error('--output and one of --iterations/--checkpoints are required')
        output = os.path.realpath(args.output)
        if os.path.commonpath([root, output]) == root:
            parser.error('refusing to write inside the telemetry root')

    begin = time.time()
    units = _discover(root)
    starts = {archive_dir: _launch_start(archive_dir) for archive_dir, _ in units}
    launches = collections.defaultdict(list)
    for key, files in units.items():
        launches[starts[key[0]] or key[0]].append((sum(size for _, size, _ in files), key))
    # The largest archives of each launch belong to nodes that trained through all its cycles.
    references = [key for group in launches.values() for _, key in sorted(group, reverse=True)[:2]]

    with multiprocessing.Pool(args.workers) as pool:
        paths = [[path for _, _, path in units[key]] for key in references]
        cycles = _merge_cycles(pool.imap_unordered(_index_unit, paths))
        print(
            f'found {len(cycles)} restart cycles by scanning {len(references)} of {len(units)} '
            f'node archive sets ({time.time() - begin:.1f}s)',
            file=sys.stderr,
        )
        if args.list_cycles:
            _print_cycles(cycles)
            return
        matches = [c for c in cycles if args.cycle in (None, c['label'])]
        if len(matches) != 1:
            _print_cycles(cycles)
            sys.exit('error: pass one of the cycles above with --cycle')
        cycle = matches[0]
        t0, t1, what = _window(cycle, args)
        limit = t1 + int(args.grace * 1e9)
        tasks = []
        for unit_id, ((archive_dir, host), files) in enumerate(sorted(units.items())):
            paths, previous = [], starts[archive_dir] or 0
            for mtime, _, path in files:
                # An archive holds the records exported after its predecessor's last write.
                if mtime >= t0 - _SLACK_NS and previous <= limit:
                    paths.append(path)
                previous = mtime
            if paths:
                tasks.append((unit_id, host, paths, cycle['run'], t0, t1, limit, not args.no_args))
        print(
            f'cycle {cycle["label"]}, {what}: {_utc(t0)} to {_utc(t1)} UTC '
            f'({(t1 - t0) / 1e9:.0f}s); reading {sum(len(task[2]) for task in tasks)} archives '
            f'of {len(tasks)} nodes',
            file=sys.stderr,
        )
        iterations, slices = [], 0
        with gzip.open(output, 'wb') as f:
            f.write(_packet(_int(10, _CLOCK_SEQ) + _bytes(6, _clocks(t0, _BOOTTIME, _REALTIME))))
            f.write(_track(_RANKS_UUID, 'Ranks', explicit_children=True))
            for body, starts, count in pool.imap(_convert_unit, tasks):
                f.write(body)
                iterations += starts
                slices += count
            if not args.no_summary:
                f.write(_summary(iterations))
    print(
        f'wrote {slices} slices to {output} ({os.path.getsize(output) / 1e6:.1f} MB) '
        f'in {time.time() - begin:.1f}s',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
