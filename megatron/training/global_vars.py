# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import re
import signal
import sys
import time
from collections.abc import Iterator, Mapping
from datetime import timedelta

import torch

from megatron.core import Timers
from megatron.core.config import set_experimental_flag
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.jit import disable_jit_fuser
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.gtp_parallel_layout import get_batch_parallel_size
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.dist_signal_handler import DistributedSignalHandler

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_ENERGY_MONITOR = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_TELEMETRY_HANDLE = None
_GLOBAL_TELEMETRY_SETUP_TIMESTAMPS = (None, None)
_WANDB_STEP_METRIC = 'iteration'
_WANDB_HISTORY_CHUNK_BYTES = 4 * 1024 * 1024


def _wandb_history_chunks(
    data: Mapping[str, object], max_bytes: int = _WANDB_HISTORY_CHUNK_BYTES
) -> Iterator[dict[str, object]]:
    """Split scalar history conservatively below W&B's serialized row limit."""
    chunk = {}
    chunk_bytes = 2
    for metric_name, value in data.items():
        item_bytes = len(metric_name.encode('utf-8')) + len(str(value).encode('utf-8')) + 16
        if chunk and chunk_bytes + item_bytes > max_bytes:
            yield chunk
            chunk = {}
            chunk_bytes = 2
        chunk[metric_name] = value
        chunk_bytes += item_bytes
    if chunk:
        yield chunk


class _WandbSharedWriter:
    """Adapt Megatron's step-based logging to W&B shared-mode history."""

    def __init__(self, wandb_module):
        self._wandb = wandb_module
        self._pending_step = None

    def __getattr__(self, name):
        return getattr(self._wandb, name)

    def _flush_step(self):
        if self._pending_step is not None:
            self._wandb.log({}, commit=True)
            self._pending_step = None

    def log(self, data, step=None, commit=None):
        """Log against a custom iteration axis because shared mode ignores ``step``."""
        for metric_name in data:
            if '*' in metric_name:
                raise ValueError(
                    f"W&B metric names must not contain '*'; use a literal placeholder such as "
                    f"'ALL' instead: {metric_name!r}"
                )

        if step is None:
            self._flush_step()
            if commit is None:
                return self._wandb.log(data)
            return self._wandb.log(data, commit=commit)

        if self._pending_step is not None and step != self._pending_step:
            self._flush_step()

        should_commit = commit if commit is not None else False
        chunks = list(_wandb_history_chunks(data))
        if len(chunks) > 1:
            self._flush_step()
            result = None
            for chunk in chunks:
                result = self._wandb.log(
                    {_WANDB_STEP_METRIC: step, **chunk}, commit=True
                )
        else:
            self._pending_step = step
            result = self._wandb.log(
                {_WANDB_STEP_METRIC: step, **data}, commit=should_commit
            )
        if should_commit or len(chunks) > 1:
            self._pending_step = None
        return result

    def finish(self, *args, **kwargs):
        """Flush the final training iteration before finishing the primary writer."""
        self._flush_step()
        return self._wandb.finish(*args, **kwargs)


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS

def get_energy_monitor():
    """Return energy monitor."""
    _ensure_var_is_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    return _GLOBAL_ENERGY_MONITOR

def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def get_telemetry():
    """Return the telemetry handle. It can be None so no need to check if initialized."""
    return _GLOBAL_TELEMETRY_HANDLE


def get_telemetry_setup_timestamps():
    """Return the start and end timestamps for the latest telemetry setup call."""
    return _GLOBAL_TELEMETRY_SETUP_TIMESTAMPS


def _shutdown_telemetry():
    """Shut down telemetry and release its published child Resource carrier."""
    global _GLOBAL_TELEMETRY_HANDLE
    telemetry_handle = _GLOBAL_TELEMETRY_HANDLE
    _GLOBAL_TELEMETRY_HANDLE = None
    if telemetry_handle is not None:
        try:
            telemetry_handle.shutdown()
        except Exception:
            pass


def _set_signal_handler(exit_signal):

    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = DistributedSignalHandler(exit_signal).__enter__()


def _graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Flushes and shuts down telemetry
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    from megatron.training.utils import print_rank_0
    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # synchronize all ranks before exiting
            try:
                # avoid deadlock if ranks don't all reach here
                torch.distributed.barrier()
            except Exception:
                pass

            torch.distributed.destroy_process_group()
    except Exception:
        pass

    # Without this, any spans/metrics still sitting in the BatchSpanProcessor's
    # queue (which only flushes on a timer or on shutdown(), never automatically
    # on process exit) are silently dropped -- including the just-closed
    # top-level job span on this exit path. sys.exit() below unwinds the call
    # stack via SystemExit, so any span opened with `with managed_span(...)`/
    # `@trace_fn` still gets its __exit__ run and end_time set correctly; this
    # call is what makes sure that data actually reaches the exporter.
    try:
        _otel_handle = get_telemetry()
        if _otel_handle is not None:
            _otel_handle.shutdown()
    except Exception:
        pass

    sys.exit(0)


def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    if args.step_batch_size_schedule is not None:
        # Imported here, as elsewhere in this module: megatron.training.utils imports back
        # into megatron.training, which imports this module.
        from megatron.training.utils import print_rank_0
        print_rank_0(f'> using step batch size schedule: {args.step_batch_size_schedule}')

    init_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        # Independent sequences, including when CP and weight sharding share ranks.
        data_parallel_size=get_batch_parallel_size(args),
        decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
        step_batch_size_schedule=args.step_batch_size_schedule,
        seq_length=args.seq_length,
    )
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_one_logger(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_energy_monitor(args)
    _set_telemetry_with_timestamps(args)

    if args.enable_experimental:
        set_experimental_flag(True)

    if args.exit_signal_handler:
        _set_signal_handler(args.exit_signal)

    if args.exit_signal_handler_for_training:
        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    if args.disable_jit_fuser:
        disable_jit_fuser()


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_ARGS
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_ENERGY_MONITOR
    global _GLOBAL_SIGNAL_HANDLER
    global _GLOBAL_TELEMETRY_HANDLE

    _GLOBAL_ARGS = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_ENERGY_MONITOR = None
    _GLOBAL_SIGNAL_HANDLER = None
    _shutdown_telemetry()

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _wandb_run_id(exp_name: str) -> str:
    """A wandb run id derived from the experiment name.

    The id is part of the run URL, so wandb rejects the characters that would be
    ambiguous there. Experiment names are often paths, so substitute rather than fail;
    the unmodified name is still what gets displayed.
    """
    return re.sub(r'[/\\#?%:]', '-', exp_name)


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        # Private attributes are runtime state rather than configuration: the parsed pretokenized
        # config cached on args would otherwise be serialized into a ~700 KB string every time
        # wandb syncs the config. Copy so the kitchen contents added below stay off args.
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        if 'kitchen_config_file' in wandb_config and wandb_config['kitchen_config_file'] is not None:
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config['kitchen_config_file'], "r") as f:
                wandb_config['kitchen_config_file_contents'] = f.read()
        # wandb.init() creates a new run unless it is given an id to resume, and this
        # runs on every process start. A long job restarts many times -- a requeue, a
        # node failure, an in-job restarter -- and each restart would otherwise open a
        # fresh run, scattering one training run across many and leaving every chart
        # covering only the stretch between two failures. Deriving a stable id from the
        # experiment name and resuming it keeps them a single timeline.
        run_id = args.wandb_run_id or _wandb_run_id(args.wandb_exp_name)
        wandb_kwargs = {
            'dir': save_dir,
            'id': run_id,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': wandb_config,
            'settings': wandb.Settings(mode='shared', x_label='training', x_primary=True),
            'resume': 'allow'}
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        run = wandb.init(**wandb_kwargs)
        run.define_metric(_WANDB_STEP_METRIC)
        run.define_metric(
            '*', step_metric=_WANDB_STEP_METRIC, step_sync=False, overwrite=True
        )
        for namespace in ('tensor-metrics-global/*', 'tensor-metrics-breakdowns/*'):
            run.define_metric(
                namespace,
                step_metric=_WANDB_STEP_METRIC,
                step_sync=False,
                summary='none',
                overwrite=True,
            )
        _GLOBAL_WANDB_WRITER = _WandbSharedWriter(wandb)


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger and args.rank == (args.world_size - 1):
        if args.one_logger_async or getattr(args, 'wandb_project', ''):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger
            config = {
               'project': args.one_logger_project,
               'name': args.one_logger_run_name,
               'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. please go to '
                  'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                  ' for details to install it')

def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        from megatron.training.utils import print_rank_0
        print_rank_0('enabling autoresume ...')
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except ImportError:
            print_rank_0('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)

def _set_energy_monitor(args):
    """Initialize energy monitor."""
    global _GLOBAL_ENERGY_MONITOR
    _ensure_var_is_not_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    _GLOBAL_ENERGY_MONITOR = EnergyMonitor()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def _set_telemetry(args):
    """Initialise OTel telemetry handle following the wandb/tensorboard pattern."""
    global _GLOBAL_TELEMETRY_HANDLE
    from megatron.core.telemetry import telemetry as _otel

    _GLOBAL_TELEMETRY_HANDLE = _otel.setup(args)


def _set_telemetry_with_timestamps(args):
    """Capture the full telemetry setup interval without emitting a separate signal."""
    global _GLOBAL_TELEMETRY_SETUP_TIMESTAMPS
    setup_start = time.time()
    try:
        _set_telemetry(args)
    finally:
        _GLOBAL_TELEMETRY_SETUP_TIMESTAMPS = (setup_start, time.time())


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_ENERGY_MONITOR
    _GLOBAL_ENERGY_MONITOR = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None

    global _GLOBAL_TELEMETRY_HANDLE
    _shutdown_telemetry()
