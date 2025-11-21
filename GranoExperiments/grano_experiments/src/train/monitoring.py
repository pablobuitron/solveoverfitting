import csv
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import psutil
import torch
from lightning.pytorch.callbacks import Callback


class StepStatsCallback(Callback):
    def __init__(
        self,
        log_path: Path,
        log_every_n_steps: int,
        effective_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._log_path = log_path
        self._log_every = max(1, log_every_n_steps)
        self._effective_batch = effective_batch_size
        self._last_time: Optional[float] = None
        self._buffer: list[tuple[int, float]] = []
        self._writer: Optional[csv.writer] = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        header = ["global_step", "step_duration_sec", "steps_per_sec"]
        if self._effective_batch is not None:
            header.append("samples_per_sec")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._log_path.exists()
        self._file = self._log_path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(header)

    def teardown(self, trainer, pl_module, stage: str) -> None:
        self._flush(trainer)
        if hasattr(self, "_file"):
            self._file.close()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._last_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._last_time is None:
            return
        duration = time.perf_counter() - self._last_time
        global_step = trainer.global_step
        self._buffer.append((global_step, duration))
        if len(self._buffer) >= self._log_every:
            self._flush(trainer)

    def _flush(self, trainer) -> None:
        if not self._buffer or self._writer is None:
            return
        for step, duration in self._buffer:
            steps_per_sec = 1.0 / duration if duration > 0 else 0.0
            row = [step, duration, steps_per_sec]
            if self._effective_batch is not None:
                row.append(steps_per_sec * self._effective_batch)
            self._writer.writerow(row)
        self._file.flush()
        self._buffer.clear()


class ResourceMonitor:
    def __init__(
        self,
        device: torch.device,
        interval_sec: float,
        log_path: Path,
        step_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._device = device
        self._interval = max(0.5, interval_sec)
        self._log_path = log_path
        self._step_provider = step_provider
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._log_path.exists()
        self._file = self._log_path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(
                [
                    "timestamp",
                    "global_step",
                    "cpu_percent",
                    "cpu_mem_percent",
                    "mps_allocated_gb",
                    "mps_reserved_gb",
                ]
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if hasattr(self, "_file"):
            self._file.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            step = self._step_provider() if self._step_provider else None
            alloc = None
            reserved = None
            if self._device.type == "mps":
                try:
                    alloc = torch.mps.current_allocated_memory() / (1024 ** 3)
                    reserved = torch.mps.driver_allocated_memory() / (1024 ** 3)
                except AttributeError:
                    alloc = reserved = None
            row = [
                timestamp,
                step,
                cpu_percent,
                mem.percent,
                alloc,
                reserved,
            ]
            self._writer.writerow(row)
            self._file.flush()
            self._stop.wait(self._interval)
