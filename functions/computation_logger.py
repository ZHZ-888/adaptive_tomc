"""Lightweight CSV logger for computation cost measurements."""

import csv
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


class ComputationLogger:
    """Append per-cycle computation cost records to one CSV file."""

    DEFAULT_FIELDS = [
        "run_id",
        "scenario_id",
        "step",
        "sim_time_s",
        "module",
        "phase",
        "elapsed_ms",
        "cpu_ms",
        "n_vehicles",
        "n_platoons",
        "n_candidates",
        "n_selected",
        "mode",
        "train_agent",
        "tsg_mode",
        "notes",
    ]

    def __init__(self, run_dir, filename="computation_cost_log.csv"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename

    def _append_row(self, row):
        file_exists = self.path.is_file() and self.path.stat().st_size > 0
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.DEFAULT_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self.DEFAULT_FIELDS})

    def log(
        self,
        *,
        module,
        phase,
        elapsed_ms,
        cpu_ms=None,
        step=None,
        sim_time_s=None,
        run_id="",
        scenario_id="",
        n_vehicles="",
        n_platoons="",
        n_candidates="",
        n_selected="",
        mode="",
        train_agent="",
        tsg_mode="",
        notes="",
    ):
        row = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "step": step,
            "sim_time_s": sim_time_s,
            "module": module,
            "phase": phase,
            "elapsed_ms": round(float(elapsed_ms), 6),
            "cpu_ms": round(float(cpu_ms), 6) if cpu_ms is not None else "",
            "n_vehicles": n_vehicles,
            "n_platoons": n_platoons,
            "n_candidates": n_candidates,
            "n_selected": n_selected,
            "mode": mode,
            "train_agent": train_agent,
            "tsg_mode": tsg_mode,
            "notes": notes,
        }
        self._append_row(row)

    @contextmanager
    def measure(self, *, module, phase="total", **meta):
        """Measure a code block and log one row when it finishes."""
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
            cpu_elapsed_ms = (time.process_time() - cpu_start) * 1000.0
            self.log(
                module=module,
                phase=phase,
                elapsed_ms=wall_elapsed_ms,
                cpu_ms=cpu_elapsed_ms,
                **meta,
            )

    @contextmanager
    def measure_every(self, *, step, interval, module, phase="total", **meta):
        """Measure only on selected steps; skip logging on all others."""
        if interval <= 0 or step % interval != 0:
            yield
            return

        with self.measure(module=module, phase=phase, step=step, **meta):
            yield


def measure_cycle(comp_logger, *, step, interval, module, sim_time_s, phase="total", **meta):
    """Return a measurement context for a control cycle, or a no-op context."""
    if comp_logger is None or interval <= 0 or step % interval != 0:
        return nullcontext()
    return comp_logger.measure(
        module=module,
        phase=phase,
        step=step,
        sim_time_s=sim_time_s,
        **meta,
    )
