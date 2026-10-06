"""Opt-in, validation-only sustained-deterioration guard and evidence."""

from collections import deque
import json
import math
from pathlib import Path
import statistics

from numerical_failure import file_sha256


POLICY_DEFAULTS = {
    "validation_deterioration_window": 0,  # Zero preserves existing training.
    "validation_deterioration_min_rounds": 100,
    "validation_deterioration_patience": 50,
    "validation_deterioration_factor": 5.0,
    "validation_deterioration_absolute_delta": 0.1,
}


class ValidationDeteriorationGuard:
    """Stop only after a sustained rise in rolling-median validation MSE."""

    def __init__(self, config):
        self.config = {key: config.get(key, default)
                       for key, default in POLICY_DEFAULTS.items()}
        self.window = int(self.config["validation_deterioration_window"])
        self.min_rounds = int(self.config["validation_deterioration_min_rounds"])
        self.patience = int(self.config["validation_deterioration_patience"])
        self.factor = float(self.config["validation_deterioration_factor"])
        self.absolute_delta = float(self.config["validation_deterioration_absolute_delta"])
        if self.window < 0 or self.min_rounds < 1 or self.patience < 1:
            raise ValueError("Invalid validation deterioration window/warmup/patience")
        if not math.isfinite(self.factor) or self.factor <= 1:
            raise ValueError("Deterioration factor must be finite and > 1")
        if not math.isfinite(self.absolute_delta) or self.absolute_delta < 0:
            raise ValueError("Deterioration absolute delta must be finite and >= 0")
        self.values = deque(maxlen=max(1, self.window))
        self.best_median = math.inf
        self.bad_count = 0
        self.round_count = 0
        self.stop = None

    def update(self, round_index, validation_mse):
        if self.window == 0:
            return None
        if round_index != self.round_count:
            raise ValueError("Deterioration guard requires consecutive rounds starting at zero")
        self.round_count += 1
        value = float(validation_mse)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Nonfinite/negative MSE belongs to the numerical failure path")
        self.values.append(value)
        if len(self.values) < self.window:
            return None
        median = statistics.median(self.values)
        self.best_median = min(self.best_median, median)
        bad = (self.round_count >= self.min_rounds
               and median > self.factor * max(self.best_median, 1e-12)
               and median - self.best_median > self.absolute_delta)
        self.bad_count = self.bad_count + 1 if bad else 0
        if self.bad_count >= self.patience:
            self.stop = {
                "status": "validation_deterioration", "round": round_index,
                "rounds_completed": self.round_count,
                "reason": "Sustained rolling-median validation deterioration",
                "rolling_validation_median": median,
                "best_rolling_validation_median": self.best_median,
                "consecutive_bad_evaluations": self.bad_count,
                "policy": self.config,
            }
        return self.stop


def replay_guard(values, config):
    guard = ValidationDeteriorationGuard(config)
    for index, value in enumerate(values):
        stop = guard.update(index, value)
        if stop:
            return stop
    return None


def stability_summary(values, window=25, tail=50):
    if len(values) < max(window, tail):
        raise ValueError("Insufficient rounds for stability assessment")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid stability curve")
    best_window = min(statistics.median(values[i:i + window])
                      for i in range(len(values) - window + 1))
    recent = sorted(values[-tail:])
    tail_median = statistics.median(recent)
    tail_p90 = recent[math.ceil(0.9 * len(recent)) - 1]
    return {"best_window_median": best_window,
            "tail_median": tail_median, "tail_p90": tail_p90,
            "tail_median_over_best_window": tail_median / max(best_window, 1e-12),
            "tail_p90_over_best_window": tail_p90 / max(best_window, 1e-12)}


def write_validation_stop(run_dir, config, stop):
    from experiment_utils import config_checksum

    run_dir = Path(run_dir)
    paths = ["effective_config.json", "mse_by_round.csv", "metrics.json",
             "predictions.npz", "checkpoints/best_validation.pt", "checkpoints/final.pt"]
    payload = {"schema_version": 1, "run_id": config["run_id"], **stop,
               "effective_config_checksum": config_checksum(config),
               "test_mse_used_for_selection": False,
               "evidence_sha256": {name: file_sha256(run_dir / name) for name in paths}}
    path = run_dir / "validation_stop.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return payload
