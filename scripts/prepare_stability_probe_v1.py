#!/usr/bin/env python3
"""Prepare the Tier-1 stability probe manifest.

The stochastic finals showed severe last-iterate blowup on the image-`x`
scenarios: best validation lands in the first 1-2% of rounds and the curve
then rises 1-2 orders of magnitude. Both learning rates in the searched grid
already blew up within the 150-round tuning horizon, and the fixed
`critic_multiplier=10` / `server_learning_rate=1.5` were never varied.

This probe tests whether stability is reachable with existing config knobs
only (no code changes), on the two worst cells at alpha=0.1, seed 0:

* `cifar10_x` / `fedgda_s`  (baseline: lr=0.01, best round ~13, final ~18x best)
* `cifar10_xz` / `fedogda_s` (baseline: lr=0.01, NaN excursions, final ~5x best)

Grid: learning_rate {0.001, 0.003} x critic_multiplier {1, 3}
x server_learning_rate {1.0, 1.5} = 8 configs per cell, 16 runs total,
full 1500 rounds. Rows are ordered most-conservative-first and interleaved
across the two cells so a serial single-GPU queue yields an early verdict for
both cells. Success criterion (fixed in advance): final-vs-best validation
gap <= 1.5x with best round > 500, without materially worsening best-val MSE.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1" / "stability_probe_v1_20260722"
MANIFEST_CSV = PROBE_DIR / "probe_manifest.csv"
SUMMARY_JSON = PROBE_DIR / "probe_design.json"
OUTPUT_ROOT = "results/stability_probe_v1_20260722"

CELLS = (
    {"dataset": "cifar10_x", "method": "fedgda_s", "method_label": "FedGDA-S", "client_optimizer": "sgd"},
    {"dataset": "cifar10_xz", "method": "fedogda_s", "method_label": "FedOGDA-S", "client_optimizer": "ogda"},
)

# Most conservative first so the serial queue front-loads the configs most
# likely to demonstrate stability; the baseline-adjacent configs run last.
GRID = (
    {"learning_rate": 0.001, "critic_multiplier": 1.0, "server_learning_rate": 1.0},
    {"learning_rate": 0.001, "critic_multiplier": 1.0, "server_learning_rate": 1.5},
    {"learning_rate": 0.001, "critic_multiplier": 3.0, "server_learning_rate": 1.0},
    {"learning_rate": 0.003, "critic_multiplier": 1.0, "server_learning_rate": 1.0},
    {"learning_rate": 0.001, "critic_multiplier": 3.0, "server_learning_rate": 1.5},
    {"learning_rate": 0.003, "critic_multiplier": 1.0, "server_learning_rate": 1.5},
    {"learning_rate": 0.003, "critic_multiplier": 3.0, "server_learning_rate": 1.0},
    {"learning_rate": 0.003, "critic_multiplier": 3.0, "server_learning_rate": 1.5},
)

FIXED = {
    "protocol_version": "highdim_coauthor_protocol_v1_stability_probe_v1",
    "run_group": "stability_probe_v1_alpha0p1",
    "training_scope": "federated",
    "seed": 0,
    "alpha": 0.1,
    "output_root": OUTPUT_ROOT,
    "implementation_status": "ready",
    "run_status": "not_started",
    "preflight_required": "False",
    "preflight_status": "not_required",
    "model": "lr",
    "federated_optimizer": "FedAvg",
    "client_num_in_total": 1000,
    "client_num_per_round": 10,
    "comm_round": 1500,
    "epochs": 3,
    "batch_size": 256,
    "partition_method": "hetero",
    "partition_alpha": 0.1,
    "data_cache_dir": "data",
    "learning_rate_status": "stability_probe_grid",
    "weight_decay": 0.05,
    "gradient_clip_norm": 1.0,
    "simple_model_selection_epochs": 100,
    "f_history_model_selection_epochs": 60,
    "model_selection_batch_size": 200,
    "using_gpu": "True",
    "gpu_id": "",
}


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> int:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in GRID:
        for cell in CELLS:
            run_id = (
                f"probe_{cell['dataset']}_{cell['method']}_seed0_alpha0p1"
                f"_lr{_tag(config['learning_rate'])}"
                f"_cm{_tag(config['critic_multiplier'])}"
                f"_slr{_tag(config['server_learning_rate'])}"
            )
            row = {
                "run_id": run_id,
                **FIXED,
                **cell,
                **config,
                "final_result_dir": (
                    f"{OUTPUT_ROOT}/{cell['dataset']}/{cell['method']}/seed_0/{run_id}"
                ),
                "notes": (
                    "Tier-1 stability probe: existing config knobs only; "
                    "success = final-vs-best validation gap <= 1.5x with best round > 500."
                ),
            }
            rows.append(row)

    fieldnames = list(rows[0])
    with MANIFEST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with SUMMARY_JSON.open("w") as handle:
        json.dump(
            {
                "manifest": str(MANIFEST_CSV.relative_to(REPO_ROOT)),
                "output_root": OUTPUT_ROOT,
                "runs": len(rows),
                "cells": [f"{c['dataset']}/{c['method']}" for c in CELLS],
                "grid": {
                    "learning_rate": [0.001, 0.003],
                    "critic_multiplier": [1.0, 3.0],
                    "server_learning_rate": [1.0, 1.5],
                },
                "fixed": {
                    "alpha": 0.1,
                    "seed": 0,
                    "comm_round": 1500,
                    "weight_decay": 0.05,
                    "gradient_clip_norm": 1.0,
                },
                "baseline_reference": {
                    "cifar10_x/fedgda_s": "final matrix run at lr=0.01, cm=10, slr=1.5",
                    "cifar10_xz/fedogda_s": "final matrix run at lr=0.01, cm=10, slr=1.5",
                },
                "success_criterion": "final-vs-best validation gap <= 1.5x and best round > 500",
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(json.dumps({"manifest": str(MANIFEST_CSV.relative_to(REPO_ROOT)), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
