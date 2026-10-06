#!/usr/bin/env python3
"""Prepare the full fresh image screen after the BatchNorm buffer fix."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import CORE_SOURCES  # noqa: E402
PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
SOURCE_DIR = PROTOCOL_ROOT / "deterministic_screen_20260813"
OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_screen_post_bn_20260822"
RESULT_ROOT = "results/highdim_deterministic_screen_post_bn_20260822"
DIAGNOSTIC_CERTIFICATION = (
    PROTOCOL_ROOT
    / "psi_adjudication_20260822_v3/bn_buffer_diagnostic_certification.json"
)
SOURCES = (
    ("screen_manifest.csv", None),
    ("screen_expand_manifest.csv", None),
    ("screen_expand2_manifest.csv", "fedgda_d"),
    ("screen_expand2_corrected_v1_manifest.csv", None),
)
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}


def _token(value: str) -> str:
    return f"{float(value):g}".replace(".", "p")


def _load_source_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] | None = None
    selected = []
    seen = set()
    for filename, method_filter in SOURCES:
        path = SOURCE_DIR / filename
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for source_row in reader:
                if method_filter is not None and source_row["method"] != method_filter:
                    continue
                key = (
                    source_row["dataset"],
                    source_row["method"],
                    int(source_row["seed"]),
                    float(source_row["learning_rate"]),
                    float(source_row["critic_multiplier"]),
                )
                if key in seen:
                    raise ValueError(f"duplicate intended screen configuration: {key}")
                seen.add(key)
                row = dict(source_row)
                row["source_manifest"] = str(path.relative_to(REPO_ROOT))
                row["source_run_id"] = source_row["run_id"]
                selected.append(row)
    if fieldnames is None:
        raise ValueError("source screen manifests have no header")
    if len(selected) != 108:
        raise ValueError(f"expected 108 unique intended configurations, got {len(selected)}")
    for field in ("server_buffer_policy", "source_manifest", "source_run_id"):
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames, selected


def _build_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for source in source_rows:
        dataset = source["dataset"]
        method = source["method"]
        seed = int(source["seed"])
        lr_token = _token(source["learning_rate"])
        cm_token = _token(source["critic_multiplier"])
        run_id = (
            f"det_screen_postbn_{dataset}_{method}_seed{seed}_alpha0p5_"
            f"lr{lr_token}_cm{cm_token}"
        )
        row = dict(source)
        row.update({
            "run_id": run_id,
            "protocol_version": "highdim_deterministic_screen_post_bn_v1",
            "run_group": "highdim_deterministic_screen_post_bn_20260822",
            "output_root": RESULT_ROOT,
            "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}",
            "implementation_status": "ready",
            "run_status": "not_started",
            "preflight_required": "True",
            "preflight_status": "bn_buffer_diagnostic_certified",
            "client_optimizer": METHOD_OPTIMIZERS[method],
            "server_buffer_policy": "direct_client_aggregate",
            "notes": (
                "Fresh post-BatchNorm-fix screen run. No checkpoint or optimizer state "
                f"is reused; intended configuration copied from {source['source_run_id']}."
            ),
        })
        rows.append(row)
    return rows


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    certification = json.loads(DIAGNOSTIC_CERTIFICATION.read_text())
    if certification.get("certification_status") != "passed":
        raise ValueError("BatchNorm diagnostic certification is absent or not passed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames, source_rows = _load_source_rows()
    rows = _build_rows(source_rows)
    manifest = OUTPUT_DIR / "screen_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = OUTPUT_DIR / "screen_summary.json"
    summary.write_text(json.dumps({
        "campaign": "highdim_deterministic_screen_post_bn_20260822",
        "fresh_initialization": True,
        "run_count": len(rows),
        "server_buffer_policy": "direct_client_aggregate",
        "source_rule": [
            {"manifest": filename, "method_filter": method_filter}
            for filename, method_filter in SOURCES
        ],
        "diagnostic_certification": str(DIAGNOSTIC_CERTIFICATION.relative_to(REPO_ROOT)),
    }, indent=2, sort_keys=True) + "\n")
    hashed_paths = [
        manifest,
        summary,
        DIAGNOSTIC_CERTIFICATION,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        REPO_ROOT / "scripts/score_highdim_screen_by_psi.py",
        REPO_ROOT / "scripts/score_highdim_screen_post_bn_20260822.py",
        REPO_ROOT / "scripts/launch_highdim_deterministic_screen_post_bn_20260822.sh",
        REPO_ROOT / "scripts/verify_highdim_bn_diagnostic_certification_20260822.py",
        REPO_ROOT / "scripts/certify_highdim_bn_diagnostic_20260822.py",
        Path(__file__),
    ]
    (OUTPUT_DIR / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _hash(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest), "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
