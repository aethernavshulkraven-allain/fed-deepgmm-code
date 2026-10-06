#!/usr/bin/env python3
"""Freeze a fresh adjudication packet from the corrected full screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import (  # noqa: E402
    CORE_DATASET_FILES,
    CORE_PROTOCOL_DOCS,
    CORE_SOURCES,
)
from verify_protocol_hashes import verify_hashes  # noqa: E402

PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
DEFAULT_SCREEN_DIR = PROTOCOL_ROOT / "deterministic_screen_post_bn_20260822"
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "psi_adjudication_post_bn_v4"
RESULT_ROOT = "results/highdim_psi_adjudication_post_bn_v4"
# Rewired 2026-08-26 (closeout plan SS4.6) away from the retrospectively
# certified psi_adjudication_20260822_v3 diagnostic to a fresh, post-hash-
# closure-expansion diagnostic that has not been generated yet -- this path
# intentionally does not exist until closeout plan Phase 3 runs it, so V4
# packet generation stays blocked (fails closed below) until then.
DIAGNOSTIC_CERTIFICATION = (
    PROTOCOL_ROOT
    / "bn_diagnostic_fresh_20260826/bn_buffer_diagnostic_certification.json"
)


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate_boundary_review(screen_results: dict, review_path: Path | None) -> None:
    required = list(screen_results.get("boundary_review_cells", []))
    if not required:
        return
    if review_path is None:
        raise ValueError(
            "screen has boundary winners; provide a frozen --boundary-review before promotion"
        )
    review = _load_json(review_path)
    expected_hash = _sha256(Path(screen_results["_path"]))
    if review.get("screen_results_sha256") != expected_hash:
        raise ValueError("boundary review does not match the current screen results")
    decisions = review.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("boundary review decisions must be an object")
    missing = sorted(set(required) - set(decisions))
    extra = sorted(set(decisions) - set(required))
    if missing or extra:
        raise ValueError(f"boundary review mismatch; missing={missing}, extra={extra}")
    rejected = {
        cell: decision
        for cell, decision in decisions.items()
        if decision != "accepted_for_adjudication"
    }
    if rejected:
        raise ValueError(f"boundary expansion/review is unresolved: {rejected}")


def _candidate_plan(cell: dict) -> list[dict]:
    candidates: dict[tuple[float, float], dict] = {}
    for label, field in (
        ("psi_rank1", "psi_rank_1"),
        ("psi_rank2", "psi_rank_2"),
        ("mse_winner", "mse_winner"),
    ):
        candidate = cell[field]
        key = (float(candidate["lr"]), float(candidate["cm"]))
        entry = candidates.setdefault(key, {
            "lr": key[0],
            "cm": key[1],
            "labels": [],
            "source_screen_run_ids": [],
            "reused_from_finals": False,
        })
        entry["labels"].append(label)
        if candidate["run_id"] not in entry["source_screen_run_ids"]:
            entry["source_screen_run_ids"].append(candidate["run_id"])
    return [candidates[key] for key in sorted(candidates)]


def prepare(
    screen_manifest_path: Path,
    screen_results_path: Path,
    output_dir: Path,
    boundary_review_path: Path | None,
) -> dict:
    screen_results = _load_json(screen_results_path)
    if not isinstance(screen_results, dict) or screen_results.get("status") != "complete":
        raise ValueError("corrected screen results are absent or incomplete")
    if screen_results.get("server_buffer_policy") != "direct_client_aggregate":
        raise ValueError("corrected screen results do not carry the frozen buffer policy")
    expected_manifest = str(screen_manifest_path.relative_to(REPO_ROOT))
    if screen_results.get("manifest") != expected_manifest:
        raise ValueError("corrected screen results refer to a different manifest")
    if int(screen_results.get("planned_runs", -1)) != 108:
        raise ValueError("corrected screen results do not cover all 108 planned runs")
    if not isinstance(screen_results.get("cells"), dict) or len(screen_results["cells"]) != 12:
        raise ValueError("corrected screen results must contain exactly 12 image cells")
    screen_results["_path"] = str(screen_results_path)
    _validate_boundary_review(screen_results, boundary_review_path)

    certification = _load_json(DIAGNOSTIC_CERTIFICATION)
    if certification.get("certification_status") != "passed":
        raise ValueError("BatchNorm diagnostic certification is not passed")
    # Bind the exact hash bundle the diagnostic certified against, not just
    # its status string -- a status field alone can't detect source drift
    # since the diagnostic was certified (closeout plan SS4.6).
    launch_hash_record = certification.get("launch_hash_record")
    if not launch_hash_record:
        raise ValueError("diagnostic certification is missing its launch_hash_record")
    verify_hashes(REPO_ROOT / str(launch_hash_record))

    with screen_manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        screen_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    templates = {}
    screen_by_run_id = {}
    for row in screen_rows:
        templates.setdefault((row["dataset"], row["method"]), row)
        screen_by_run_id[row["run_id"]] = row
    if len(screen_by_run_id) != 108:
        raise ValueError("corrected screen manifest must contain 108 unique run_ids")
    for cell in screen_results["cells"].values():
        for field in ("psi_rank_1", "psi_rank_2", "mse_winner"):
            candidate = cell[field]
            source = screen_by_run_id.get(candidate["run_id"])
            if source is None:
                raise ValueError(f"{field} references an unknown corrected-screen run")
            expected = (
                str(cell["dataset"]),
                str(cell["method"]),
                float(candidate["lr"]),
                float(candidate["cm"]),
            )
            actual = (
                source["dataset"],
                source["method"],
                float(source["learning_rate"]),
                float(source["critic_multiplier"]),
            )
            if actual != expected:
                raise ValueError(f"{field} does not match its corrected-screen manifest row")
    for field in ("server_buffer_policy", "source_screen_run_ids", "compact_predictions_only"):
        if field not in fieldnames:
            fieldnames.append(field)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    stage_counts = {}
    for stage, suffix_predicate in (
        ("signal", lambda dataset: not dataset.endswith("_x")),
        ("x", lambda dataset: dataset.endswith("_x")),
    ):
        plan = []
        rows = []
        for cell_name, cell in sorted(screen_results["cells"].items()):
            dataset, method = cell["dataset"], cell["method"]
            if not suffix_predicate(dataset):
                continue
            candidates = _candidate_plan(cell)
            plan.append({"dataset": dataset, "method": method, "candidates": candidates})
            template = templates[(dataset, method)]
            for candidate in candidates:
                for seed in (0, 1, 2):
                    lr = float(candidate["lr"])
                    cm = float(candidate["cm"])
                    run_id = (
                        f"det_adjudicate_v4_{dataset}_{method}_seed{seed}_alpha0p5_"
                        f"lr{_token(lr)}_cm{_token(cm)}"
                    )
                    row = dict(template)
                    row.update({
                        "run_id": run_id,
                        "protocol_version": "highdim_psi_adjudication_post_bn_v4",
                        "run_group": "highdim_psi_adjudication_post_bn_v4",
                        "seed": str(seed),
                        "output_root": RESULT_ROOT,
                        "final_result_dir": (
                            f"{RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}"
                        ),
                        "run_status": "not_started",
                        "preflight_required": "True",
                        "preflight_status": "bn_buffer_diagnostic_certified",
                        "comm_round": "500",
                        "learning_rate": f"{lr:g}",
                        "critic_multiplier": f"{cm:g}",
                        "server_buffer_policy": "direct_client_aggregate",
                        # Every dataset in this campaign is a femnist_*/cifar10_*
                        # image scenario; skip the ~10 GiB-scale full test-tensor
                        # write for every V4 run (closeout plan Phase 1 SS4.4) --
                        # same as every later stage in this pipeline.
                        "compact_predictions_only": "True",
                        "source_screen_run_ids": ";".join(candidate["source_screen_run_ids"]),
                        # The template row's own source_manifest/source_run_id
                        # describe *that* screen row's provenance (e.g. which
                        # expansion manifest it came from) -- an artifact of
                        # whichever row happened to be picked as the template
                        # for this (dataset, method) cell, not this new V4
                        # row's provenance. This row's real lineage is already
                        # recorded precisely in source_screen_run_ids above;
                        # carrying the template's stale fields forward would
                        # misattribute this row to an unrelated ancestor.
                        "source_manifest": "",
                        "source_run_id": "",
                        "notes": (
                            f"Fresh corrected-screen promotion ({','.join(candidate['labels'])}); "
                            "no pre-fix model state or optimizer state is reused."
                        ),
                    })
                    rows.append(row)
        manifest_path = output_dir / f"adjudication_{stage}_manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary_path = output_dir / f"adjudication_{stage}_summary.json"
        summary_path.write_text(json.dumps({
            "campaign": "highdim_psi_adjudication_post_bn_v4",
            "fresh_initialization": True,
            "model_state_reuse": False,
            "new_runs": len(rows),
            "server_buffer_policy": "direct_client_aggregate",
            "source_screen_results": str(screen_results_path.relative_to(REPO_ROOT)),
            "plan": plan,
        }, indent=2, sort_keys=True) + "\n")
        generated_paths.extend((manifest_path, summary_path))
        stage_counts[stage] = len(rows)

    provenance_paths = [
        screen_results_path,
        screen_manifest_path,
        DIAGNOSTIC_CERTIFICATION,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        REPO_ROOT / "scripts/score_highdim_adjudication_20260819.py",
        REPO_ROOT / "scripts/launch_highdim_psi_adjudication_post_bn_v4_signal.sh",
        REPO_ROOT / "scripts/launch_highdim_psi_adjudication_post_bn_v4_x.sh",
        REPO_ROOT / "scripts/verify_highdim_bn_diagnostic_certification_20260822.py",
        REPO_ROOT / "scripts/certify_highdim_bn_diagnostic_20260822.py",
        Path(__file__),
    ]
    if boundary_review_path is not None:
        provenance_paths.append(boundary_review_path)
    hash_records = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(generated_paths + provenance_paths)
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(
        json.dumps(hash_records, indent=2, sort_keys=True) + "\n"
    )
    return stage_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-manifest",
        type=Path,
        default=DEFAULT_SCREEN_DIR / "screen_manifest.csv",
    )
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=DEFAULT_SCREEN_DIR / "screen_results.json",
    )
    parser.add_argument("--boundary-review", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        counts = prepare(
            args.screen_manifest.resolve(),
            args.screen_results.resolve(),
            args.output_dir.resolve(),
            args.boundary_review.resolve() if args.boundary_review else None,
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"ADJUDICATION PACKET BLOCKED: {exc}")
        return 2
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
