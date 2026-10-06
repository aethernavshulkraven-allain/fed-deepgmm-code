"""Static contracts for the strict post-BatchNorm-fix campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_highdim_psi_adjudication_post_bn_v4 as prepare_v4  # noqa: E402


SCREEN_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
)


class CorrectedScreenPacketTests(unittest.TestCase):
    def test_exact_diagnostic_has_a_passed_artifact_bound_certification(self) -> None:
        certification_path = (
            REPO_ROOT
            / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
            / "bn_buffer_diagnostic_certification.json"
        )
        certification = json.loads(certification_path.read_text())
        self.assertEqual(certification["certification_status"], "passed")
        self.assertEqual(certification["minimum_f_bn_running_variance"], 9.897330425614937e-07)
        self.assertTrue(all(certification["criteria"].values()))
        self.assertEqual(len(certification["artifact_hashes"]), 6)

    def test_full_intended_grid_is_fresh_unique_and_policy_pinned(self) -> None:
        with (SCREEN_DIR / "screen_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = {
            (
                row["dataset"], row["method"], row["seed"],
                row["learning_rate"], row["critic_multiplier"],
            )
            for row in rows
        }
        self.assertEqual(len(rows), 108)
        self.assertEqual(len(keys), 108)
        self.assertEqual(len({(row["dataset"], row["method"]) for row in rows}), 12)
        self.assertTrue(all(row["run_id"].startswith("det_screen_postbn_") for row in rows))
        self.assertTrue(all(
            row["server_buffer_policy"] == "direct_client_aggregate" for row in rows
        ))
        self.assertTrue(all(
            row["client_optimizer"] == ("sgd" if row["method"] == "fedgda_d" else "ogda")
            for row in rows
        ))

    def test_old_adjudication_launchers_fail_closed_before_gpu_work(self) -> None:
        for relative in (
            "scripts/launch_highdim_psi_adjudication_20260822_v3.sh",
            "scripts/wait_and_launch_highdim_psi_adjudication_20260819.sh",
        ):
            text = (REPO_ROOT / relative).read_text()
            refusal = text.index("REFUSING TO RUN")
            exit_index = text.index("exit 1", refusal)
            first_runner = text.find("scripts/run_manifest.py")
            self.assertGreater(exit_index, refusal)
            self.assertTrue(first_runner == -1 or exit_index < first_runner)


class LegacyEntryPointsFailClosedTests(unittest.TestCase):
    """Every pre-BatchNorm-fix launch/orchestration entry point must refuse
    to do any GPU work at all, unconditionally -- actually invoked here
    (not just checked statically), so this proves runtime behavior, not
    just where the guard text sits in the file.
    """

    LEGACY_BASH_SCRIPTS = (
        "scripts/launch_highdim_deterministic_screen_20260813.sh",
        "scripts/launch_highdim_deterministic_screen_expand_20260813.sh",
        "scripts/launch_highdim_deterministic_screen_expand2_20260818.sh",
        "scripts/launch_highdim_deterministic_screen_expand2_corrected_v1_20260819.sh",
    )

    def test_legacy_bash_launchers_refuse_unconditionally(self) -> None:
        for relative in self.LEGACY_BASH_SCRIPTS:
            with self.subTest(script=relative):
                result = subprocess.run(
                    ["bash", relative],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("REFUSING TO RUN", result.stderr)

    def test_legacy_orchestrator_refuses_unconditionally(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/orchestrate_highdim_deterministic_finals_20260813.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSING TO RUN", result.stderr)

    def test_legacy_manifests_are_all_recorded_as_ineligible(self) -> None:
        registry_path = (
            REPO_ROOT
            / "experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json"
        )
        registry = json.loads(registry_path.read_text())
        self.assertGreater(len(registry["records"]), 0)
        statuses = {record["scientific_status"] for record in registry["records"]}
        self.assertEqual(statuses, {"legacy_ineligible"})


class V4PreparationTests(unittest.TestCase):
    def test_packet_generation_is_fresh_and_stage_separated(self) -> None:
        with (SCREEN_DIR / "screen_manifest.csv").open(newline="") as handle:
            screen_rows = list(csv.DictReader(handle))
        grouped = defaultdict(list)
        for row in screen_rows:
            grouped[(row["dataset"], row["method"])].append(row)
        cells = {}
        for (dataset, method), rows in grouped.items():
            selected = []
            for row in rows[:2]:
                selected.append({
                    "run_id": row["run_id"],
                    "lr": float(row["learning_rate"]),
                    "cm": float(row["critic_multiplier"]),
                    "gmm_eval": 1.0,
                    "val_mse": 1.0,
                })
            cells[f"{dataset}|{method}"] = {
                "dataset": dataset,
                "method": method,
                "psi_rank_1": selected[0],
                "psi_rank_2": selected[1],
                "mse_winner": selected[0],
            }
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            results = root / "screen_results.json"
            results.write_text(json.dumps({
                "status": "complete",
                "manifest": str(
                    (SCREEN_DIR / "screen_manifest.csv").relative_to(REPO_ROOT)
                ),
                "planned_runs": 108,
                "server_buffer_policy": "direct_client_aggregate",
                "boundary_review_cells": [],
                "cells": cells,
            }))
            output = root / "v4"
            # DIAGNOSTIC_CERTIFICATION now points at a fresh, post-hash-
            # closure-expansion diagnostic namespace (closeout plan SS4.6)
            # that intentionally doesn't exist until Phase 3 runs it -- build
            # a self-contained, hash-bindable fixture instead of depending on
            # a real diagnostic run.
            hashed_file = REPO_ROOT / "scripts/highdim_protocol_hash_closure_20260822.py"
            launch_hashes_path = root / "fake_diagnostic_launch_hashes.json"
            launch_hashes_path.write_text(json.dumps([{
                "path": str(hashed_file.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(hashed_file.read_bytes()).hexdigest(),
            }]))
            certification_path = root / "bn_buffer_diagnostic_certification.json"
            certification_path.write_text(json.dumps({
                "certification_status": "passed",
                "launch_hash_record": str(launch_hashes_path.relative_to(REPO_ROOT)),
            }))
            with mock.patch.object(prepare_v4, "DIAGNOSTIC_CERTIFICATION", certification_path):
                counts = prepare_v4.prepare(
                    SCREEN_DIR / "screen_manifest.csv",
                    results,
                    output,
                    None,
                )
            self.assertEqual(counts, {"signal": 48, "x": 24})
            for stage, expected_count in counts.items():
                with (output / f"adjudication_{stage}_manifest.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected_count)
                self.assertTrue(all(
                    row["run_id"].startswith("det_adjudicate_v4_") for row in rows
                ))
                self.assertTrue(all(
                    row["server_buffer_policy"] == "direct_client_aggregate"
                    for row in rows
                ))
                # Every dataset in this campaign is a femnist_*/cifar10_*
                # image scenario -- V4 rows must skip the ~10 GiB-scale full
                # test-tensor write, same as every later stage.
                self.assertTrue(all(
                    row["compact_predictions_only"] == "True" for row in rows
                ))

    def test_fresh_diagnostic_is_now_certified_post_phase_3(self) -> None:
        # Real current repo state (closeout plan SS4.6 + SS6.1): Phase 3
        # populated the fresh diagnostic namespace and it certified passed --
        # V4 packet generation is no longer blocked on this specific gate.
        self.assertTrue(prepare_v4.DIAGNOSTIC_CERTIFICATION.exists())
        certification = prepare_v4._load_json(prepare_v4.DIAGNOSTIC_CERTIFICATION)
        self.assertEqual(certification["certification_status"], "passed")

    def test_screen_results_and_boundary_decision_are_now_frozen_post_phase_4(self) -> None:
        # Real current repo state (closeout plan Phase 4 SS7): the 4
        # pre-round-0 certifications were refreshed under the hardened
        # hash_bundle_sha256 check, the corrected screen was rescored
        # against them, and the boundary decision (cm=2 sufficed for
        # cifar10_z|fedogda_d, no cm=10 run) was frozen as a machine-
        # readable artifact tied to screen_results.json's exact hash. V4
        # packet generation is no longer blocked on this gate either.
        screen_results_path = (
            REPO_ROOT
            / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
            / "screen_results.json"
        )
        self.assertTrue(screen_results_path.exists())
        screen_results = json.loads(screen_results_path.read_text())
        self.assertEqual(screen_results["status"], "complete")
        self.assertEqual(screen_results["planned_runs"], 108)
        self.assertEqual(len(screen_results["cells"]), 12)
        self.assertEqual(len(screen_results["terminal_ineligible_runs"]), 4)
        self.assertEqual(len(screen_results["boundary_review_cells"]), 6)

        boundary_path = (
            REPO_ROOT
            / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
            / "BOUNDARY_REVIEW_20260827.json"
        )
        self.assertTrue(boundary_path.exists())
        boundary = json.loads(boundary_path.read_text())
        expected_hash = hashlib.sha256(screen_results_path.read_bytes()).hexdigest()
        self.assertEqual(boundary["screen_results_sha256"], expected_hash)
        self.assertEqual(set(boundary["decisions"]), set(screen_results["boundary_review_cells"]))
        self.assertTrue(all(v == "accepted_for_adjudication" for v in boundary["decisions"].values()))

    def test_diagnostic_hash_bundle_mismatch_is_rejected_not_just_status(self) -> None:
        with (SCREEN_DIR / "screen_manifest.csv").open(newline="") as handle:
            screen_rows = list(csv.DictReader(handle))
        grouped = defaultdict(list)
        for row in screen_rows:
            grouped[(row["dataset"], row["method"])].append(row)
        cells = {}
        for (dataset, method), rows in grouped.items():
            selected = [{
                "run_id": row["run_id"], "lr": float(row["learning_rate"]),
                "cm": float(row["critic_multiplier"]), "gmm_eval": 1.0, "val_mse": 1.0,
            } for row in rows[:2]]
            cells[f"{dataset}|{method}"] = {
                "dataset": dataset, "method": method,
                "psi_rank_1": selected[0], "psi_rank_2": selected[1], "mse_winner": selected[0],
            }
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            launch_hashes_path = root / "fake_diagnostic_launch_hashes.json"
            launch_hashes_path.write_text(json.dumps([{
                "path": "scripts/highdim_protocol_hash_closure_20260822.py",
                "sha256": "0" * 64,  # deliberately wrong
            }]))
            certification_path = root / "bn_buffer_diagnostic_certification.json"
            certification_path.write_text(json.dumps({
                "certification_status": "passed",
                "launch_hash_record": str(launch_hashes_path.relative_to(REPO_ROOT)),
            }))
            results = root / "screen_results.json"
            results.write_text(json.dumps({
                "status": "complete",
                "manifest": str((SCREEN_DIR / "screen_manifest.csv").relative_to(REPO_ROOT)),
                "planned_runs": 108, "server_buffer_policy": "direct_client_aggregate",
                "boundary_review_cells": [], "cells": cells,
            }))
            with mock.patch.object(prepare_v4, "DIAGNOSTIC_CERTIFICATION", certification_path):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    prepare_v4.prepare(
                        SCREEN_DIR / "screen_manifest.csv", results, root / "v4", None,
                    )

    def test_candidate_union_deduplicates_multi_label_configuration(self) -> None:
        cell = {
            "psi_rank_1": {"lr": 0.1, "cm": 5, "run_id": "a"},
            "psi_rank_2": {"lr": 0.01, "cm": 5, "run_id": "b"},
            "mse_winner": {"lr": 0.1, "cm": 5, "run_id": "a"},
        }
        candidates = prepare_v4._candidate_plan(cell)
        self.assertEqual(len(candidates), 2)
        merged = next(candidate for candidate in candidates if candidate["lr"] == 0.1)
        self.assertEqual(merged["labels"], ["psi_rank1", "mse_winner"])
        self.assertFalse(merged["reused_from_finals"])

    def test_boundary_review_is_hash_bound_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "screen_results.json"
            results_path.write_text("{}\n")
            screen = {
                "_path": str(results_path),
                "boundary_review_cells": ["femnist_z|fedogda_d"],
            }
            with self.assertRaisesRegex(ValueError, "provide a frozen"):
                prepare_v4._validate_boundary_review(screen, None)
            review_path = root / "review.json"
            review_path.write_text(json.dumps({
                "screen_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
                "decisions": {"femnist_z|fedogda_d": "accepted_for_adjudication"},
            }))
            prepare_v4._validate_boundary_review(screen, review_path)
            results_path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_v4._validate_boundary_review(screen, review_path)


if __name__ == "__main__":
    unittest.main()
