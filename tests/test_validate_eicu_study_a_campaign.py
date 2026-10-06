from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_eicu_study_a_campaign.py"
CONTRACT_PATH = (
    REPO_ROOT
    / "experiments"
    / "eicu_study_a_validation"
    / "default_contract.json"
)

SPEC = importlib.util.spec_from_file_location("study_a_validator", SCRIPT_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


FIELDS = [
    "run_id",
    "role",
    "dataset",
    "scenario_name",
    "g0",
    "scenario_seed",
    "optimizer_seed",
    "seed_pair_id",
    "campaign_role",
    "method",
    "aggregation_weighting",
    "objective_mode",
    "alignment_label",
    "primary_selection_metric",
    "test_mse_used_for_selection",
    "selection_source",
    "scenario_checksum",
    "scenario_scope",
    "study_claim",
    "config_path",
    "scenario_metadata_path",
    "result_path",
    "output_root",
    "input_dim_g",
    "input_dim_f",
]


class SyntheticCampaign:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_dir = root / "configs"
        self.scenario_root = root / "scenarios"
        self.results_root = root / "results"
        self.manifest = root / "manifest.csv"
        self.config_dir.mkdir()
        self.scenario_root.mkdir()
        self.results_root.mkdir()
        self.rows: list[dict[str, Any]] = []
        self._scenario_checksums: dict[tuple[str, int], str] = {}
        self._build()

    def _build(self) -> None:
        for g0 in ("linear", "interaction", "mlp"):
            # Matches the shipped contract's frozen confirmatory scenario_seed
            # values (protocol_v1.md S7.3) -- the contract's role rules
            # declare these seeds explicitly, so a synthetic campaign must use
            # them too, not an arbitrary range.
            for seed in (101, 102, 103, 104, 105):
                artifact_name = f"{g0}_{seed}.bin"
                artifact = self.scenario_root / artifact_name
                artifact.write_bytes(f"full-eicu:{g0}:{seed}".encode("utf-8"))
                checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
                self._scenario_checksums[(g0, seed)] = checksum
                metadata = {
                    "scenario_checksum": checksum,
                    "eligible_client_provenance": {
                        "cohort_hash": f"cohort-{seed}",
                        "eligible_client_ids_hash": "clients-v1",
                        "minimum_rows_per_client": 50,
                    },
                    "g0": g0,
                    "seed": seed,
                    "dimensions": {
                        "input_dim": 8,
                        "instrument_dim": 4,
                        "outcome_dim": 1,
                    },
                    "is_demo": False,
                    "scenario_scope": "full_eicu",
                    "artifact_path": artifact_name,
                }
                self.write_json(self.scenario_root / f"{g0}_{seed}.json", metadata)

                for method in ("fedgda_s", "fedogda_s"):
                    self._add_row(
                        role="confirmatory",
                        g0=g0,
                        seed=seed,
                        method=method,
                        aggregation="uniform_clients",
                        label="primary_extension",
                    )
                for method in ("gda_d", "sgda_s", "oadam_s"):
                    self._add_row(
                        role="centralized_baseline",
                        g0=g0,
                        seed=seed,
                        method=method,
                        aggregation="none",
                        label="centralized_reference",
                    )
                for method in ("fedgda_s", "fedogda_s"):
                    self._add_row(
                        role="aggregation_ablation",
                        g0=g0,
                        seed=seed,
                        method=method,
                        aggregation="sample_size",
                        label="non_paper_aligned_aggregation_ablation",
                    )
        self.write_manifest()

    def _add_row(
        self,
        *,
        role: str,
        g0: str,
        seed: int,
        method: str,
        aggregation: str,
        label: str,
    ) -> None:
        run_id = f"{role}__{g0}__seed{seed}__{method}"
        row: dict[str, Any] = {
            "run_id": run_id,
            "role": role,
            "dataset": "eicu",
            "scenario_name": f"eicu_{g0}_seed{seed}",
            "g0": g0,
            "method": method,
            "scenario_seed": seed,
            "optimizer_seed": seed + 1000,
            "seed_pair_id": f"pair_{seed:02d}",
            "campaign_role": "aggregation_ablation" if role == "aggregation_ablation" else "",
            "aggregation_weighting": aggregation,
            "objective_mode": "paper_aligned",
            "alignment_label": label,
            "primary_selection_metric": "equal_client_validation_mse",
            "test_mse_used_for_selection": "false",
            "selection_source": "validation_only",
            "scenario_checksum": self._scenario_checksums[(g0, seed)],
            "scenario_scope": "full_eicu",
            "study_claim": "extension_no_published_target",
            "config_path": f"{run_id}.json",
            "scenario_metadata_path": f"{g0}_{seed}.json",
            "result_path": run_id,
            "output_root": "",
            "input_dim_g": 8,
            "input_dim_f": 4,
        }
        self.rows.append(row)
        self.write_config(row)

    @staticmethod
    def write_json(path: Path, document: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, allow_nan=True)
            handle.write("\n")

    def write_config(self, row: dict[str, Any]) -> None:
        config = {
            key: (
                False if key == "test_mse_used_for_selection" else row[key]
            )
            for key in (
                "run_id",
                "role",
                "dataset",
                "scenario_name",
                "g0",
                "method",
                "scenario_seed",
                "optimizer_seed",
                "seed_pair_id",
                "campaign_role",
                "aggregation_weighting",
                "objective_mode",
                "alignment_label",
                "primary_selection_metric",
                "test_mse_used_for_selection",
                "selection_source",
                "scenario_checksum",
                "scenario_scope",
                "input_dim_g",
                "input_dim_f",
            )
        }
        config["learning_rate"] = 0.01
        config["rounds"] = 20
        self.write_json(self.config_dir / str(row["config_path"]), config)

    def write_manifest(self) -> None:
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def materialize_results(self) -> None:
        for row in self.rows:
            result = self.results_root / str(row["result_path"])
            result.mkdir(parents=True, exist_ok=True)
            config = json.loads(
                (self.config_dir / str(row["config_path"])).read_text(
                    encoding="utf-8"
                )
            )
            self.write_json(result / "effective_config.json", config)
            is_primary = row["role"] == "confirmatory"
            metrics = {
                "run_id": row["run_id"],
                "method": row["method"],
                "scenario_seed": row["scenario_seed"],
                "seed": row["scenario_seed"],
                "scenario_checksum": row["scenario_checksum"],
                "selection_metric": "equal_client_validation_mse",
                "selection_source": "validation_only",
                "best_validation_round": 1,
                "selected_round": 1,
                "test_mse_at_best_validation": 1.2,
                "test_mse_reported_after_selection": True,
                "primary_equal_client_validation_mse": 1.0,
                "primary_equal_client_test_mse": 1.2,
                "secondary_sample_weighted_validation_mse": 0.9,
                "secondary_sample_weighted_test_mse": 1.1,
                "diverged": False,
                "divergence_evidence": {
                    "nonfinite_parameters": False,
                    "nonfinite_metrics": False,
                },
                "is_primary": is_primary,
                "alignment_label": row["alignment_label"],
            }
            self.write_json(result / "metrics.json", metrics)
            with (result / "mse_by_round.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["round", "primary_val_mse"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"round": 0, "primary_val_mse": 2.0},
                        {"round": 1, "primary_val_mse": 1.0},
                    ]
                )
            (result / "predictions.npz").write_bytes(b"synthetic-npz-placeholder")
            (result / "best_checkpoint.pt").write_bytes(b"best")
            (result / "final_checkpoint.pt").write_bytes(b"final")
            if row["role"] == "confirmatory":
                (result / "per_client_metrics.csv").write_text(
                    "client_id,validation_mse,test_mse\n1,1.0,1.2\n",
                    encoding="utf-8",
                )

    def row(
        self,
        *,
        role: str,
        g0: str = "linear",
        seed: int = 101,
        method: str | None = None,
    ) -> dict[str, Any]:
        for row in self.rows:
            if (
                row["role"] == role
                and row["g0"] == g0
                and row["scenario_seed"] == seed
                and (method is None or row["method"] == method)
            ):
                return row
        raise AssertionError("synthetic row not found")


class ValidatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.campaign = SyntheticCampaign(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def validate(self, phase: str = "prelaunch") -> dict[str, Any]:
        return VALIDATOR.validate_campaign(
            manifest=self.campaign.manifest,
            contract=CONTRACT_PATH,
            config_dir=self.campaign.config_dir,
            scenario_root=self.campaign.scenario_root,
            results_root=self.campaign.results_root,
            phase=phase,
        )

    @staticmethod
    def codes(report: dict[str, Any]) -> set[str]:
        return {issue["code"] for issue in report["blocking_errors"]}

    def test_valid_complete_105_row_contract(self) -> None:
        report = self.validate()
        self.assertTrue(report["launchable"], report["blocking_errors"][:3])
        self.assertEqual(report["counts"]["manifest_rows"], 105)
        self.assertEqual(report["counts"]["fixed_rows_expected"], 105)
        self.assertEqual(report["counts"]["fixed_rows_observed"], 105)
        self.assertTrue(
            all(role["complete"] for role in report["coverage"].values())
        )

    def test_valid_complete_postrun_contract(self) -> None:
        self.campaign.materialize_results()
        report = self.validate("postrun")
        self.assertTrue(report["reportable"], report["blocking_errors"][:3])
        self.assertEqual(report["counts"]["results_validated"], 105)
        self.assertEqual(report["completion"]["confirmatory"]["completed"], 30)

    def test_missing_seed_method_pair(self) -> None:
        victim = self.campaign.row(
            role="confirmatory", g0="mlp", seed=105, method="fedogda_s"
        )
        self.campaign.rows.remove(victim)
        self.campaign.write_manifest()
        report = self.validate()
        self.assertFalse(report["launchable"])
        self.assertIn("matrix_incomplete", self.codes(report))
        self.assertIn("federated_pair_incomplete", self.codes(report))

    def test_duplicate_run_id(self) -> None:
        self.campaign.rows[1]["run_id"] = self.campaign.rows[0]["run_id"]
        self.campaign.write_manifest()
        self.assertIn("duplicate_run_id", self.codes(self.validate()))

    def test_confirmatory_sample_size_violation(self) -> None:
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        row["aggregation_weighting"] = "sample_size"
        self.campaign.write_manifest()
        report = self.validate()
        self.assertIn("sample_size_role_violation", self.codes(report))

    def test_ablation_without_explicit_label(self) -> None:
        row = self.campaign.row(role="aggregation_ablation", method="fedgda_s")
        row["alignment_label"] = "ablation"
        self.campaign.write_manifest()
        report = self.validate()
        self.assertIn("ablation_label_missing", self.codes(report))

    def test_test_mse_tuning_violation(self) -> None:
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        row["test_mse_used_for_selection"] = "true"
        self.campaign.write_manifest()
        report = self.validate()
        self.assertIn("test_mse_selection_violation", self.codes(report))

    def test_scenario_checksum_mismatch(self) -> None:
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        row["scenario_checksum"] = "0" * 64
        self.campaign.write_manifest()
        report = self.validate()
        self.assertIn("scenario_manifest_mismatch", self.codes(report))

    def test_input_dimension_mismatch(self) -> None:
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        row["input_dim_g"] = 999
        self.campaign.write_manifest()
        report = self.validate()
        self.assertIn("scenario_manifest_mismatch", self.codes(report))

    def test_demo_marked_confirmatory(self) -> None:
        path = self.campaign.scenario_root / "linear_101.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["is_demo"] = True
        self.campaign.write_json(path, metadata)
        report = self.validate()
        self.assertIn("demo_confirmatory_violation", self.codes(report))
        allowed = VALIDATOR.validate_campaign(
            manifest=self.campaign.manifest,
            contract=CONTRACT_PATH,
            config_dir=self.campaign.config_dir,
            scenario_root=self.campaign.scenario_root,
            results_root=self.campaign.results_root,
            phase="prelaunch",
            allow_demo=True,
        )
        self.assertTrue(allowed["launchable"], allowed["blocking_errors"][:3])

    def test_missing_result_artifact(self) -> None:
        self.campaign.materialize_results()
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        (
            self.campaign.results_root
            / str(row["result_path"])
            / "predictions.npz"
        ).unlink()
        report = self.validate("postrun")
        self.assertIn("result_artifact_missing", self.codes(report))
        self.assertIn("completion_count_mismatch", self.codes(report))

    def test_metrics_and_effective_config_disagreement(self) -> None:
        self.campaign.materialize_results()
        row = self.campaign.row(role="centralized_baseline", method="gda_d")
        result = self.campaign.results_root / str(row["result_path"])
        config = json.loads(
            (result / "effective_config.json").read_text(encoding="utf-8")
        )
        config["method"] = "wrong"
        self.campaign.write_json(result / "effective_config.json", config)
        metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
        metrics["scenario_checksum"] = "bad"
        self.campaign.write_json(result / "metrics.json", metrics)
        report = self.validate("postrun")
        codes = self.codes(report)
        self.assertIn("effective_config_manifest_mismatch", codes)
        self.assertIn("metrics_manifest_mismatch", codes)

    def test_poor_finite_mse_is_not_divergence(self) -> None:
        self.campaign.materialize_results()
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        metrics_path = (
            self.campaign.results_root / str(row["result_path"]) / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["diverged"] = True
        metrics["primary_equal_client_test_mse"] = 1e100
        self.campaign.write_json(metrics_path, metrics)
        report = self.validate("postrun")
        self.assertIn("false_divergence_label", self.codes(report))

    def test_nonfinite_metric_with_divergence_evidence_is_valid(self) -> None:
        self.campaign.materialize_results()
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        metrics_path = (
            self.campaign.results_root / str(row["result_path"]) / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["diverged"] = True
        metrics["primary_equal_client_test_mse"] = math.inf
        metrics["divergence_evidence"]["nonfinite_metrics"] = True
        self.campaign.write_json(metrics_path, metrics)
        report = self.validate("postrun")
        self.assertTrue(report["reportable"], report["blocking_errors"][:3])

    def test_nonfinite_metric_without_diverged_flag_is_rejected(self) -> None:
        self.campaign.materialize_results()
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        metrics_path = (
            self.campaign.results_root / str(row["result_path"]) / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["primary_equal_client_test_mse"] = math.nan
        self.campaign.write_json(metrics_path, metrics)
        report = self.validate("postrun")
        self.assertIn("missing_divergence_label", self.codes(report))

    def test_output_path_escape(self) -> None:
        row = self.campaign.row(role="confirmatory", method="fedgda_s")
        row["result_path"] = "../escaped"
        self.campaign.write_manifest()
        self.assertIn("output_path_escape", self.codes(self.validate()))

    def test_validation_is_read_only_unless_report_output_requested(self) -> None:
        def fingerprint(root: Path) -> dict[str, str]:
            return {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }

        before = fingerprint(self.campaign.root)
        report = self.validate()
        after = fingerprint(self.campaign.root)
        self.assertTrue(report["launchable"])
        self.assertEqual(before, after)

        out = self.campaign.root / "reports"
        exit_code = VALIDATOR.main(
            [
                "--manifest",
                str(self.campaign.manifest),
                "--contract",
                str(CONTRACT_PATH),
                "--config-dir",
                str(self.campaign.config_dir),
                "--scenario-root",
                str(self.campaign.scenario_root),
                "--results-root",
                str(self.campaign.results_root),
                "--out",
                str(out),
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {path.name for path in out.iterdir()},
            {
                "eicu_study_a_validation.json",
                "eicu_study_a_validation.md",
            },
        )
        for relative, digest in before.items():
            self.assertEqual(
                digest,
                hashlib.sha256((self.campaign.root / relative).read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
