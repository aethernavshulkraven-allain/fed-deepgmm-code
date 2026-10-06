"""Integration tests for scripts/score_highdim_screen_post_bn_20260822.py.

Focused on the exact-tie handling gap: rank_cell's own sort key is
(-round(gmm_eval, 9), val_mse). val_mse is the documented secondary
tiebreak for a gmm_eval tie, so that alone must not be reported as
unresolved -- only a tie on BOTH keys is genuinely ambiguous, and Python's
stable sort would otherwise silently pick whichever candidate happens to
come first in the manifest.

score_screen hard-requires exactly 108 rows across exactly 12 (dataset,
method) cells (the real screen's shape), so these tests build a full,
otherwise-uncontroversial 108-row fixture and inject the case under test
into exactly one cell.
"""

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import score_highdim_screen_post_bn_20260822 as scorer  # noqa: E402


FIELDNAMES = [
    "run_id", "dataset", "method", "seed", "learning_rate", "critic_multiplier",
    "client_optimizer", "comm_round", "final_result_dir", "server_buffer_policy",
]
DATASETS = [f"fixture{i}" for i in range(6)]
METHODS = ("fedgda_d", "fedogda_d")
COMM_ROUND = scorer.SCREEN_COMM_ROUND  # 150, the frozen screen protocol length.


def write_run(run_dir: Path, *, dataset, method, run_id, seed, learning_rate,
              critic_multiplier, gmm_eval, val_mse,
              gmm_eval_by_round=None, val_mse_by_round=None) -> None:
    """Writes a full COMM_ROUND-row run. By default every round repeats the
    same (gmm_eval, val_mse) value, so the last-50 mean equals gmm_eval/val_mse
    exactly -- existing tests built around a single scalar keep working
    unchanged. Pass gmm_eval_by_round/val_mse_by_round (len == COMM_ROUND) to
    exercise a real per-round trajectory, e.g. where the best-round value and
    the last-50 mean disagree."""
    psi_series = list(gmm_eval_by_round) if gmm_eval_by_round is not None else [gmm_eval] * COMM_ROUND
    mse_series = list(val_mse_by_round) if val_mse_by_round is not None else [val_mse] * COMM_ROUND
    assert len(psi_series) == COMM_ROUND and len(mse_series) == COMM_ROUND
    best_gmm_eval = max(psi_series)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": seed, "comm_round": COMM_ROUND,
        "learning_rate": learning_rate, "critic_multiplier": critic_multiplier,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False,
        "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": False, "best_gmm_eval": best_gmm_eval, "best_validation_mse": min(mse_series),
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": None,
        "nonfinite_diagnostics": [],
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse",
            "equal_client_val_mse", "train_moment_violation",
            "val_moment_violation", "gmm_train_objective", "gmm_val_objective",
            "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ])
        for i in range(COMM_ROUND):
            mse_i, psi_i = mse_series[i], psi_series[i]
            writer.writerow([
                i, mse_i, mse_i, mse_i, "", 0.2, 0.2, 0.1, 0.1,
                psi_i, "", "", "True", "False",
            ])


class ScreenTieHandlingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_root = self.tmp / "results"
        self.rows: list[dict] = []
        self._counter = 0

    def _add_row(self, *, dataset, method, gmm_eval, val_mse,
                 gmm_eval_by_round=None, val_mse_by_round=None, critic_multiplier=None):
        self._counter += 1
        seed, lr = 0, round(0.01 * self._counter, 6)
        cm = 5.0 if critic_multiplier is None else critic_multiplier
        run_id = f"det_screen_postbn_{dataset}_{method}_{self._counter}"
        run_dir = self.results_root / dataset / method / f"seed_{seed}" / run_id
        write_run(
            run_dir, dataset=dataset, method=method, run_id=run_id, seed=seed,
            learning_rate=lr, critic_multiplier=cm,
            gmm_eval=gmm_eval, val_mse=val_mse,
            gmm_eval_by_round=gmm_eval_by_round, val_mse_by_round=val_mse_by_round,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": str(seed),
            "learning_rate": str(lr), "critic_multiplier": str(cm),
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND),
            "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    ROWS_PER_CELL = 9  # 12 cells x 9 rows = 108, matching score_screen's hard requirement.

    def _fill_remaining_cells_uncontroversially(self, skip: tuple[str, str]) -> None:
        # 12 cells total; every cell needs a clear, unambiguous winner so
        # only the cell under test can raise.
        for dataset in DATASETS:
            for method in METHODS:
                if (dataset, method) == skip:
                    continue
                for rank in range(self.ROWS_PER_CELL):
                    self._add_row(
                        dataset=dataset, method=method,
                        gmm_eval=1.0 + rank, val_mse=0.9 - rank * 0.01,
                    )

    def _write_manifest(self) -> Path:
        assert len(self.rows) == 108, f"fixture must total 108 rows, got {len(self.rows)}"
        manifest_path = self.tmp / "screen_manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def _add_cell_filler(self, cell: tuple[str, str], count: int) -> None:
        # Clearly-lower gmm_eval so these never contend for rank 1/2.
        for i in range(count):
            self._add_row(dataset=cell[0], method=cell[1], gmm_eval=-1.0 - i, val_mse=1.0)

    def test_exact_psi_and_mse_tie_raises_with_full_tie_set(self):
        # Two candidates tie exactly on both gmm_eval and val_mse -- a
        # genuinely unresolved case per rank_cell's documented tiebreak order.
        cell = ("fixture0", "fedgda_d")
        self._add_row(dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.5)
        self._add_row(dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.5)
        self._add_cell_filler(cell, self.ROWS_PER_CELL - 2)
        self._fill_remaining_cells_uncontroversially(skip=cell)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "unresolved exact Psi tie"):
            scorer.score_screen(manifest_path)

    def test_gmm_eval_tie_broken_by_distinct_val_mse_is_not_reported_as_a_tie(self):
        # Same gmm_eval, different val_mse -- val_mse is the documented
        # secondary tiebreak, so this must resolve cleanly, not raise.
        cell = ("fixture0", "fedgda_d")
        self._add_row(dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.3)
        self._add_row(dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.5)
        self._add_cell_filler(cell, self.ROWS_PER_CELL - 2)
        self._fill_remaining_cells_uncontroversially(skip=cell)
        manifest_path = self._write_manifest()
        result = scorer.score_screen(manifest_path)
        result_cell = result["cells"]["fixture0|fedgda_d"]
        # val_mse is now a mean over 150 repeated floats rather than a direct
        # passthrough, so it can carry harmless floating-point summation
        # drift -- compare numerically, not by exact equality.
        self.assertAlmostEqual(result_cell["psi_rank_1"]["val_mse"], 0.3)

    def test_best_round_and_last50_disagree_on_cell_winner(self):
        # Candidate A spikes once early (round 10) to a huge Psi that would
        # win under a best-round statistic, but is mediocre over the frozen
        # last-50 window (rounds 100..149). Candidate B is modest everywhere
        # but has the higher last-50 mean. The frozen protocol
        # (PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md) requires last-50-mean
        # ranking, so B -- not A -- must be picked as rank 1, and A's huge
        # best-round value must not leak into ranking.
        cell = ("fixture0", "fedgda_d")
        a_series = [1.0] * COMM_ROUND
        a_series[10] = 100.0  # best-round spike, outside the last-50 window
        b_series = [2.0] * COMM_ROUND
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=None, val_mse=None,
            gmm_eval_by_round=a_series, val_mse_by_round=[0.6] * COMM_ROUND,
        )
        a_run_id = self.rows[-1]["run_id"]
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=None, val_mse=None,
            gmm_eval_by_round=b_series, val_mse_by_round=[0.5] * COMM_ROUND,
        )
        b_run_id = self.rows[-1]["run_id"]
        self._add_cell_filler(cell, self.ROWS_PER_CELL - 2)
        self._fill_remaining_cells_uncontroversially(skip=cell)
        manifest_path = self._write_manifest()
        result = scorer.score_screen(manifest_path)
        result_cell = result["cells"]["fixture0|fedgda_d"]
        self.assertEqual(result_cell["psi_rank_1"]["run_id"], b_run_id)
        self.assertEqual(result_cell["psi_rank_1"]["gmm_eval"], 2.0)
        self.assertEqual(result_cell["psi_rank_2"]["run_id"], a_run_id)
        self.assertEqual(result_cell["psi_rank_2"]["gmm_eval"], 1.0)
        # best_gmm_eval is preserved as diagnostic metadata but never used
        # for ranking -- A's huge spike shows up here despite A losing rank 1.
        self.assertEqual(result_cell["psi_rank_2"]["best_gmm_eval_diagnostic"], 100.0)
        self.assertEqual(result_cell["mse_winner"]["run_id"], b_run_id)

    def test_mse_winner_at_boundary_does_not_trigger_review_when_psi_rank1_is_not(self):
        # BOUNDARY_RULE_AMENDMENT_20260818.md's replacement rule step 1 is
        # scoped to "the Psi rank-1 candidate" only. A cell whose MSE winner
        # happens to sit at the tested critic-multiplier max, while the Psi
        # rank-1 winner does not, must NOT be flagged for boundary review --
        # score_highdim_screen_by_psi.py's reference implementation likewise
        # only ever checks the Psi-ranked top candidate.
        cell = ("fixture0", "fedgda_d")
        psi_winner_run_id = None
        mse_winner_run_id = None
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.9, critic_multiplier=5.0,
        )
        psi_winner_run_id = self.rows[-1]["run_id"]
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=1.9, val_mse=0.85, critic_multiplier=5.0,
        )
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=1.0, val_mse=0.1, critic_multiplier=10.0,
        )
        mse_winner_run_id = self.rows[-1]["run_id"]
        self._add_cell_filler(cell, self.ROWS_PER_CELL - 3)
        self._fill_remaining_cells_uncontroversially(skip=cell)
        manifest_path = self._write_manifest()
        result = scorer.score_screen(manifest_path)
        result_cell = result["cells"]["fixture0|fedgda_d"]
        self.assertEqual(result_cell["psi_rank_1"]["run_id"], psi_winner_run_id)
        self.assertEqual(result_cell["psi_rank_1"]["cm"], 5.0)
        self.assertEqual(result_cell["mse_winner"]["run_id"], mse_winner_run_id)
        self.assertEqual(result_cell["mse_winner"]["cm"], 10.0)
        self.assertEqual(result_cell["boundary_detail"]["psi_rank_1"], [])
        self.assertNotEqual(result_cell["boundary_detail"]["mse_winner"], [])
        self.assertFalse(result_cell["boundary_review_required"])
        self.assertNotIn("fixture0|fedgda_d", result["boundary_review_cells"])

    def test_psi_rank1_at_boundary_does_trigger_review(self):
        # The inverse case: when the Psi rank-1 winner itself sits at the
        # tested max, review IS required, regardless of the MSE winner.
        cell = ("fixture0", "fedgda_d")
        self._add_row(
            dataset=cell[0], method=cell[1], gmm_eval=2.0, val_mse=0.9, critic_multiplier=10.0,
        )
        psi_winner_run_id = self.rows[-1]["run_id"]
        self._add_cell_filler(cell, self.ROWS_PER_CELL - 1)
        self._fill_remaining_cells_uncontroversially(skip=cell)
        manifest_path = self._write_manifest()
        result = scorer.score_screen(manifest_path)
        result_cell = result["cells"]["fixture0|fedgda_d"]
        self.assertEqual(result_cell["psi_rank_1"]["run_id"], psi_winner_run_id)
        self.assertNotEqual(result_cell["boundary_detail"]["psi_rank_1"], [])
        self.assertTrue(result_cell["boundary_review_required"])
        self.assertIn("fixture0|fedgda_d", result["boundary_review_cells"])


class RowDispositionsAndHashesTest(unittest.TestCase):
    """Phase 4 SS7.1: screen_results.json must account for all 108 planned
    rows individually (not just the 12 per-cell summaries), including the
    certified-terminal ones, plus source/manifest/result hashes."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_root = self.tmp / "results"
        self.rows: list[dict] = []
        self.ledger: dict = {}
        self._counter = 0

    def _add_row(self, *, dataset, method, gmm_eval, val_mse, critic_multiplier=5.0):
        self._counter += 1
        seed, lr = 0, round(0.01 * self._counter, 6)
        run_id = f"det_screen_postbn_{dataset}_{method}_{self._counter}"
        run_dir = self.results_root / dataset / method / f"seed_{seed}" / run_id
        write_run(
            run_dir, dataset=dataset, method=method, run_id=run_id, seed=seed,
            learning_rate=lr, critic_multiplier=critic_multiplier,
            gmm_eval=gmm_eval, val_mse=val_mse,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": str(seed),
            "learning_rate": str(lr), "critic_multiplier": str(critic_multiplier),
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND),
            "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    def _fill_cell(self, cell: tuple[str, str], count: int) -> None:
        for rank in range(count):
            self._add_row(dataset=cell[0], method=cell[1], gmm_eval=1.0 + rank, val_mse=0.9 - rank * 0.01)

    def _fill_remaining_cells(self, skip: tuple[str, str]) -> None:
        for dataset in DATASETS:
            for method in METHODS:
                if (dataset, method) != skip:
                    self._fill_cell((dataset, method), 9)

    def _add_terminal_row(self, cell: tuple[str, str]) -> str:
        import run_manifest  # noqa: PLC0415 -- only this fixture needs it

        self._counter += 1
        seed, lr, cm = 0, round(0.01 * self._counter, 6), 5.0
        run_id = f"det_screen_postbn_{cell[0]}_{cell[1]}_{self._counter}"
        original_dir = self.results_root / cell[0] / cell[1] / f"seed_{seed}" / run_id
        original_dir.mkdir(parents=True)
        (original_dir / "effective_config.json").write_text(json.dumps({"run_id": run_id}))
        self.rows.append({
            "run_id": run_id, "dataset": cell[0], "method": cell[1], "seed": str(seed),
            "learning_rate": str(lr), "critic_multiplier": str(cm),
            "client_optimizer": "sgd" if cell[1] == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND),
            "final_result_dir": str(original_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })
        cert_run_id = f"{run_id}_certified"
        cert_dir = self.tmp / "certification" / cert_run_id
        cert_dir.mkdir(parents=True)
        # Must match the original row's own dataset/method/seed/lr/cm --
        # validate_pretraining_failure_artifact() now cross-verifies exact
        # configuration equivalence against the manifest row, not just the
        # certification's own internal self-consistency.
        cert_config = {
            "run_id": cert_run_id, "dataset": cell[0], "variant": cell[1],
            "client_optimizer": "sgd" if cell[1] == "fedgda_d" else "ogda",
            "random_seed": seed, "comm_round": COMM_ROUND,
            "learning_rate": lr, "critic_multiplier": cm,
            "server_buffer_policy": "direct_client_aggregate",
        }
        (cert_dir / "effective_config.json").write_text(json.dumps(cert_config))
        # validate_pretraining_failure_artifact() resolves hash_bundle_id and
        # runs the real hash verifier against it -- needs a genuinely
        # verifiable bundle, not a placeholder string.
        import hashlib  # noqa: PLC0415 -- only this fixture needs it
        bundle_target = REPO_ROOT / "scripts" / "verify_protocol_hashes.py"
        bundle_path = self.tmp / "hash_bundle.json"
        bundle_path.write_text(json.dumps([
            {"path": "scripts/verify_protocol_hashes.py", "sha256": hashlib.sha256(bundle_target.read_bytes()).hexdigest()},
        ]))
        payload = {
            "schema_version": 1, "run_id": cert_run_id,
            "effective_config_checksum": run_manifest.config_checksum(cert_config),
            "failure_phase": "model_selection", "federated_rounds_started": 0,
            "model_selection_epochs_attempted": 60, "best_model_selection_score": None,
            "per_epoch_finite_status": [{"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True}],
            "first_nonfinite_epoch": None,
            "terminal_reason": "No valid model-selection candidate was selected",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: ...",
            "stdout_sha256": None, "stderr_sha256": None,
            "hash_bundle_id": str(bundle_path.relative_to(REPO_ROOT)),
            "hash_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        }
        (cert_dir / "pretraining_failure.json").write_text(json.dumps(payload))
        self.ledger[run_id] = {"certified_run_id": cert_run_id, "certified_run_dir": str(cert_dir)}
        return run_id

    def _write_manifest(self) -> Path:
        assert len(self.rows) == 108, f"fixture must total 108 rows, got {len(self.rows)}"
        manifest_path = self.tmp / "screen_manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def _write_ledger(self) -> Path:
        ledger_path = self.tmp / "certification_ledger.json"
        ledger_path.write_text(json.dumps(self.ledger))
        return ledger_path

    def test_row_dispositions_cover_all_108_rows_including_the_terminal_one(self):
        cell = ("fixture0", "fedgda_d")
        terminal_run_id = self._add_terminal_row(cell)
        self._fill_cell(cell, 8)
        self._fill_remaining_cells(skip=cell)
        manifest_path = self._write_manifest()
        ledger_path = self._write_ledger()
        result = scorer.score_screen(manifest_path, ledger_path)
        self.assertEqual(len(result["row_dispositions"]), 108)
        by_run_id = {d["run_id"]: d for d in result["row_dispositions"]}
        self.assertEqual(set(by_run_id), {row["run_id"] for row in self.rows})
        eligible = [d for d in result["row_dispositions"] if d["disposition"] == "eligible"]
        terminal = [
            d for d in result["row_dispositions"]
            if d["disposition"] == "terminal_pretraining_ineligible"
        ]
        self.assertEqual(len(eligible), 107)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["run_id"], terminal_run_id)
        self.assertEqual(len(terminal[0]["result_sha256"]), 64)
        for entry in eligible:
            self.assertEqual(len(entry["result_sha256"]), 64)
            self.assertIn("gmm_eval", entry)
            self.assertIn("val_mse", entry)

    def test_screen_manifest_and_source_hashes_present(self):
        cell = ("fixture0", "fedgda_d")
        self._fill_cell(cell, 9)
        self._fill_remaining_cells(skip=cell)
        manifest_path = self._write_manifest()
        result = scorer.score_screen(manifest_path)
        self.assertEqual(result["screen_manifest_sha256"], scorer._sha256(manifest_path))
        self.assertTrue(result["source_hashes"])
        source_paths = {entry["path"] for entry in result["source_hashes"]}
        self.assertIn("scripts/run_manifest.py", source_paths)
        # The scorer must hash itself and its ranking dependency -- a
        # silent edit to either would change every winner in this file
        # without a source hash ever catching it otherwise.
        self.assertIn("scripts/score_highdim_screen_post_bn_20260822.py", source_paths)
        self.assertIn("scripts/score_highdim_screen_by_psi.py", source_paths)
        self.assertIn(
            "experiments/highdim_coauthor_protocol_v1/doe_review_and_revised_grid.md",
            source_paths,
        )
        self.assertIn(
            "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822/"
            "HIGH_DIM_DETERMINISTIC_CLOSEOUT_PLAN_20260826.md",
            source_paths,
        )
        # The real boundary-review doc exists in this repo right now (this
        # closeout's own decision record) -- confirms the optional-hash path
        # actually picks it up when present, not just when it's absent.
        self.assertIn(
            "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822/"
            "BOUNDARY_REVIEW_20260827.md",
            source_paths,
        )
        for entry in result["source_hashes"]:
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertIn("git_revision", result["git_provenance"])

    def test_eligible_result_sha256_changes_if_the_round_curve_changes(self):
        # The exact gap a review caught: hashing only metrics.json meant
        # mse_by_round.csv -- the file the actual ranking is computed from
        # -- could change underneath a frozen result_sha256 undetected.
        cell = ("fixture0", "fedgda_d")
        self._fill_cell(cell, 9)
        self._fill_remaining_cells(skip=cell)
        manifest_path = self._write_manifest()
        first = scorer.score_screen(manifest_path)
        first_disposition = next(
            d for d in first["row_dispositions"] if d["run_id"] == self.rows[0]["run_id"]
        )
        run_dir = Path(self.rows[0]["final_result_dir"])
        with (run_dir / "mse_by_round.csv").open() as handle:
            lines = handle.readlines()
        # Perturb one data cell without touching row count/shape.
        lines[-1] = lines[-1].replace("0.9", "0.899999")
        (run_dir / "mse_by_round.csv").write_text("".join(lines))
        second = scorer.score_screen(manifest_path)
        second_disposition = next(
            d for d in second["row_dispositions"] if d["run_id"] == self.rows[0]["run_id"]
        )
        self.assertNotEqual(first_disposition["result_sha256"], second_disposition["result_sha256"])


_CURVE_HEADER = [
    "round", "train_mse", "val_mse", "primary_val_mse",
    "equal_client_val_mse", "train_moment_violation",
    "val_moment_violation", "gmm_train_objective", "gmm_val_objective",
    "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
    "finite", "diverged",
]


class Last50MeanTest(unittest.TestCase):
    """Unit tests for scorer._last50_mean isolated from the full 108-row
    score_screen fixture."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_curve(self, rows):
        run_dir = self.tmp / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_CURVE_HEADER)
            writer.writerows(rows)
        return run_dir

    def _uniform_rows(self, psi=1.0):
        return [
            [i, 0.1, 0.1, 0.1, "", 0.1, 0.1, 0.1, 0.1, psi, "", "", "True", "False"]
            for i in range(COMM_ROUND)
        ]

    def test_wrong_row_count_rejected(self):
        run_dir = self._write_curve(self._uniform_rows()[:-1])
        with self.assertRaisesRegex(ValueError, "expected exactly 150"):
            scorer._last50_mean(run_dir, "fixture-run")

    def test_duplicated_round_index_rejected(self):
        rows = self._uniform_rows()
        rows[149][0] = 148  # duplicate index inside the last-50 window
        run_dir = self._write_curve(rows)
        with self.assertRaisesRegex(ValueError, "missing, duplicated, or unordered"):
            scorer._last50_mean(run_dir, "fixture-run")

    def test_blank_value_in_window_rejected(self):
        rows = self._uniform_rows()
        rows[120][9] = ""  # blank gmm_eval inside the last-50 window
        run_dir = self._write_curve(rows)
        with self.assertRaisesRegex(ValueError, "blank"):
            scorer._last50_mean(run_dir, "fixture-run")

    def test_nonfinite_value_in_window_rejected(self):
        rows = self._uniform_rows()
        rows[130][9] = "nan"
        run_dir = self._write_curve(rows)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            scorer._last50_mean(run_dir, "fixture-run")

    def test_correct_last50_window_and_mean(self):
        rows = []
        for i in range(COMM_ROUND):
            # rounds 0-99 are far outside the window and would badly skew a
            # mean if the window boundary were off by even one round.
            psi = 0.0 if i < 100 else float(i - 99)  # rounds 100..149 -> 1..50
            rows.append([i, 0.1, 0.1, 0.1, "", 0.1, 0.1, 0.1, 0.1, psi, "", "", "True", "False"])
        run_dir = self._write_curve(rows)
        psi_mean, _ = scorer._last50_mean(run_dir, "fixture-run")
        self.assertAlmostEqual(psi_mean, sum(range(1, 51)) / 50)


if __name__ == "__main__":
    unittest.main()
