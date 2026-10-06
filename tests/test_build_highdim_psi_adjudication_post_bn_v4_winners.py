"""Tests for scripts/build_highdim_psi_adjudication_post_bn_v4_winners.py --
the adapter combining score_highdim_adjudication_20260819.py's separate
signal (8 cells) and X (4 cells) results into the single 12-cell
v4_winners.json contract the stability/finals preparers consume."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_highdim_psi_adjudication_post_bn_v4_winners as adapter  # noqa: E402

DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")
SIGNAL_DATASETS = [d for d in DATASETS if not d.endswith("_x")]
X_DATASETS = [d for d in DATASETS if d.endswith("_x")]


def _winner_candidate(dataset, method, lr, cm, *, outcome="promoted"):
    run_ids = {
        str(seed): f"det_adjudicate_v4_{dataset}_{method}_seed{seed}_alpha0p5_lr{lr:g}_cm{cm:g}".replace(".", "p", 1)
        for seed in (0, 1, 2)
    }
    # run_id token replacement above is approximate; what matters for the
    # adapter is only that run_dir's basename is stable and seed-keyed --
    # mirror resolve_new_run_dir's real shape closely enough to be realistic.
    seeds = [
        {
            "seed": seed, "psi_last50_mean": 1.0, "mse_last50_mean": 0.1,
            "diverged": False, "artifacts_complete": True, "finite": True,
            "best_round_psi": 1.0, "run_dir": f"results/fake/{dataset}/{method}/seed_{seed}/{run_ids[str(seed)]}",
            "note": "", "status": "complete",
        }
        for seed in (0, 1, 2)
    ]
    return {
        "dataset": dataset, "method": method, "outcome": outcome,
        "winner": {
            "candidate_id": f"{dataset}/{method}/lr{lr:g}_cm{cm:g}",
            "label": "psi_rank1", "eligible": True, "incomplete": False,
            "median_psi": 1.0, "median_mse": 0.1, "seeds": seeds,
        } if outcome in adapter.WINNING_OUTCOMES else None,
        "eligible_candidates": ["x"], "excluded_candidates": [], "tie_set": ["x"],
        "detail": "fixture",
        "all_candidates": [],
    }


def _fixture_results(datasets, *, all_promoted=True):
    results = {}
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(METHODS):
            outcome = "promoted" if all_promoted else ("retune_required" if i == 0 and j == 0 else "promoted")
            results[f"{dataset}/{method}"] = _winner_candidate(
                dataset, method, 0.001 * (i + 1), 5.0 + j, outcome=outcome,
            )
    return results


class CombineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name, payload) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload))
        return path

    def test_combines_8_signal_and_4_x_cells_into_12(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS))
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        out_path = self.tmp / "v4_winners.json"
        result = adapter.combine(signal_path, x_path, out_path)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["alpha"], 0.5)
        self.assertEqual(result["seeds"], [0, 1, 2])
        self.assertEqual(len(result["cells"]), 12)
        self.assertEqual(set(result["cells"]), {
            f"{d}|{m}" for d in DATASETS for m in METHODS
        })
        self.assertTrue(out_path.exists())
        on_disk = json.loads(out_path.read_text())
        self.assertEqual(on_disk, result)

    def test_winner_lr_cm_and_run_ids_extracted_correctly(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS))
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        result = adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")
        cell = result["cells"]["femnist_z|fedgda_d"]
        self.assertEqual(cell["winner"]["lr"], 0.001)
        self.assertEqual(cell["winner"]["cm"], 5.0)
        self.assertEqual(set(cell["winner"]["run_ids"]), {"0", "1", "2"})
        for seed, run_id in cell["winner"]["run_ids"].items():
            self.assertIn(f"seed{seed}", run_id)

    def test_non_winning_outcome_blocks_the_whole_build(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS, all_promoted=False))
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(ValueError, "retune_required"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_overlapping_cells_between_signal_and_x_rejected(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS))
        # Deliberately corrupt: X results also claim a signal-only cell.
        bad_x = _fixture_results(X_DATASETS)
        bad_x["femnist_z/fedgda_d"] = _winner_candidate("femnist_z", "fedgda_d", 0.01, 5.0)
        x_path = self._write("x.json", bad_x)
        with self.assertRaisesRegex(ValueError, r"x results cover the wrong cells.*unexpected=\['femnist_z/fedgda_d'\]"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_wrong_signal_cell_count_rejected(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS[:-1]))
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(ValueError, "signal results cover the wrong cells"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_wrong_cell_identity_with_correct_count_rejected(self):
        # The exact exploit this hardening closes: exactly 8 signal + 4 X
        # entries, but one signal cell names a dataset outside the frozen
        # grid while a required real cell is missing entirely -- a
        # count-only check would accept this.
        bad_signal = _fixture_results(SIGNAL_DATASETS[:-1])
        bad_signal["bogus_z/fedgda_d"] = _winner_candidate("bogus_z", "fedgda_d", 0.01, 5.0)
        bad_signal["bogus_z/fedogda_d"] = _winner_candidate("bogus_z", "fedogda_d", 0.01, 5.0)
        signal_path = self._write("signal.json", bad_signal)
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(
            ValueError,
            r"missing=\['cifar10_xz/fedgda_d', 'cifar10_xz/fedogda_d'\].*"
            r"unexpected=\['bogus_z/fedgda_d', 'bogus_z/fedogda_d'\]",
        ):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_json_key_not_matching_its_own_dataset_method_fields_rejected(self):
        bad_signal = _fixture_results(SIGNAL_DATASETS)
        # The cell's own dataset/method fields say femnist_z/fedgda_d, but
        # it's filed under a different, unrelated key.
        entry = bad_signal.pop("femnist_z/fedgda_d")
        bad_signal["femnist_z/fedgda_d"] = {**entry, "method": "fedogda_d"}
        signal_path = self._write("signal.json", bad_signal)
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(ValueError, "does not match its own"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_non_positive_critic_multiplier_rejected(self):
        bad_signal = _fixture_results(SIGNAL_DATASETS)
        entry = bad_signal["femnist_z/fedgda_d"]
        entry["winner"]["candidate_id"] = "femnist_z/fedgda_d/lr0.001_cm-5"
        signal_path = self._write("signal.json", bad_signal)
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(ValueError, "non-positive or nonfinite cm"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_incomplete_winner_seed_rejected(self):
        bad_signal = _fixture_results(SIGNAL_DATASETS)
        bad_signal["femnist_z/fedgda_d"]["winner"]["seeds"][0]["artifacts_complete"] = False
        signal_path = self._write("signal.json", bad_signal)
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        with self.assertRaisesRegex(ValueError, "not complete/finite/nondivergent"):
            adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")

    def test_source_hashes_recorded_in_output(self):
        signal_path = self._write("signal.json", _fixture_results(SIGNAL_DATASETS))
        x_path = self._write("x.json", _fixture_results(X_DATASETS))
        result = adapter.combine(signal_path, x_path, self.tmp / "v4_winners.json")
        hashed_paths = {entry["path"] for entry in result["source_hashes"]}
        self.assertEqual(
            hashed_paths,
            {
                str(signal_path.relative_to(REPO_ROOT)),
                str(x_path.relative_to(REPO_ROOT)),
            },
        )
        for entry in result["source_hashes"]:
            self.assertEqual(len(entry["sha256"]), 64)

    def test_real_v4_signal_results_exist_but_x_does_not_yet(self):
        # Ground truth for the current repo state (2026-08-29): signal ran
        # for real and scored cleanly (all 8 cells resolved); X has not
        # started yet.
        v4_dir = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_post_bn_v4"
        self.assertTrue((v4_dir / "adjudication_signal_results.json").exists())
        self.assertFalse((v4_dir / "adjudication_x_results.json").exists())


if __name__ == "__main__":
    unittest.main()
