"""Tests for the pre-run tuning-selection correctness guards added to
``select_eicu_study_a_v2_tuning.py``:

* ``at_grid_edge`` / ``grid_edge_flags`` / ``grid_edge_warnings`` -- flags a
  winning tuning candidate that sits on the boundary of the explored
  learning_rate / server_learning_rate grid.
* the horizon-mismatch fields recorded on every selected group.
* ``build_horizon_confirmation_manifest`` / ``write_horizon_confirmation_manifest``
  -- generates (never launches) a 6-row validation-only manifest that
  re-runs each selected config at the final horizon.

Does not exercise the pre-existing selection-rule tests already covered by
``tests/test_eicu_study_a_v2_offhours.py::TestTuningSelection``.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import select_eicu_study_a_v2_tuning as select_v2  # noqa: E402


_BASE_LR = float(select_v2.load_protocol()["tuning"]["base_learning_rate"])
REAL_GRID_LEARNING_RATES = tuple(
    _BASE_LR * m for m in select_v2.LOCAL_LR_MULTIPLIERS
)
REAL_GRID_SERVER_LRS = select_v2.SERVER_LEARNING_RATES


def _grid_index(learning_rate: float, server_lr: float) -> int:
    """Index of a grid point in the order _write_group_runs emits them."""
    return REAL_GRID_LEARNING_RATES.index(learning_rate) * len(
        REAL_GRID_SERVER_LRS
    ) + REAL_GRID_SERVER_LRS.index(server_lr)


def _write_group_runs(root: Path, g0: str, method: str, winner_index: int) -> list[dict]:
    """One candidate per protocol grid point; candidate winner_index has the
    lowest validation MSE (and is therefore selected).

    The grid is read from the protocol rather than restated here, so amending
    the protocol widens these fixtures with it.
    """
    rows = []
    index = 0
    for lr in REAL_GRID_LEARNING_RATES:
        for slr in REAL_GRID_SERVER_LRS:
            row = {
                "g0": g0,
                "method": method,
                "output_root": str(root),
                "dataset": "eicu_test",
                "optimizer_seed": 1011,
                "run_id": f"{g0}-{method}-{index}",
                "learning_rate": lr,
                "server_learning_rate": slr,
            }
            path = select_v2.run_dir(row)
            path.mkdir(parents=True)
            val_mse = 1.0 if index == winner_index else 10.0 + index
            metrics = {
                "diverged": False,
                "best_validation_mse": val_mse,
                "equal_client_validation_moment_violation_at_best_validation": 1.0,
            }
            (path / "metrics.json").write_text(json.dumps(metrics))
            rows.append(row)
            index += 1
    return rows


class GridEdgeFlagsTest(unittest.TestCase):
    """Unit tests for the edge-detection logic itself.

    These use a deliberately synthetic grid rather than the protocol's, so
    that widening the real grid cannot silently invalidate them -- the
    low/interior/high property under test is independent of which points the
    protocol happens to explore.
    """

    SYNTHETIC_LRS = (0.1, 0.2, 0.3)

    def _lr_candidates(self):
        return [
            {"learning_rate": lr, "server_learning_rate": 1.0}
            for lr in self.SYNTHETIC_LRS
        ]

    def test_middle_learning_rate_is_not_at_edge(self):
        winner_row = {"learning_rate": 0.2, "server_learning_rate": 1.0}
        edge, lr_grid, slr_grid = select_v2.grid_edge_flags(
            self._lr_candidates(), winner_row
        )
        self.assertNotIn("learning_rate", edge)
        self.assertEqual(lr_grid, list(self.SYNTHETIC_LRS))

    def test_low_and_high_learning_rate_are_at_edge(self):
        low_edge, _, _ = select_v2.grid_edge_flags(
            self._lr_candidates(), {"learning_rate": 0.1, "server_learning_rate": 1.0}
        )
        high_edge, _, _ = select_v2.grid_edge_flags(
            self._lr_candidates(), {"learning_rate": 0.3, "server_learning_rate": 1.0}
        )
        self.assertEqual(low_edge["learning_rate"], "low")
        self.assertEqual(high_edge["learning_rate"], "high")

    def test_two_point_server_learning_rate_is_always_at_edge(self):
        # With only 2 explored points there is no interior value, so every
        # winner is at an edge on that parameter -- intentional, see
        # grid_edge_flags docstring. Amendment 1 widened the real server-LR
        # grid to 3 points precisely so an interior winner became expressible,
        # but the 2-point property still holds and is worth pinning.
        candidates = [
            {"learning_rate": 0.1, "server_learning_rate": slr} for slr in (1.0, 2.0)
        ]
        low, _, _ = select_v2.grid_edge_flags(
            candidates, {"learning_rate": 0.1, "server_learning_rate": 1.0}
        )
        high, _, _ = select_v2.grid_edge_flags(
            candidates, {"learning_rate": 0.1, "server_learning_rate": 2.0}
        )
        self.assertEqual(low["server_learning_rate"], "low")
        self.assertEqual(high["server_learning_rate"], "high")

    def test_single_explored_value_has_no_edge(self):
        # Degenerate/malformed input: only one distinct value explored.
        # There is nothing to be "at the edge" of.
        candidates = [{"learning_rate": 0.001, "server_learning_rate": 1.0}] * 6
        edge, lr_grid, slr_grid = select_v2.grid_edge_flags(
            candidates, {"learning_rate": 0.001, "server_learning_rate": 1.0}
        )
        self.assertEqual(edge, {})
        self.assertEqual(lr_grid, [0.001])


class SelectGridEdgeAndHorizonFieldsTest(unittest.TestCase):
    def test_interior_winner_is_not_flagged_edge(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = []
            # An interior point on both axes. The pre-amendment grid had only
            # two server-LR points, so that axis could never be interior; the
            # widened grid makes a genuinely interior winner expressible.
            interior_lr = REAL_GRID_LEARNING_RATES[1]
            interior_slr = REAL_GRID_SERVER_LRS[1]
            winner = _grid_index(interior_lr, interior_slr)
            for g0 in select_v2.G0_VARIANTS:
                for method in select_v2.FEDERATED_METHODS:
                    rows.extend(_write_group_runs(root, g0, method, winner_index=winner))
            selected, report = select_v2.select(rows)
            choice = selected["linear:fedgda_s"]
            self.assertAlmostEqual(choice["learning_rate"], interior_lr)
            self.assertNotIn("learning_rate", choice["at_grid_edge_parameters"])
            self.assertNotIn("server_learning_rate", choice["at_grid_edge_parameters"])
            self.assertFalse(choice["at_grid_edge"])
            self.assertEqual(choice["tuning_rounds"], select_v2.TUNING_ROUNDS)
            self.assertEqual(choice["final_rounds"], select_v2.FINAL_ROUNDS)
            # Amendment 1 tunes and reports at one budget, so there is no
            # mismatch left to warn about.
            self.assertEqual(choice["horizon_mismatch_warning"], "")
            self.assertEqual(
                report["linear:fedgda_s"]["selected"]["at_grid_edge"],
                choice["at_grid_edge"],
            )

    def test_edge_winner_flags_both_parameters(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = []
            for g0 in select_v2.G0_VARIANTS:
                for method in select_v2.FEDERATED_METHODS:
                    # Winner index 0 -> lowest lr and lowest slr: both low edge.
                    rows.extend(_write_group_runs(root, g0, method, winner_index=0))
            selected, _ = select_v2.select(rows)
            choice = selected["linear:fedgda_s"]
            self.assertEqual(choice["at_grid_edge_parameters"]["learning_rate"], "low")
            self.assertEqual(choice["at_grid_edge_parameters"]["server_learning_rate"], "low")

    def test_grid_edge_warnings_lists_every_flagged_group(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = []
            for g0 in select_v2.G0_VARIANTS:
                for method in select_v2.FEDERATED_METHODS:
                    rows.extend(_write_group_runs(root, g0, method, winner_index=0))
            selected, _ = select_v2.select(rows)
            warnings = select_v2.grid_edge_warnings(selected)
            # 3 g0 x 2 methods = 6 groups, all winning at index 0 (an edge).
            self.assertEqual(len(warnings), 6)
            self.assertTrue(all("grid edge" in w for w in warnings))


class HorizonConfirmationManifestTest(unittest.TestCase):
    def _fake_selected(self):
        return {
            f"{g0}:{method}": {
                "learning_rate": 0.001,
                "server_learning_rate": 1.0,
                "run_id": f"tuning_{g0}_{method}_winner",
            }
            for g0 in select_v2.G0_VARIANTS
            for method in select_v2.FEDERATED_METHODS
        }

    def _fake_metadata(self):
        return {
            g0: {
                "n_clients": 179,
                "input_dim": 47,
                "instrument_dim": 47,
                "scenario_checksum_sha256": f"deadbeef_{g0}",
                "scenario_scope": "demo",
            }
            for g0 in select_v2.G0_VARIANTS
        }

    def test_builds_exactly_six_rows_covering_every_group(self):
        rows = select_v2.build_horizon_confirmation_manifest(
            self._fake_selected(),
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        self.assertEqual(len(rows), 6)
        keys = {(row["g0"], row["method"]) for row in rows}
        expected = {
            (g0, method)
            for g0 in select_v2.G0_VARIANTS
            for method in select_v2.FEDERATED_METHODS
        }
        self.assertEqual(keys, expected)

    def test_rounds_default_to_the_final_campaign_horizon(self):
        """Confirmation runs must use the horizon the campaign reports at.

        This previously also asserted the rounds differed from TUNING_ROUNDS.
        Amendment 1 set the two budgets equal on purpose, so that inequality is
        no longer a defect to guard against -- only the positive assertion is
        meaningful now.
        """
        rows = select_v2.build_horizon_confirmation_manifest(
            self._fake_selected(),
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        self.assertTrue(all(row["comm_round"] == select_v2.FINAL_ROUNDS for row in rows))

    def test_carries_over_selected_hyperparameters_unchanged(self):
        selected = self._fake_selected()
        selected["linear:fedogda_s"]["learning_rate"] = 0.002
        selected["linear:fedogda_s"]["server_learning_rate"] = 1.5
        rows = select_v2.build_horizon_confirmation_manifest(
            selected,
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        row = next(r for r in rows if r["g0"] == "linear" and r["method"] == "fedogda_s")
        self.assertAlmostEqual(row["learning_rate"], 0.002)
        self.assertAlmostEqual(row["server_learning_rate"], 1.5)

    def test_is_validation_only_like_the_tuning_selection(self):
        rows = select_v2.build_horizon_confirmation_manifest(
            self._fake_selected(),
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        self.assertTrue(all(row["test_mse_used_for_selection"] == "false" for row in rows))
        self.assertTrue(all(row["selection_source"] == "validation_only" for row in rows))
        self.assertTrue(all(row["run_status"] == "not_started" for row in rows))
        self.assertTrue(all(row["implementation_status"] == "generated_not_launched" for row in rows))

    def test_run_manifest_mandatory_fields_are_present(self):
        # scripts/run_manifest.py's _build_config reads these keys without a
        # default (row["dataset"], row["method"], row["run_id"],
        # row["comm_round"], row["epochs"], row["client_num_in_total"],
        # row["client_num_per_round"], row["batch_size"],
        # row["client_optimizer"], row["seed"], row["partition_alpha"]) --
        # a "well-formed" generated manifest must supply all of them, even
        # though this task never launches it.
        mandatory = (
            "dataset", "method", "run_id", "comm_round", "epochs",
            "client_num_in_total", "client_num_per_round", "batch_size",
            "client_optimizer", "seed", "partition_alpha",
        )
        rows = select_v2.build_horizon_confirmation_manifest(
            self._fake_selected(),
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        for row in rows:
            for field in mandatory:
                self.assertIn(field, row, f"missing mandatory field {field!r}")
                self.assertNotEqual(row[field], "", f"empty mandatory field {field!r}")

    def test_missing_group_in_selected_raises(self):
        selected = self._fake_selected()
        del selected["linear:fedogda_s"]
        with self.assertRaises(KeyError):
            select_v2.build_horizon_confirmation_manifest(
                selected,
                self._fake_metadata(),
                dataset="eicu_semisynth_offhours_v2_demo",
                output_root="/tmp/fake_output_root",
                data_cache_dir="/tmp/fake_data_cache_dir",
            )

    def test_missing_scenario_metadata_raises(self):
        metadata = self._fake_metadata()
        del metadata["mlp"]
        with self.assertRaises(KeyError):
            select_v2.build_horizon_confirmation_manifest(
                self._fake_selected(),
                metadata,
                dataset="eicu_semisynth_offhours_v2_demo",
                output_root="/tmp/fake_output_root",
                data_cache_dir="/tmp/fake_data_cache_dir",
            )

    def test_write_produces_round_trippable_json_and_csv(self):
        rows = select_v2.build_horizon_confirmation_manifest(
            self._fake_selected(),
            self._fake_metadata(),
            dataset="eicu_semisynth_offhours_v2_demo",
            output_root="/tmp/fake_output_root",
            data_cache_dir="/tmp/fake_data_cache_dir",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = Path(tempdir) / "horizon_confirmation_manifest.json"
            json_path, csv_path = select_v2.write_horizon_confirmation_manifest(rows, out_path)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            with json_path.open() as handle:
                reloaded = json.load(handle)
            self.assertEqual(len(reloaded), 6)
            with csv_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 6)
            self.assertEqual(csv_rows[0]["comm_round"], str(select_v2.FINAL_ROUNDS))

    def test_never_launches_anything(self):
        # This is machinery-only: the manifest generator must not import or
        # call anything from run_manifest.py's launch path (subprocess,
        # Popen, etc.) -- a static check that the module doesn't shell out.
        import inspect

        source = inspect.getsource(select_v2)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("Popen", source)


if __name__ == "__main__":
    unittest.main()
