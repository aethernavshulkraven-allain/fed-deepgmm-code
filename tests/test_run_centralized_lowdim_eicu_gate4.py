"""Gate 4 tests: centralized eICU baselines use the paper-aligned objective
and equal-client (hospital-aware) validation/test selection, not the legacy
pooled objective/selection used by the zoo baselines.

Runs a real (tiny) training loop against a hand-built two-hospital scenario
NPZ -- following the repo's existing "verify with a real run, not only
hand-derived unit tests" convention -- and checks the reported metrics
against an independently recomputed equal-client value from the actual saved
checkpoint, rather than only checking that fields exist.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

import run_centralized_lowdim as centralized  # noqa: E402
from experiment_utils import equal_client_structural_mse  # noqa: E402


def _write_two_hospital_scenario(path, n_per_client=(20, 4), seed=0):
    """Two clients of very different size, so equal-client and
    sample-weighted aggregation can genuinely diverge; g0 is a known linear
    function of [D, W] so structural MSE is directly checkable.
    """
    rng = np.random.default_rng(seed)
    payload = {"splits": ["train", "dev", "test"]}
    for split in ("train", "dev", "test"):
        rows = []
        client_ids = []
        for client, n in enumerate(n_per_client):
            d = rng.integers(0, 2, size=n).astype("float64")
            w = rng.normal(size=n)
            rows.append(np.column_stack([d, w]))
            client_ids.append(np.full(n, client, dtype="int64"))
        x = np.concatenate(rows, axis=0)
        client_id = np.concatenate(client_ids, axis=0)
        z = x.copy()  # instrument = treatment/covariate, adequate for a smoke run
        g_true = 1.0 * x[:, 0] + 0.5 * x[:, 1]
        y = g_true + rng.normal(scale=0.1, size=x.shape[0])
        payload[f"{split}_x"] = x
        payload[f"{split}_z"] = z
        payload[f"{split}_y"] = y.reshape(-1, 1)
        payload[f"{split}_g"] = g_true.reshape(-1, 1)
        payload[f"{split}_w"] = x
        payload[f"{split}_client_id"] = client_id
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)


class GateFourCentralizedEicuTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gate4_centralized_")
        self.data_dir = os.path.join(self.tmpdir, "data")
        self.scenario_name = "linear_scenario_seed0"
        _write_two_hospital_scenario(
            os.path.join(self.data_dir, "eicu_semisynth", f"{self.scenario_name}.npz")
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _args(self, **overrides):
        from types import SimpleNamespace

        defaults = dict(
            dataset="eicu_semisynth",
            scenario_name=self.scenario_name,
            method="gda",
            seed=0,
            output_dir=os.path.join(self.tmpdir, "out"),
            iterations=5,
            batch_size=None,
            g_lr=0.01,
            f_lr=0.05,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            data_dir=self.data_dir,
            run_id="test_run",
            gpu_id=None,
            device=None,
            no_cuda=True,
            log_test_mse_by_round=False,
            overwrite=True,
            objective_mode="paper_aligned",
            scenario_seed=0,
            seed_pair_id="confirmatory_01",
            campaign_role=None,
            scenario_checksum=None,
            protocol_version="eicu_study_a_v1",
            # Campaign-provenance identity fields. A real centralized
            # campaign row always supplies these via CLI (see
            # scripts/run_eicu_centralized_baselines.py's build_command and
            # scripts/run_eicu_study_a_v2_centralized.py, which pass
            # --role/--g0/--alignment-label/--scenario-scope/--study-claim
            # for every launched row) -- write_effective_config's provenance
            # guard (experiment_utils.py CAMPAIGN_PROVENANCE_FIELDS) requires
            # them whenever protocol_version is set on an eICU run, so this
            # fixture must supply realistic values too, the same way a real
            # campaign invocation would, rather than leaving them empty.
            role="centralized_baseline",
            g0="linear",
            alignment_label="centralized_reference",
            scenario_scope="demo",
            study_claim="semi_synthetic_benchmark_no_clinical_claim",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_legacy_objective_mode_is_rejected_for_eicu(self):
        args = self._args(objective_mode="legacy")
        with self.assertRaises(ValueError):
            centralized.run(args)

    def test_paper_aligned_run_completes_and_uses_paper_aligned_objective(self):
        args = self._args()
        output_dir = centralized.run(args)
        import json

        with open(os.path.join(output_dir, "effective_config.json")) as handle:
            config = json.load(handle)
        self.assertEqual(config["objective_mode"], "paper_aligned")
        self.assertEqual(config["objective"], "PaperAlignedMomentObjective")
        self.assertEqual(config["aggregation_weighting"], "none")
        self.assertEqual(config["scenario_seed"], 0)
        self.assertEqual(config["seed_pair_id"], "confirmatory_01")

    def test_metrics_use_equal_client_semantics_and_match_independent_recomputation(self):
        args = self._args()
        output_dir = centralized.run(args)
        import json
        import torch

        with open(os.path.join(output_dir, "metrics.json")) as handle:
            metrics = json.load(handle)

        self.assertIsNotNone(metrics["equal_client_test_mse_at_best_validation"])
        self.assertIsNotNone(metrics["sample_weighted_test_mse_at_best_validation"])
        # test_mse_at_best_validation is the required compatibility field --
        # for eICU it must carry equal-client semantics, not the pooled value.
        self.assertAlmostEqual(
            metrics["test_mse_at_best_validation"],
            metrics["equal_client_test_mse_at_best_validation"],
            places=10,
        )

        # Independently recompute equal-client test MSE from the actual saved
        # best-validation checkpoint, rather than trusting the same code path
        # that produced the reported metric.
        checkpoint = torch.load(
            os.path.join(output_dir, "checkpoints", "best_validation.pt"), map_location="cpu"
        )
        input_dim_g = 2
        g, _ = centralized.build_models(
            torch.device("cpu"), input_dim_g=input_dim_g, input_dim_f=input_dim_g,
            hidden_widths=centralized.EICU_HIDDEN_WIDTHS,
        )
        g.load_state_dict(checkpoint["g_state_dict"])
        g.eval()
        _, _, test, _ = centralized.load_pooled_splits(
            "eicu_semisynth", __import__("pathlib").Path(self.data_dir), torch.device("cpu"),
            scenario_name=self.scenario_name,
        )
        with torch.no_grad():
            prediction = g(test.x)
        recomputed_equal_client, _, per_client = equal_client_structural_mse(
            prediction, test.g, test.client_id
        )
        self.assertAlmostEqual(
            recomputed_equal_client, metrics["equal_client_test_mse_at_best_validation"], places=6
        )
        self.assertEqual(set(per_client.keys()), {0, 1})

    def test_per_client_metrics_csv_is_written(self):
        args = self._args()
        output_dir = centralized.run(args)
        csv_path = os.path.join(output_dir, "per_client_metrics.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, newline="") as handle:
            rows = list(__import__("csv").DictReader(handle))
        self.assertEqual({row["client_id"] for row in rows}, {"0", "1"})

    def test_sgda_method_is_supported_for_eicu(self):
        args = self._args(method="sgda", batch_size=8)
        output_dir = centralized.run(args)
        import json

        with open(os.path.join(output_dir, "metrics.json")) as handle:
            metrics = json.load(handle)
        self.assertIn("test_mse_at_best_validation", metrics)


if __name__ == "__main__":
    unittest.main()
