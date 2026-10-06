from types import SimpleNamespace
import unittest

import torch

from scripts.preflight_full_gradient import (
    FullGradientPreflightError,
    infer_batch_sample_count,
    validate_full_gradient_invariants,
)


def _mock_dataset(train_data_local_num_dict, train_data_local_dict):
    return [
        sum(train_data_local_num_dict.values()),
        0,
        0,
        None,
        None,
        None,
        train_data_local_num_dict,
        train_data_local_dict,
        {},
        {},
        1,
    ]


def _args(batch_size=0, client_num_in_total=2, client_num_per_round=2):
    return SimpleNamespace(
        dataset="abs",
        random_seed=0,
        partition_alpha=0.5,
        batch_size=batch_size,
        client_num_in_total=client_num_in_total,
        client_num_per_round=client_num_per_round,
    )


class FullGradientPreflightTest(unittest.TestCase):
    def test_infers_shared_batch_sample_count(self):
        batch = (
            torch.zeros(3, 1),
            torch.ones(3, 1),
            torch.arange(3).reshape(3, 1),
        )
        self.assertEqual(infer_batch_sample_count(batch), 3)

    def test_rejects_inconsistent_batch_field_lengths(self):
        batch = (torch.zeros(3, 1), torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            infer_batch_sample_count(batch)

    def test_accepts_one_full_batch_per_selected_client(self):
        dataset = _mock_dataset(
            {0: 2, 1: 3},
            {
                0: [(torch.zeros(2, 1), torch.zeros(2, 1))],
                1: [(torch.zeros(3, 1), torch.zeros(3, 1))],
            },
        )
        report = validate_full_gradient_invariants(
            _args(),
            dataset,
            {"batch_size": 0, "mode": "deterministic"},
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checked_client_count"], 2)
        self.assertEqual(report["sample_count_sum"], 5)

    def test_rejects_partial_client_batch(self):
        dataset = _mock_dataset(
            {0: 2, 1: 3},
            {
                0: [(torch.zeros(2, 1), torch.zeros(2, 1))],
                1: [(torch.zeros(2, 1), torch.zeros(2, 1))],
            },
        )
        with self.assertRaises(FullGradientPreflightError):
            validate_full_gradient_invariants(
                _args(),
                dataset,
                {"batch_size": 0, "mode": "deterministic"},
            )

    def test_rejects_partial_participation(self):
        dataset = _mock_dataset(
            {0: 2, 1: 3},
            {
                0: [(torch.zeros(2, 1), torch.zeros(2, 1))],
                1: [(torch.zeros(3, 1), torch.zeros(3, 1))],
            },
        )
        with self.assertRaises(FullGradientPreflightError):
            validate_full_gradient_invariants(
                _args(client_num_per_round=1),
                dataset,
                {"batch_size": 0, "mode": "deterministic"},
            )

    def test_rejects_effective_config_batch_size_drift(self):
        dataset = _mock_dataset(
            {0: 2, 1: 3},
            {
                0: [(torch.zeros(2, 1), torch.zeros(2, 1))],
                1: [(torch.zeros(3, 1), torch.zeros(3, 1))],
            },
        )
        with self.assertRaises(FullGradientPreflightError):
            validate_full_gradient_invariants(
                _args(),
                dataset,
                {"batch_size": 128, "mode": "deterministic"},
            )


if __name__ == "__main__":
    unittest.main()
