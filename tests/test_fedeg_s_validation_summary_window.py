"""Regression gate: the stability report must survive deterioration stopping being off.

`run_stochastic_eg_stability.summarize` feeds `validation_deterioration_window`
straight into `stability_summary`, but those are two different quantities: one
decides when to STOP a run, the other decides the rolling window used to REPORT
it. The FedEG-S validation stage disables stopping by setting that window to 0 --
the documented disable value -- which left the report asking for the median of an
empty slice and failed every run after a full 1500 rounds of training.
"""

import csv
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "fedgmm/sp_decentralized_mnist_lr_example"))

import run_fedeg_s_validation_20260922 as campaign      # noqa: E402
import run_stochastic_eg_stability as stability         # noqa: E402

ROUNDS = 300


def make_job(tmp_path, window):
    """A completed run: a full curve plus the metrics that must agree with it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    values = [0.30 - 0.0004*i + 0.01*((i % 7)/7.0) for i in range(ROUNDS)]
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "primary_val_mse"])
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({"round": index, "primary_val_mse": value})
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best_validation_mse": min(values)}) + "\n")
    row = {"run_id": "val_femnist_x_a0p1_x_ref_s31", "dataset": "femnist_x",
           "alpha": "0.1", "method": "fed_eg_s", "seed": "31", "candidate": "0"}
    # `comm_round` is present on every real job config -- `resolve_job` reads it
    # too -- and the summary needs it to judge full-horizon completion.
    config = {**campaign.NO_DETERIORATION_STOP,
              "validation_deterioration_window": window,
              "comm_round": ROUNDS}
    return types.SimpleNamespace(run_dir=run_dir, row=row, config=config), values


def test_campaign_summarizer_works_with_stopping_disabled(tmp_path):
    """The exact configuration the stage runs: stopping window 0."""
    job, values = make_job(tmp_path, window=0)
    assert job.config["validation_deterioration_window"] == 0
    result = campaign.summarize(job)
    assert result["stability_report_window"] == campaign.STABILITY_REPORT_WINDOW
    assert result["best_window_median"] > 0
    assert result["rounds"] == ROUNDS
    assert result["best"] == pytest.approx(min(values))
    assert result["completed_full_horizon"] is True
    assert isinstance(result["tail_stability_within_stability_study_thresholds"], bool)


def test_shared_summarizer_is_the_one_that_breaks(tmp_path):
    """Pin the defect being worked around, so a later fix upstream is noticed."""
    job, _ = make_job(tmp_path, window=0)
    with pytest.raises(Exception):
        stability.summarize(job)


def test_reporting_window_is_independent_of_the_stopping_window(tmp_path):
    """Changing the stopping window must not change the reported statistics."""
    baseline = campaign.summarize(make_job(tmp_path / "a", window=0)[0])
    other = campaign.summarize(make_job(tmp_path / "b", window=25)[0])
    for key in ("best_window_median", "tail_median", "tail_p90",
                "stability_report_window"):
        assert baseline[key] == pytest.approx(other[key]), key


def test_reporting_window_shrinks_to_fit_a_short_curve(tmp_path):
    """A curve shorter than the window must not raise."""
    job, _ = make_job(tmp_path, window=0)
    rows = list(csv.DictReader((job.run_dir / "mse_by_round.csv").open()))[:10]
    with (job.run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "primary_val_mse"])
        writer.writeheader()
        writer.writerows(rows)
    values = [float(r["primary_val_mse"]) for r in rows]
    (job.run_dir / "metrics.json").write_text(json.dumps(
        {"best_validation_mse": min(values)}) + "\n")
    result = campaign.summarize(job)
    assert result["stability_report_window"] == len(values)
    assert result["rounds"] == len(values)


def test_truncated_run_is_not_marked_full_horizon(tmp_path):
    """The design's eligibility rule turns on finishing the horizon, so a run
    that stopped short must not be recorded as having finished it."""
    job, _ = make_job(tmp_path, window=0)
    rows = list(csv.DictReader((job.run_dir / "mse_by_round.csv").open()))[:60]
    with (job.run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "primary_val_mse"])
        writer.writeheader()
        writer.writerows(rows)
    values = [float(r["primary_val_mse"]) for r in rows]
    (job.run_dir / "metrics.json").write_text(json.dumps(
        {"best_validation_mse": min(values)}) + "\n")
    result = campaign.summarize(job)
    assert result["rounds"] == 60 < ROUNDS
    assert result["completed_full_horizon"] is False


def test_tail_stability_is_not_named_as_eligibility(tmp_path):
    """Guard against the stability study's thresholds silently returning as a
    second, stricter eligibility filter."""
    job, _ = make_job(tmp_path, window=0)
    result = campaign.summarize(job)
    assert "selection_eligible" not in result
    assert "tail_stability_within_stability_study_thresholds" in result
