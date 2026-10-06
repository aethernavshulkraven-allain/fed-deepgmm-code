"""Label every existing result folder's aggregation-weighting provenance.

Before this fix, ``FedAvgAPI._aggregate``/``_aggregate_reg`` unconditionally used
sample-size-weighted (ordinary FedAvg) aggregation, contradicting the paper's
equal-client federated objective ``U = (1/N) sum_i U^i``. This script does not
rerun anything -- it only reads existing ``effective_config.json`` files and
labels each run:

* ``legacy_sample_weighted`` -- no ``aggregation_weighting`` key at all, i.e.
  the run predates this fix and used the old, unconditional, undocumented
  sample-size weighting. This is the label for every run in the repo today.
* ``sample_size_explicit`` -- the key is present and set to ``sample_size``:
  a post-fix run that explicitly chose legacy weighting (e.g. to keep a
  tuning campaign comparable to its own earlier runs).
* ``uniform_clients`` -- the key is present and set to ``uniform_clients``:
  paper-aligned.
* ``unrecognized_value`` -- the key is present but not one of the two known
  choices (should not happen given the validation added alongside this fix;
  flagged defensively rather than silently mislabeled).
* ``no_effective_config`` -- ``metrics.json`` exists but its sibling
  ``effective_config.json`` does not, so provenance cannot be determined at all.

Usage:
    python scripts/audit_aggregation_weighting.py
    python scripts/audit_aggregation_weighting.py --results-dir results --out experiments/aggregation_weighting_audit
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGACY_SAMPLE_WEIGHTED = "legacy_sample_weighted"
SAMPLE_SIZE_EXPLICIT = "sample_size_explicit"
UNIFORM_CLIENTS = "uniform_clients"
UNRECOGNIZED_VALUE = "unrecognized_value"
NO_EFFECTIVE_CONFIG = "no_effective_config"

KNOWN_CHOICES = ("uniform_clients", "sample_size")


def load_json(path):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def label_run(config):
    """Classify one run's aggregation-weighting provenance from its config."""
    if config is None:
        return NO_EFFECTIVE_CONFIG, None

    if "aggregation_weighting" not in config:
        return LEGACY_SAMPLE_WEIGHTED, None

    value = config["aggregation_weighting"]
    if value == "sample_size":
        return SAMPLE_SIZE_EXPLICIT, value
    if value == "uniform_clients":
        return UNIFORM_CLIENTS, value
    return UNRECOGNIZED_VALUE, value


def scan_runs(results_dir):
    """One row per ``metrics.json`` found under ``results_dir``.

    Mirrors ``consolidate_results.py``'s ``scan_runs`` traversal
    (``results_dir.rglob("metrics.json")`` paired with the sibling
    ``effective_config.json``) so the two audits stay consistent about what
    counts as "a run".
    """
    rows = []
    for dirpath, _dirnames, filenames in os.walk(results_dir):
        if "metrics.json" not in filenames:
            continue
        metrics_path = os.path.join(dirpath, "metrics.json")
        config_path = os.path.join(dirpath, "effective_config.json")
        config = load_json(config_path) if os.path.exists(config_path) else None
        label, raw_value = label_run(config)

        rel_parts = os.path.relpath(dirpath, results_dir).split(os.sep)
        config = config or {}
        rows.append(
            {
                "family": rel_parts[0] if rel_parts else "",
                "path": os.path.relpath(dirpath, REPO_ROOT),
                "dataset": str(config.get("dataset", "unknown")),
                "variant": str(config.get("variant", config.get("method", "unknown"))),
                "seed": config.get("random_seed", config.get("seed", "")),
                "run_id": str(config.get("run_id", os.path.basename(dirpath))),
                "aggregation_weighting_label": label,
                "aggregation_weighting_raw_value": (
                    "" if raw_value is None else raw_value
                ),
            }
        )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "family",
        "path",
        "dataset",
        "variant",
        "seed",
        "run_id",
        "aggregation_weighting_label",
        "aggregation_weighting_raw_value",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize(rows):
    counts = {}
    by_family = {}
    for row in rows:
        label = row["aggregation_weighting_label"]
        counts[label] = counts.get(label, 0) + 1
        family = row["family"]
        by_family.setdefault(family, {}).setdefault(label, 0)
        by_family[family][label] += 1
    return counts, by_family


def render_report(rows, counts, by_family):
    lines = []
    add = lines.append

    add("# Aggregation-weighting provenance audit\n")
    add(
        "This is a **labeling pass only** -- nothing was rerun, retrained, or "
        "modified. It exists to make explicit which existing results used the "
        "pre-fix, unconditional sample-size-weighted aggregation before any "
        "scaled training resumes.\n"
    )
    add(f"Total runs discovered: **{len(rows)}**\n")

    add("## Counts by label\n")
    add("| label | count | meaning |")
    add("|---|---|---|")
    meanings = {
        LEGACY_SAMPLE_WEIGHTED: "predates this fix; used the old, unconditional sample-size weighting",
        SAMPLE_SIZE_EXPLICIT: "post-fix run that explicitly chose legacy weighting",
        UNIFORM_CLIENTS: "paper-aligned equal-client weighting",
        UNRECOGNIZED_VALUE: "unexpected value -- investigate",
        NO_EFFECTIVE_CONFIG: "metrics.json present but effective_config.json missing",
    }
    for label in sorted(counts, key=lambda l: -counts[l]):
        add(f"| `{label}` | {counts[label]} | {meanings.get(label, '')} |")
    add("")

    if counts.get(UNRECOGNIZED_VALUE):
        add(
            f"**{counts[UNRECOGNIZED_VALUE]} run(s) have an unrecognized "
            "`aggregation_weighting` value.** This should not happen given the "
            "validation added alongside this fix; investigate before treating "
            "any of these as trustworthy.\n"
        )

    add("## By result family\n")
    add("| family | " + " | ".join(sorted(meanings)) + " |")
    add("|---|" + "---|" * len(meanings))
    for family in sorted(by_family):
        row_counts = by_family[family]
        cells = [str(row_counts.get(label, 0)) for label in sorted(meanings)]
        add(f"| `{family}` | " + " | ".join(cells) + " |")
    add("")

    add("## Consequence for scaled training\n")
    add(
        f"Every run labeled `{LEGACY_SAMPLE_WEIGHTED}` was produced under "
        "sample-size-weighted aggregation, silently, before this option existed. "
        "**None of the old experiment matrix has been automatically rerun.** "
        "Whether to rerun any of it under `uniform_clients` is a separate, "
        "deliberate decision -- this audit only establishes the current "
        "provenance so that decision can be made with full information."
    )
    add("")

    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir", default=os.path.join(REPO_ROOT, "results")
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "experiments", "aggregation_weighting_audit"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    rows = scan_runs(args.results_dir)
    counts, by_family = summarize(rows)
    report = render_report(rows, counts, by_family)

    csv_path = write_csv(os.path.join(args.out, "aggregation_weighting_audit.csv"), rows)

    summary_path = os.path.join(args.out, "aggregation_weighting_audit_summary.json")
    os.makedirs(args.out, exist_ok=True)
    with open(summary_path, "w") as handle:
        json.dump(
            {"total_runs": len(rows), "counts": counts, "by_family": by_family},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    report_path = os.path.join(args.out, "README.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    print(report)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
