"""Generate separately named class-consistent CIFAR-10 X/XZ archives.

Companion to `generate_paired_emnist_data.py`, applying the same correction to
CIFAR-10. The originals are read-only inputs: this writes new archives under a
new root, verifies the originals are byte-identical afterwards, and refuses to
overwrite anything.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scenarios.paired_cifar10_scenario import PairedClassConsistentCIFAR10Scenario


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(source_root, output_root, datasets):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite generated data: {output_root}")
    if source_root == output_root:
        raise ValueError("Original and generated data roots must differ")
    evidence = {"seed": 527, "normalization": "original_train_y_shared_between_arms",
                "construction": "raw_y_class = g_class + (toy_y - toy_g)",
                "source_images": "reused unchanged; never redrawn",
                "datasets": {}}
    for dataset in datasets:
        source = source_root / dataset / "main.npz"
        source_hash = sha256(source)
        scenario = PairedClassConsistentCIFAR10Scenario(source)
        scenario.setup(num_train=scenario.sizes["train"],
                       num_dev=scenario.sizes["dev"], num_test=scenario.sizes["test"])
        destination = output_root / dataset / "main.npz"
        scenario.to_file(str(destination))
        unchanged = {}
        with np.load(source, allow_pickle=False) as a, \
                np.load(destination, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Source/generated archive schemas differ")
            for key in a.files:
                if not key.endswith("_y"):
                    unchanged[key] = bool(np.array_equal(a[key], b[key]))
                    if not unchanged[key]:
                        raise ValueError(f"Paired invariant changed: {dataset}/{key}")
        if sha256(source) != source_hash:
            raise RuntimeError(f"Original archive changed: {source}")
        evidence["datasets"][dataset] = {
            "source": str(source), "source_sha256": source_hash,
            "generated": str(destination), "generated_sha256": sha256(destination),
            "normalization_mean": scenario.mean, "normalization_std": scenario.std,
            "unchanged_arrays": unchanged, "splits": scenario.audit,
        }
        print(f"Generated and verified {destination}", flush=True)
    with (output_root / "generation_audit.json").open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["cifar10_x", "cifar10_xz"])
    args = parser.parse_args()
    generate(args.source_root, args.output_root, args.datasets)
