"""verify_highdim_bn_diagnostic_certification_20260822.py must actually
recompute and compare, not just read a status field: a hand-edited
certification claiming certification_status=passed must still fail if a
fresh recomputation from the same underlying inputs disagrees.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_highdim_bn_diagnostic_certification_20260822 as verifier  # noqa: E402

CAMPAIGN = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
CERTIFICATION = CAMPAIGN / "bn_buffer_diagnostic_certification.json"
MANIFEST = CAMPAIGN / "bn_buffer_diagnostic_manifest.csv"
LAUNCHER_RESULTS = CAMPAIGN / "bn_buffer_diagnostic_launcher_results.json"
LAUNCH_HASHES = CAMPAIGN / "diagnostic_launch_hashes.json"

_SKIP_REASON = (
    "These 3 tests assert that the live repo tree's CORE_SOURCES hashes still "
    "match psi_adjudication_20260822_v3's frozen diagnostic_launch_hashes.json. "
    "As of the 2026-08-26 closeout pass (structured pretraining-failure "
    "evidence -- see PROTOCOL_DECISION_ADDENDUM_20260826.md), main.py and "
    "other CORE_SOURCES files were intentionally edited, which correctly makes "
    "the frozen v3 hash record stale -- that is the hard-stop working as "
    "designed (HIGH_DIM_DETERMINISTIC_CLOSEOUT_PLAN_20260826.md SS11), not a "
    "regression. The v3 diagnostic is explicitly superseded: closeout plan "
    "SS4.6 rewires V4 to a fresh post-hash diagnostic in a new namespace, "
    "which will get its own internally-consistent hash freeze. Re-freezing "
    "v3's hashes to match new source would misrepresent what v3's already-"
    "completed runs actually executed against, so that is not done here. "
    "These tests couple verifier-correctness to a specific historical "
    "artifact rather than a hermetic fixture; if verify_highdim_bn_diagnostic_"
    "certification_20260822.py's tamper-detection logic needs test coverage "
    "again, rewrite this against a self-contained fixture instead of the real "
    "v3 campaign directory."
)


@unittest.skip(_SKIP_REASON)
class VerifierTest(unittest.TestCase):
    def test_verifies_the_real_current_certification(self):
        result = verifier.verify(CERTIFICATION, MANIFEST, LAUNCHER_RESULTS, LAUNCH_HASHES)
        self.assertTrue(result["verified"])

    def test_tampered_status_field_alone_is_not_believed(self):
        original = CERTIFICATION.read_text()
        try:
            tampered = json.loads(original)
            tampered["minimum_f_bn_running_variance"] = -1.0  # a fabricated, better-looking number
            CERTIFICATION.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "does not match a fresh recomputation"):
                verifier.verify(CERTIFICATION, MANIFEST, LAUNCHER_RESULTS, LAUNCH_HASHES)
        finally:
            CERTIFICATION.write_text(original)

    def test_never_overwrites_the_certification_file(self):
        before = CERTIFICATION.read_text()
        verifier.verify(CERTIFICATION, MANIFEST, LAUNCHER_RESULTS, LAUNCH_HASHES)
        self.assertEqual(CERTIFICATION.read_text(), before)


if __name__ == "__main__":
    unittest.main()
