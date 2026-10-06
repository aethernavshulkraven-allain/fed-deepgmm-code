"""Report whether a campaign run currently holds its lock.

Exit 0 when a run is active, 1 when none is. The campaign's own advisory lock is
the authoritative signal: a process-name match is not, because the launcher
`cd`s into the frozen runtime directory and execs the bare script name, so the
full path never appears in any command line.

The probe only tries the lock and releases it. It never blocks and never runs
anything itself.
"""

import fcntl
import sys
from pathlib import Path

if __name__ == "__main__":
    lock_path = Path(sys.argv[1]).resolve() / "campaign.lock"
    if not lock_path.exists():
        print("inactive (no lock file)")
        sys.exit(1)
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            print("active")
            sys.exit(0)
        fcntl.flock(handle, fcntl.LOCK_UN)
    print("inactive")
    sys.exit(1)
