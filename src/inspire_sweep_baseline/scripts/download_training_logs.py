#!/usr/bin/env python3
"""Download sparse checkpoints and other logs through SSH/rsync.

Usage (from the workspace root):
    python3 src/inspire_sweep_baseline/scripts/download_training_logs.py --host USER@SERVER --remote-dir '~/logs' --dry-run
    Remove --dry-run to download. Always specify the host and remote directory.

Each checkpoint directory keeps iteration multiples of --interval plus the
--latest highest numbered .pt files (the union, without duplicates). Other
regular files are copied unchanged. Files starting with numerical_failure are
always included when present, including unnumbered .pt files. Other unnumbered
.pt files and symlinks are skipped.
The destination is logs/TEMP/YYYY-MM-DD/<remote directory name>, using Korea time.
Existing local files are never deleted; repeated downloads use rsync updates.
Requires local Python 3, ssh, rsync and remote Python 3, rsync.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
CHECKPOINT = re.compile(r".*?(\d+)\.pt")
REMOTE_INVENTORY = """
import json
import os
import sys

root = os.path.abspath(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(root):
    raise SystemExit('Remote directory does not exist: ' + root)
files = []
def fail(error):
    raise error
for directory, _, names in os.walk(root, onerror=fail, followlinks=False):
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.islink(path) and os.path.isfile(path):
            files.append(os.path.relpath(path, root))
print(json.dumps({'root': root, 'files': files}))
"""


def select_files(files: list[str], interval: int, latest: int):
    """Select independently per directory, ordering checkpoints by iteration."""
    selected = set()
    groups = defaultdict(list)
    skipped = []
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\0" in name:
            raise ValueError(f"Invalid remote relative path: {name!r}")
        if path.name.startswith("numerical_failure") or path.suffix != ".pt":
            selected.add(name)
            continue
        match = CHECKPOINT.fullmatch(path.name)
        if match is None:
            skipped.append(name)
            continue
        groups[str(path.parent)].append((int(match[1]), name))
    for directory, checkpoints in sorted(groups.items()):
        checkpoints.sort()
        keep = {name for iteration, name in checkpoints if iteration % interval == 0}
        if latest:
            keep.update(name for _, name in checkpoints[-latest:])
        selected.update(keep)
        print(f"{directory}: .pt {len(checkpoints)} -> {len(keep)}, "
              f"latest iteration={checkpoints[-1][0]}", flush=True)
    return sorted(selected), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="SSH user@host or SSH config alias")
    parser.add_argument("--remote-dir", required=True, help="Remote directory (quote ~/paths)")
    parser.add_argument("--dest-root", type=Path, default=WORKSPACE / "logs" / "TEMP")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--latest", type=int, default=5)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--dry-run", action="store_true", help="List selection without downloading")
    args = parser.parse_args()
    if args.interval <= 0 or args.latest < 0 or not 1 <= args.port <= 65535:
        parser.error("--interval must be positive, --latest nonnegative, --port 1..65535")
    if not re.fullmatch(r"(?:[\w.-]+@)?[\w.-]+", args.host):
        parser.error("--host must be a hostname or IPv4 address, optionally prefixed with user@")
    for program in ("ssh", "rsync"):
        if shutil.which(program) is None:
            parser.error(f"Required command not found: {program}")

    ssh = ["ssh", "-p", str(args.port)]
    remote_command = shlex.join(["python3", "-c", REMOTE_INVENTORY, args.remote_dir])
    result = subprocess.run([*ssh, args.host, remote_command],
                            stdout=subprocess.PIPE, check=True)
    inventory = json.loads(result.stdout)
    remote_root = inventory["root"]
    remote_name = PurePosixPath(remote_root).name
    if not remote_root.startswith("/") or not remote_name:
        raise ValueError(f"Expected a remote directory below /, got {remote_root!r}")
    selected, skipped = select_files(inventory["files"], args.interval, args.latest)
    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    destination = args.dest_root.expanduser().resolve() / today / remote_name
    print(f"Source: {args.host}:{remote_root}/", flush=True)
    print(f"Destination: {destination}/", flush=True)
    print(f"Selected: {len(selected)} files (including non-.pt logs)", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} .pt files without an iteration number:", flush=True)
        for name in skipped:
            print(f"  {name}", flush=True)
    if args.dry_run:
        for name in selected:
            print(name)
        return 0
    if not selected:
        print("Nothing to download.")
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "--archive", "--protect-args", "--human-readable",
               "--info=progress2", "--partial-dir=.rsync-partial", "--from0",
               "--files-from=-", "-e", shlex.join(ssh),
               f"{args.host}:{remote_root}/", str(destination) + "/"]
    # A NUL-delimited manifest preserves spaces/newlines and transfers only the
    # selected files. No recursive flag or deletion option is used.
    manifest = b"".join(name.encode("utf-8") + b"\0" for name in selected)
    subprocess.run(command, input=manifest, check=True)
    print(f"Download complete: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
