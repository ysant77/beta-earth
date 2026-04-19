"""Consolidate per-request parquet logs into monthly snapshots.

Reads all `requests/*.parquet` files from the `asterisk-labs/betaearth-requests`
HF dataset, groups them by `YYYY-MM` from the `timestamp` column, writes one
consolidated `data/YYYY-MM.parquet` per closed month, then deletes the
originals. The current (in-progress) month is skipped so live writes are not
disturbed.

Usage:
    HF_TOKEN=hf_xxx python scripts/consolidate_request_logs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "asterisk-labs/betaearth-requests"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List actions without uploading or deleting")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required", file=sys.stderr)
        return 1
    api = HfApi(token=token)

    # List all per-request files
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    request_files = [f for f in all_files if f.startswith("requests/") and f.endswith(".parquet")]
    print(f"Found {len(request_files)} per-request files")

    # Skip the current month — consolidate only closed months
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")

    # Group by YYYY-MM from filename prefix (filenames look like "requests/2026-04-19T06-57-19-354287Z.parquet")
    by_month: dict[str, list[str]] = defaultdict(list)
    for f in request_files:
        stem = Path(f).stem  # e.g. "2026-04-19T06-57-19-354287Z"
        if len(stem) < 7 or stem[4] != "-":
            print(f"WARN: skipping unexpected filename {f}")
            continue
        month = stem[:7]  # "YYYY-MM"
        if month == current_month:
            continue  # Don't touch the live month
        by_month[month].append(f)

    if not by_month:
        print("Nothing to consolidate (no closed months with pending files)")
        return 0

    for month in sorted(by_month):
        files = by_month[month]
        print(f"\n[{month}] consolidating {len(files)} files...")
        if args.dry_run:
            continue

        # Download + concat
        with tempfile.TemporaryDirectory() as workdir:
            workdir = Path(workdir)
            dfs = []
            for f in files:
                local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f, local_dir=workdir, token=token)
                dfs.append(pd.read_parquet(local))
            merged = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

            out_path = workdir / f"{month}.parquet"
            merged.to_parquet(out_path, index=False)
            print(f"[{month}] {len(merged)} rows → data/{month}.parquet")

            # Upload consolidated file
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=f"data/{month}.parquet",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Consolidate {month}: {len(merged)} requests from {len(files)} files",
            )

        # Delete originals (single commit per month for atomicity)
        api.delete_files(
            delete_patterns=files,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Remove {len(files)} per-request files merged into data/{month}.parquet",
        )
        print(f"[{month}] cleaned up {len(files)} per-request files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
