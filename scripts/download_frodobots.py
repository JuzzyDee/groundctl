#!/usr/bin/env python3
"""
FrodoBots-2K Dataset Downloader
--------------------------------
Downloads and extracts each dataset part sequentially.
Uses parallel range requests to saturate high-speed connections.

- Resume-safe: skips already-completed downloads and extractions
- Parallel chunk downloading (configurable, default 16 connections)
- Progress bars for download, extraction, and overall
- Fetches the dataset manifest from S3 if no local CSV is provided

Usage:
    python download_frodobots.py                          # uses defaults
    python download_frodobots.py --dest /path/to/data     # custom output dir
    python download_frodobots.py --csv local-manifest.csv # use local CSV
    python download_frodobots.py --workers 32             # more connections

Install deps: pip install requests tqdm
"""

import argparse
import csv
import os
import sys
import tempfile
import time
import zipfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_CSV_URL  = "https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/complete-dataset.csv"
DEFAULT_DEST     = Path("./frodobots-2k")

PARALLEL_CHUNKS  = 16          # simultaneous connections per file — tune to taste
CHUNK_SIZE_BYTES = 8 * 1024 * 1024   # 8 MB read buffer per thread
REQUEST_TIMEOUT  = 30          # seconds before giving up on a stalled chunk

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_gb(value: str) -> float:
    return float(value.strip().replace("GB", ""))

def gb_str(gb: float) -> str:
    return f"{gb:.2f} GB"

def fetch_csv(url: str) -> Path:
    """Download the dataset manifest CSV to a temp file."""
    print(f"  Fetching dataset manifest from {url}...")
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    tmp.write_text(resp.text)
    print(f"  Manifest saved to {tmp}")
    return tmp

def load_dataset(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["compressed_gb"]   = parse_gb(row["compressed size"])
        row["uncompressed_gb"] = parse_gb(row["uncompressed size"])
        row["filename"]        = row["url"].split("/")[-1]
        row["stem"]            = row["filename"].replace(".zip", "")
    return rows

def is_extracted(row: dict, extracted_dir: Path) -> bool:
    folder = extracted_dir / row["stem"]
    return folder.exists() and any(folder.iterdir())

def get_remote_size(url: str) -> int | None:
    """HEAD request to get Content-Length."""
    try:
        r = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None

# ── Parallel download ─────────────────────────────────────────────────────────

def _download_chunk(url: str, start: int, end: int, dest: Path,
                    progress_bar: tqdm, lock: threading.Lock,
                    retries: int = 3) -> bool:
    """Download a byte range and write it directly into the correct offset."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, stream=True,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            with open(dest, "r+b") as f:
                f.seek(start)
                for data in resp.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                    if data:
                        f.write(data)
                        with lock:
                            progress_bar.update(len(data))
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n  Chunk {start}-{end} failed after {retries} attempts: {e}")
                return False
            time.sleep(2 ** attempt)
    return False


def download_parallel(url: str, dest: Path, expected_gb: float,
                      n_workers: int = PARALLEL_CHUNKS) -> bool:
    """
    Download url to dest using n_workers parallel range requests.
    Pre-allocates the file so all threads can write concurrently.
    """
    expected_bytes = int(expected_gb * 1024 ** 3)

    if dest.exists() and dest.stat().st_size >= expected_bytes * 0.999:
        print(f"  Already downloaded: {dest.name}")
        return True

    remote_size = get_remote_size(url)
    if not remote_size:
        print(f"  Couldn't get remote file size — aborting")
        return False

    # Pre-allocate file so threads can seek+write in parallel
    if not dest.exists() or dest.stat().st_size != remote_size:
        print(f"  Allocating {gb_str(remote_size / 1024**3)} on disk...")
        with open(dest, "wb") as f:
            f.seek(remote_size - 1)
            f.write(b"\0")

    chunk_size = remote_size // n_workers
    ranges = []
    for i in range(n_workers):
        start = i * chunk_size
        end   = (start + chunk_size - 1) if i < n_workers - 1 else remote_size - 1
        ranges.append((start, end))

    print(f"  Downloading with {n_workers} parallel connections...")
    lock   = threading.Lock()
    all_ok = True

    with tqdm(
        desc=f"  {dest.name}",
        total=remote_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        colour="cyan",
    ) as bar:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_download_chunk, url, s, e, dest, bar, lock): (s, e)
                for s, e in ranges
            }
            for future in as_completed(futures):
                if not future.result():
                    all_ok = False

    return all_ok

# ── Extraction ────────────────────────────────────────────────────────────────

def extract_file(zip_path: Path, dest_dir: Path, expected_gb: float) -> bool:
    expected_bytes = int(expected_gb * 1024 ** 3)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members    = zf.infolist()
            total_size = sum(m.file_size for m in members)

            with tqdm(
                desc=f"  Extract {zip_path.name}",
                total=total_size or expected_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                colour="green",
            ) as bar:
                for member in members:
                    zf.extract(member, dest_dir)
                    bar.update(member.file_size)
        return True
    except (zipfile.BadZipFile, Exception) as e:
        print(f"\n  Extraction failed: {e}")
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the FrodoBots-2K dataset (~900 GB compressed)"
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Output directory for downloaded and extracted data (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to local dataset manifest CSV (default: fetched from S3)"
    )
    parser.add_argument(
        "--workers", type=int, default=PARALLEL_CHUNKS,
        help=f"Parallel connections per file (default: {PARALLEL_CHUNKS})"
    )
    args = parser.parse_args()

    compressed_dir = args.dest / "compressed"
    extracted_dir  = args.dest / "data"
    compressed_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    if args.csv and args.csv.exists():
        csv_path = args.csv
    elif args.csv:
        print(f"ERROR: CSV not found at {args.csv}")
        sys.exit(1)
    else:
        csv_path = fetch_csv(DATASET_CSV_URL)

    rows = load_dataset(csv_path)
    n_workers = args.workers

    total_compressed_gb   = sum(r["compressed_gb"]   for r in rows)
    total_uncompressed_gb = sum(r["uncompressed_gb"] for r in rows)

    print("=" * 60)
    print("  FrodoBots-2K Dataset Downloader")
    print("=" * 60)
    print(f"  Parts              : {len(rows)}")
    print(f"  Total compressed   : {gb_str(total_compressed_gb)}")
    print(f"  Total extracted    : {gb_str(total_uncompressed_gb)}")
    print(f"  Parallel conns     : {n_workers} per file")
    print(f"  Download dir       : {compressed_dir}")
    print(f"  Extract dir        : {extracted_dir}")
    print("=" * 60)

    downloaded_gb = 0.0
    extracted_gb  = 0.0
    failed_parts  = []
    start_time    = time.time()

    for i, row in enumerate(rows):
        part      = row["part"]
        url       = row["url"]
        filename  = row["filename"]
        comp_gb   = row["compressed_gb"]
        uncomp_gb = row["uncompressed_gb"]
        zip_path  = compressed_dir / filename

        print(f"\n{'─' * 60}")
        print(f"  Part {part:>2} / {len(rows)-1}  |  {filename}")
        print(f"  {gb_str(comp_gb)} compressed -> {gb_str(uncomp_gb)} extracted")

        elapsed = time.time() - start_time
        speed   = downloaded_gb / (elapsed / 3600) if elapsed > 60 else 0
        eta_hr  = (total_compressed_gb - downloaded_gb) / speed if speed > 0 else 0
        pct     = downloaded_gb / total_compressed_gb * 100 if total_compressed_gb else 0
        print(
            f"  Overall: {gb_str(downloaded_gb)} / {gb_str(total_compressed_gb)} ({pct:.1f}%)"
            + (f"  |  {speed:.1f} GB/hr  |  ETA ~{eta_hr:.1f}h" if speed > 0 else "")
        )
        print(f"{'─' * 60}")

        if is_extracted(row, extracted_dir):
            print(f"  Already extracted — skipping")
            downloaded_gb += comp_gb
            extracted_gb  += uncomp_gb
            continue

        ok = download_parallel(url, zip_path, comp_gb, n_workers)
        if not ok:
            failed_parts.append(part)
            continue

        downloaded_gb += comp_gb

        ok = extract_file(zip_path, extracted_dir, uncomp_gb)
        if not ok:
            failed_parts.append(part)
            continue

        extracted_gb += uncomp_gb
        print(f"  Part {part} complete")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Done!  {gb_str(extracted_gb)} extracted in {elapsed/3600:.1f}h")
    if failed_parts:
        print(f"  Failed parts: {failed_parts}  — re-run to resume")
    else:
        print(f"  All {len(rows)} parts complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
