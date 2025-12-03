"""
Lightweight PDF preprocessing script.

Reads PDFs from /arxiv/pdfs/{year}/, extracts structured tokens via P0's tokenizer,
and writes JSONL chunks (section/equation/figure/table/text) under exports/pdfs_structured/.

Usage:
  # Specific year range (inclusive)
  PYTHONPATH=.. python -m models.scripts.preprocess_pdfs --years 2018 2020 --max-files 1000

  # All years and all files (may be large)
  PYTHONPATH=.. python -m models.scripts.preprocess_pdfs --max-files 0

  # Use a VLM (e.g., Qwen/Qwen3-VL-2B-Instruct) for OCR
  PYTHONPATH=.. python -m models.scripts.preprocess_pdfs --max-files 0 --qwen-model Qwen/Qwen3-VL-2B-Instruct

Supports resumable runs: existing shards in the output directory are scanned and
their pdf_path entries are skipped, so re-running continues where it left off.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import time

from models.tier3_pdf.pdf_tokenization import PDFTokenizationModel


def iter_pdf_paths(year_start: int | None, year_end: int | None, limit: int | None) -> List[Path]:
    paths: List[Path] = []
    base = Path("/arxiv/pdfs")
    years = []
    if year_start and year_end:
        years = list(range(year_start, year_end + 1))
    else:
        years = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
    for y in years:
        for p in (base / str(y)).glob("*.pdf"):
            paths.append(p)
            if limit and len(paths) >= limit:
                return paths
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, help="Start/end year inclusive")
    ap.add_argument("--max-files", type=int, default=1000, help="Max PDFs to process (0 = all)")
    ap.add_argument("--out-dir", type=str, default="exports/pdfs_structured", help="Output directory for JSONL shards")
    ap.add_argument("--shard-size", type=int, default=1000, help="Records per JSONL shard")
    ap.add_argument("--progress-every", type=int, default=100, help="Print status every N PDFs")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip PDFs already present in existing shards")
    ap.add_argument("--qwen-model", type=str, default=None, help="Optional Qwen VLM model name for OCR (e.g., Qwen/Qwen3-VL-2B-Instruct)")
    ap.add_argument("--max-pages", type=int, default=3, help="Max pages to parse per PDF for structured tokens")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR (faster)")
    ap.add_argument("--no-clip", action="store_true", help="Disable CLIP embeddings (faster)")
    args = ap.parse_args()

    tok = PDFTokenizationModel()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_start = args.years[0] if args.years else None
    year_end = args.years[1] if args.years else None
    limit = None if args.max_files is not None and args.max_files <= 0 else args.max_files
    paths = iter_pdf_paths(year_start, year_end, limit)

    # Resume support: gather processed pdf_paths from existing shards.
    processed = set()
    shard_idx = 0
    if args.resume:
        existing = sorted(out_dir.glob("pdf_structured_*.jsonl"))
        shard_idx = len(existing)
        for shard_file in existing:
            try:
                with shard_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and obj.get("pdf_path"):
                                processed.add(obj["pdf_path"])
                        except Exception:
                            continue
            except Exception:
                continue

    shard = []
    start_time = time.time()
    total = len(paths)
    for idx, pdf_path in enumerate(paths):
        if str(pdf_path) in processed:
            if (idx + 1) % args.progress_every == 0:
                elapsed = time.time() - start_time
                print(f"[resume] skipped processed {idx + 1}/{total} (elapsed {elapsed:.1f}s)")
            continue
        try:
            tokens = tok.tokenize(
                str(pdf_path),
                vlm_model=args.qwen_model,
                max_pages=args.max_pages,
                use_ocr=not args.no_ocr,
                use_clip=not args.no_clip,
            )
        except Exception as exc:
            print(f"[warn] failed to tokenize {pdf_path}: {exc}")
            continue
        shard.append({"pdf_path": str(pdf_path), "tokens": tokens})
        if len(shard) >= args.shard_size:
            out_path = out_dir / f"pdf_structured_{shard_idx:05d}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in shard:
                    f.write(json.dumps(rec) + "\n")
            print(f"[write] shard {shard_idx} ({len(shard)} recs)")
            shard = []
            shard_idx += 1
        if (idx + 1) % args.progress_every == 0:
            elapsed = time.time() - start_time
            print(f"[status] processed {idx + 1}/{total} (elapsed {elapsed:.1f}s)")
    if shard:
        out_path = out_dir / f"pdf_structured_{shard_idx:05d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in shard:
                f.write(json.dumps(rec) + "\n")
        print(f"[write] shard {shard_idx} ({len(shard)} recs)")

    elapsed = time.time() - start_time
    print(f"[done] wrote shards to {out_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
