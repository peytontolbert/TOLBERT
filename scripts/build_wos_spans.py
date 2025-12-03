"""
Build JSONL spans and ontology metadata for a WOS-style hierarchical text dataset.

This script is a reference implementation of the ResearchHierarchy construction
described in the TOLBERT paper. It does **not** ship any WOS data; instead, it
expects you to provide a CSV file with one row per document and three levels of
labels.

Expected input CSV schema (you can adapt this to your variant as needed):

    text_col:   column containing the text to encode (e.g., abstract)
    l1_col:     top-level field (e.g., "Computer Science")
    l2_col:     second-level subfield
    l3_col:     third-level discipline

By default we assume:

    --text-col text
    --l1-col   level1
    --l2-col   level2
    --l3-col   level3

You can override these on the command line.

Outputs:
  - --spans-out:
      JSONL file with one record per document:
        {
          "span_id": "doc_000001",
          "text": "... abstract or text ...",
          "source_id": "doc_000001",
          "node_path": [root_id, l1_id, l2_id, l3_id],
          "meta": {
             "level1": "...",
             "level2": "...",
             "level3": "..."
          }
        }

  - --nodes-out (optional):
      JSONL file of ontology nodes with fields:
        node_id, level, type, parent_id, name

  - --level-sizes-out (optional):
      Small JSON helper:
        {"level_sizes": {1: num_l1, 2: num_l2, 3: num_l3}}

These files are compatible with `tolbert.data.TreeOfLifeDataset` and the
training skeleton in `scripts/train_tolbert.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DocRow:
    text: str
    l1: str
    l2: str
    l3: str


def load_wos_csv(
    path: Path,
    text_col: str,
    l1_col: str,
    l2_col: str,
    l3_col: str,
) -> List[DocRow]:
    rows: List[DocRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "") or ""
            l1 = row.get(l1_col, "") or ""
            l2 = row.get(l2_col, "") or ""
            l3 = row.get(l3_col, "") or ""
            if not text:
                continue
            rows.append(DocRow(text=text, l1=l1, l2=l2, l3=l3))
    return rows


def build_ontology(rows: List[DocRow]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build integer node IDs for:
      - level 1: unique l1 labels
      - level 2: unique (l1, l2) pairs
      - level 3: unique (l1, l2, l3) triples

    Node IDs are global; path uses:
      [root_id, l1_id, l2_id, l3_id]
    """
    next_id = 0
    root_id = next_id
    next_id += 1

    l1_ids: Dict[str, int] = {}
    l2_ids: Dict[str, int] = {}
    l3_ids: Dict[str, int] = {}

    for r in rows:
        if r.l1 and r.l1 not in l1_ids:
            l1_ids[r.l1] = next_id
            next_id += 1
        if r.l1 and r.l2:
            key2 = f"{r.l1}::{r.l2}"
            if key2 not in l2_ids:
                l2_ids[key2] = next_id
                next_id += 1
        if r.l1 and r.l2 and r.l3:
            key3 = f"{r.l1}::{r.l2}::{r.l3}"
            if key3 not in l3_ids:
                l3_ids[key3] = next_id
                next_id += 1

    return l1_ids, l2_ids, l3_ids


def write_nodes_jsonl(
    out_path: Path,
    l1_ids: Dict[str, int],
    l2_ids: Dict[str, int],
    l3_ids: Dict[str, int],
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        # Root
        f.write(
            json.dumps(
                {
                    "node_id": 0,
                    "level": 0,
                    "type": "root",
                    "parent_id": None,
                    "name": "Root",
                    "attributes": {},
                }
            )
            + "\n"
        )

        # Level 1 nodes
        for name, nid in l1_ids.items():
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 1,
                        "type": "field",
                        "parent_id": 0,
                        "name": name,
                        "attributes": {},
                    }
                )
                + "\n"
            )

        # Level 2 nodes
        for key, nid in l2_ids.items():
            l1, l2 = key.split("::", 1)
            parent_id = l1_ids[l1]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 2,
                        "type": "subfield",
                        "parent_id": parent_id,
                        "name": l2,
                        "attributes": {"level1": l1},
                    }
                )
                + "\n"
            )

        # Level 3 nodes
        for key, nid in l3_ids.items():
            l1, l2, l3 = key.split("::", 2)
            # Parent is the (l1,l2) node
            parent_id = l2_ids[f"{l1}::{l2}"]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 3,
                        "type": "discipline",
                        "parent_id": parent_id,
                        "name": l3,
                        "attributes": {"level1": l1, "level2": l2},
                    }
                )
                + "\n"
            )


def build_spans(
    rows: List[DocRow],
    l1_ids: Dict[str, int],
    l2_ids: Dict[str, int],
    l3_ids: Dict[str, int],
) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    root_id = 0
    for idx, r in enumerate(rows):
        span_id = f"doc_{idx:06d}"
        l1_id = l1_ids.get(r.l1)
        l2_id = None
        l3_id = None
        if r.l1 and r.l2:
            key2 = f"{r.l1}::{r.l2}"
            l2_id = l2_ids.get(key2)
        if r.l1 and r.l2 and r.l3:
            key3 = f"{r.l1}::{r.l2}::{r.l3}"
            l3_id = l3_ids.get(key3)

        # Build node_path; allow partially-labeled paths by omitting unknowns.
        path = [root_id]
        if l1_id is not None:
            path.append(l1_id)
        if l2_id is not None:
            path.append(l2_id)
        if l3_id is not None:
            path.append(l3_id)

        spans.append(
            {
                "span_id": span_id,
                "text": r.text,
                "source_id": span_id,
                "node_path": path,
                "meta": {
                    "level1": r.l1,
                    "level2": r.l2,
                    "level3": r.l3,
                },
            }
        )
    return spans


def write_spans_jsonl(spans: List[Dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in spans:
            f.write(json.dumps(rec) + "\n")


def write_level_sizes(
    l1_ids: Dict[str, int],
    l2_ids: Dict[str, int],
    l3_ids: Dict[str, int],
    out_path: Path,
) -> None:
    level_sizes = {
        1: len(l1_ids),
        2: len(l2_ids),
        3: len(l3_ids),
    }
    out = {"level_sizes": level_sizes}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build WOS-style spans JSONL and ontology metadata.")
    ap.add_argument("--input-csv", type=str, required=True, help="Input CSV file with text + labels.")
    ap.add_argument("--spans-out", type=str, required=True, help="Output spans JSONL file.")
    ap.add_argument("--nodes-out", type=str, default="", help="Optional nodes JSONL output.")
    ap.add_argument(
        "--level-sizes-out",
        type=str,
        default="",
        help="Optional JSON file with level_sizes helper dict.",
    )
    ap.add_argument("--text-col", type=str, default="text", help="Text column name.")
    ap.add_argument("--l1-col", type=str, default="level1", help="Level-1 label column name.")
    ap.add_argument("--l2-col", type=str, default="level2", help="Level-2 label column name.")
    ap.add_argument("--l3-col", type=str, default="level3", help="Level-3 label column name.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = load_wos_csv(
        csv_path,
        text_col=args.text_col,
        l1_col=args.l1_col,
        l2_col=args.l2_col,
        l3_col=args.l3_col,
    )

    l1_ids, l2_ids, l3_ids = build_ontology(rows)

    spans = build_spans(rows, l1_ids=l1_ids, l2_ids=l2_ids, l3_ids=l3_ids)

    spans_out = Path(args.spans_out)
    spans_out.parent.mkdir(parents=True, exist_ok=True)
    write_spans_jsonl(spans, spans_out)

    if args.nodes_out:
        nodes_out = Path(args.nodes_out)
        nodes_out.parent.mkdir(parents=True, exist_ok=True)
        write_nodes_jsonl(nodes_out, l1_ids=l1_ids, l2_ids=l2_ids, l3_ids=l3_ids)

    if args.level_sizes_out:
        ls_out = Path(args.level_sizes_out)
        ls_out.parent.mkdir(parents=True, exist_ok=True)
        write_level_sizes(l1_ids=l1_ids, l2_ids=l2_ids, l3_ids=l3_ids, out_path=ls_out)


if __name__ == "__main__":
    main()


