"""
Build ArXiv-CLS–style JSONL spans and simple ontology metadata.

This script is a small helper for the ArXiv-CLS variant described in the
TOLBERT paper. It mirrors the style of `build_wos_spans.py` / `build_codehierarchy_spans.py`,
but targets the 2-level arXiv category hierarchy.

The ArXiv-CLS setup in the paper uses:

  - Documents: recent arXiv CS papers (e.g., title + abstract text).
  - Labels: a two-level arXiv category taxonomy, e.g.:
        domain  = "cs"
        category = "cs.LG"

You are expected to provide a metadata file (CSV or JSONL) with at least:

  - doc_id:   unique paper identifier (e.g., arXiv ID)
  - domain:   top-level arXiv area (e.g., "cs", "math")
  - category: fine-grained arXiv category (e.g., "cs.LG", "cs.CL")
  - text:     text span to encode (e.g., title + abstract)

You can override these column names via CLI flags so you can adapt to
your particular preprocessing pipeline.

Outputs
=======

- --spans-out:
    JSONL file with one record per paper of the form:

      {
        "span_id": "arxiv-2101.00001",
        "text": "... title + abstract ...",
        "source_id": "arxiv-2101.00001",
        "node_path": [root_id, domain_id, category_id],
        "meta": {
          "doc_id": "...",
          "domain": "...",
          "category": "..."
        }
      }

- --nodes-out (optional):
    JSONL ontology file with fields:
      node_id, level, type, parent_id, name, attributes

- --level-sizes-out (optional):
    Small JSON helper:

      {"level_sizes": {1: num_domains, 2: num_categories}}

    which you can paste into a config (see `configs/wos_example.yaml` or
    `configs/codehierarchy_example.yaml` for reference).

The resulting `spans_out` file is directly consumable by
`tolbert.data.TreeOfLifeDataset` and the training skeleton in
`scripts/train_tolbert.py`, as long as your config's `level_sizes`
matches the ontology this script builds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ArxivRow:
    doc_id: str
    domain: str
    category: str
    text: str


def _row_to_meta(
    row: Dict[str, object],
    *,
    id_col: str,
    domain_col: str,
    category_col: str,
    text_col: str,
) -> Optional[ArxivRow]:
    try:
        doc_id = str(row[id_col])
        domain = str(row[domain_col])
        category = str(row[category_col])
        text = str(row[text_col])
    except KeyError as exc:
        raise KeyError(f"Missing required column {exc!s} in metadata record: {row}") from exc

    if not text or not text.strip():
        # Skip empty-text rows; they are not useful training examples.
        return None

    return ArxivRow(doc_id=doc_id, domain=domain, category=category, text=text)


def load_arxiv_metadata(
    path: Path,
    *,
    id_col: str,
    domain_col: str,
    category_col: str,
    text_col: str,
) -> List[ArxivRow]:
    """
    Load ArXiv-CLS metadata from CSV or JSON(L).
    """
    rows: List[ArxivRow] = []
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Allow records nested under "data" for compatibility with some pipelines.
                if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    obj = obj["data"]
                if not isinstance(obj, dict):
                    continue
                meta = _row_to_meta(
                    obj,
                    id_col=id_col,
                    domain_col=domain_col,
                    category_col=category_col,
                    text_col=text_col,
                )
                if meta is not None:
                    rows.append(meta)
        return rows

    # Default: CSV with header
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = _row_to_meta(
                row,
                id_col=id_col,
                domain_col=domain_col,
                category_col=category_col,
                text_col=text_col,
            )
            if meta is not None:
                rows.append(meta)
    return rows


def build_ontology(rows: List[ArxivRow]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build integer node IDs for:

      - level 1:  unique domains (e.g., "cs", "math")
      - level 2:  (domain, category) pairs (e.g., "cs::cs.LG")

    Level conventions:
      level 0: root (id 0)
      level 1: domain nodes
      level 2: category nodes (per domain)
    """
    next_id = 0
    root_id = next_id
    next_id += 1

    domain_ids: Dict[str, int] = {}
    category_ids: Dict[str, int] = {}

    for r in rows:
        if r.domain not in domain_ids:
            domain_ids[r.domain] = next_id
            next_id += 1

        cat_key = f"{r.domain}::{r.category}"
        if cat_key not in category_ids:
            category_ids[cat_key] = next_id
            next_id += 1

    # root_id is currently unused in the returned dicts but documented for clarity
    _ = root_id
    return domain_ids, category_ids


def write_nodes_jsonl(
    out_path: Path,
    domain_ids: Dict[str, int],
    category_ids: Dict[str, int],
) -> None:
    """
    Emit a minimal nodes.jsonl compatible with docs/tree_of_life.md.
    """
    with out_path.open("w", encoding="utf-8") as f:
        # Root (level 0)
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

        # Domains (level 1)
        for domain_name, nid in domain_ids.items():
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 1,
                        "type": "domain",
                        "parent_id": 0,
                        "name": domain_name,
                        "attributes": {},
                    }
                )
                + "\n"
            )

        # Categories (level 2)
        for key, nid in category_ids.items():
            domain_name, category_name = key.split("::", 1)
            parent_id = domain_ids[domain_name]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 2,
                        "type": "category",
                        "parent_id": parent_id,
                        "name": category_name,
                        "attributes": {"domain": domain_name},
                    }
                )
                + "\n"
            )


def build_spans(
    rows: List[ArxivRow],
    domain_ids: Dict[str, int],
    category_ids: Dict[str, int],
) -> List[Dict[str, object]]:
    """
    Build span records with node_path = [root_id, domain_id, category_id].
    """
    spans: List[Dict[str, object]] = []
    root_id = 0

    for r in rows:
        domain_id = domain_ids[r.domain]
        cat_key = f"{r.domain}::{r.category}"
        category_id = category_ids[cat_key]

        node_path = [root_id, domain_id, category_id]

        span_id = r.doc_id
        source_id = r.doc_id

        spans.append(
            {
                "span_id": span_id,
                "text": r.text,
                "source_id": source_id,
                "node_path": node_path,
                "meta": {
                    "doc_id": r.doc_id,
                    "domain": r.domain,
                    "category": r.category,
                },
            }
        )

    return spans


def write_spans_jsonl(spans: List[Dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in spans:
            f.write(json.dumps(rec) + "\n")


def write_level_sizes(
    domain_ids: Dict[str, int],
    category_ids: Dict[str, int],
    out_path: Path,
) -> None:
    """
    Write level_sizes helper as a dict[int, int] mapping level index
    (excluding the root) to number of classes at that level.

    For ArXiv-CLS we use:
      level 1: domains
      level 2: categories
    """
    level_sizes = {
        1: len(domain_ids),
        2: len(category_ids),
    }
    out = {"level_sizes": level_sizes}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build ArXiv-CLS spans JSONL and ontology metadata.",
    )
    ap.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="CSV or JSONL file with at least: doc_id, domain, category, text.",
    )
    ap.add_argument(
        "--spans-out",
        type=str,
        required=True,
        help="Output path for spans JSONL file.",
    )
    ap.add_argument(
        "--nodes-out",
        type=str,
        default="",
        help="Optional output path for nodes JSONL (ontology nodes).",
    )
    ap.add_argument(
        "--level-sizes-out",
        type=str,
        default="",
        help="Optional output path for level_sizes JSON file.",
    )

    # Column name overrides for flexibility.
    ap.add_argument(
        "--id-col",
        type=str,
        default="doc_id",
        help="Column name for document ID.",
    )
    ap.add_argument(
        "--domain-col",
        type=str,
        default="domain",
        help='Column name for top-level arXiv area (e.g., "cs").',
    )
    ap.add_argument(
        "--category-col",
        type=str,
        default="category",
        help='Column name for fine-grained arXiv category (e.g., "cs.LG").',
    )
    ap.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name containing the paper text (e.g., title + abstract).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    meta_path = Path(args.metadata_file)
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata_file does not exist or is not a file: {meta_path}")

    rows = load_arxiv_metadata(
        meta_path,
        id_col=args.id_col,
        domain_col=args.domain_col,
        category_col=args.category_col,
        text_col=args.text_col,
    )
    if not rows:
        raise RuntimeError(f"No valid ArXiv-CLS records were loaded from metadata file: {meta_path}")

    domain_ids, category_ids = build_ontology(rows)

    spans = build_spans(rows, domain_ids=domain_ids, category_ids=category_ids)

    spans_out = Path(args.spans_out)
    spans_out.parent.mkdir(parents=True, exist_ok=True)
    write_spans_jsonl(spans, spans_out)

    if args.nodes_out:
        nodes_out = Path(args.nodes_out)
        nodes_out.parent.mkdir(parents=True, exist_ok=True)
        write_nodes_jsonl(nodes_out, domain_ids, category_ids)

    if args.level_sizes_out:
        ls_out = Path(args.level_sizes_out)
        ls_out.parent.mkdir(parents=True, exist_ok=True)
        write_level_sizes(domain_ids, category_ids, ls_out)


if __name__ == "__main__":
    main()



