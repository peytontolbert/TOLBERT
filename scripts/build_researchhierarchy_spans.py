"""
Build ResearchHierarchy-style JSONL spans and simple ontology metadata for WOS / arXiv papers.

This is a reference implementation of the dataset construction described for the
ResearchHierarchy benchmark in the TOLBERT paper. It expects you to provide a local
metadata file that assigns each paper to a 3-level research taxonomy and (optionally)
its backing PDF path.

The script is intentionally lightweight and mirrors the structure of
`build_codehierarchy_spans.py`:

Input
=====
- --metadata_file:
    A CSV or JSONL file with, at minimum, the columns:
      doc_id, field, subfield, discipline, text

    Optionally it may also include:
      pdf_path: absolute or repo-relative path to the PDF for this paper
      source:  short tag like "wos" or "arxiv" identifying the corpus

    You can override these column names with CLI flags.

Outputs
=======
- --spans_out:
    JSONL file with one record per paper of the form:
      {
        "span_id": "WOS-12345",
        "text": "... title + abstract or full text ...",
        "source_id": "/papers/pdfs/2019/WOS-12345.pdf",
        "node_path": [root_id, field_id, subfield_id, discipline_id],
        "meta": {
          "field": "...",
          "subfield": "...",
          "discipline": "...",
          "doc_id": "...",
          "pdf_path": "...",        # if available
          "source": "wos" | "arxiv" # if available
        }
      }

- --nodes_out (optional):
    JSONL file describing ontology nodes with fields:
      node_id, level, type, parent_id, name, attributes

- --level_sizes_out (optional):
    Small JSON file with:
      {"level_sizes": [num_level0, num_level1, num_level2, num_level3]}
    which you can copy into your training config as `level_sizes`.

The intended use is:
  1) Prepare a single metadata file that combines your WOS and arXiv papers,
     mapped into a 3-level taxonomy (field → subfield → discipline).
  2) Point this script at it to obtain spans and ontology metadata compatible
     with the rest of the TOLBERT pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PaperMeta:
    doc_id: str
    field: str
    subfield: str
    discipline: str
    text: str
    pdf_path: Optional[str] = None
    source: Optional[str] = None


def load_metadata(
    path: Path,
    *,
    id_col: str,
    field_col: str,
    subfield_col: str,
    discipline_col: str,
    text_col: str,
    pdf_path_col: Optional[str],
    source_col: Optional[str],
) -> List[PaperMeta]:
    """
    Load WOS / arXiv-style paper metadata from CSV or JSON(L).

    The loader is column-name agnostic: you can override the column names via CLI.
    """
    metas: List[PaperMeta] = []

    def _row_to_meta(row: Dict[str, object]) -> Optional[PaperMeta]:
        try:
            doc_id = str(row[id_col])
            field = str(row[field_col])
            subfield = str(row[subfield_col])
            discipline = str(row[discipline_col])
            text = str(row[text_col])
        except KeyError as exc:
            raise KeyError(f"Missing required column {exc!s} in metadata record: {row}") from exc

        if not text.strip():
            # Skip empty-text entries; they are not useful training instances.
            return None

        pdf_path_val: Optional[str] = None
        if pdf_path_col and pdf_path_col in row and row[pdf_path_col] is not None:
            pdf_path_val = str(row[pdf_path_col])

        source_val: Optional[str] = None
        if source_col and source_col in row and row[source_col] is not None:
            source_val = str(row[source_col])

        return PaperMeta(
            doc_id=doc_id,
            field=field,
            subfield=subfield,
            discipline=discipline,
            text=text,
            pdf_path=pdf_path_val,
            source=source_val,
        )

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Accept either a flat object or nested under "data"
                if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    obj = obj["data"]
                if not isinstance(obj, dict):
                    continue
                meta = _row_to_meta(obj)
                if meta is not None:
                    metas.append(meta)
        return metas

    # Default: CSV with header.
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = _row_to_meta(row)
            if meta is not None:
                metas.append(meta)
    return metas


def build_ontology(
    metas: List[PaperMeta],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build simple integer ID mappings:
      - field -> field_node_id
      - (field, subfield) -> subfield_node_id
      - (field, subfield, discipline) -> discipline_node_id

    Level conventions:
      level 0: root (id 0)
      level 1: field nodes
      level 2: subfield nodes
      level 3: discipline nodes
    """
    next_id = 0
    root_id = next_id
    next_id += 1

    field_ids: Dict[str, int] = {}
    subfield_ids: Dict[str, int] = {}
    discipline_ids: Dict[str, int] = {}

    for m in metas:
        if m.field not in field_ids:
            field_ids[m.field] = next_id
            next_id += 1

        sub_key = f"{m.field}::{m.subfield}"
        if sub_key not in subfield_ids:
            subfield_ids[sub_key] = next_id
            next_id += 1

        disc_key = f"{m.field}::{m.subfield}::{m.discipline}"
        if disc_key not in discipline_ids:
            discipline_ids[disc_key] = next_id
            next_id += 1

    # Sanity check: at least one node at each level if any metas were provided.
    if metas and (not field_ids or not subfield_ids or not discipline_ids):
        raise RuntimeError("Failed to build a non-empty 3-level ontology from metadata.")

    # root_id is currently unused in the returned dicts but kept for documentation.
    _ = root_id
    return field_ids, subfield_ids, discipline_ids


def write_nodes_jsonl(
    out_path: Path,
    field_ids: Dict[str, int],
    subfield_ids: Dict[str, int],
    discipline_ids: Dict[str, int],
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

        # Fields (level 1)
        for field_name, nid in field_ids.items():
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 1,
                        "type": "field",
                        "parent_id": 0,
                        "name": field_name,
                        "attributes": {},
                    }
                )
                + "\n"
            )

        # Subfields (level 2)
        for key, nid in subfield_ids.items():
            field_name, subfield_name = key.split("::", 1)
            parent_id = field_ids[field_name]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 2,
                        "type": "subfield",
                        "parent_id": parent_id,
                        "name": subfield_name,
                        "attributes": {"field": field_name},
                    }
                )
                + "\n"
            )

        # Disciplines (level 3)
        for key, nid in discipline_ids.items():
            field_name, subfield_name, discipline_name = key.split("::", 2)
            sub_key = f"{field_name}::{subfield_name}"
            parent_id = subfield_ids[sub_key]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 3,
                        "type": "discipline",
                        "parent_id": parent_id,
                        "name": discipline_name,
                        "attributes": {
                            "field": field_name,
                            "subfield": subfield_name,
                        },
                    }
                )
                + "\n"
            )


def build_span_records(
    metas: List[PaperMeta],
    field_ids: Dict[str, int],
    subfield_ids: Dict[str, int],
    discipline_ids: Dict[str, int],
) -> List[Dict[str, object]]:
    """
    Iterate over papers and generate one span per paper.
    """
    root_id = 0
    spans: List[Dict[str, object]] = []

    for m in metas:
        field_id = field_ids[m.field]
        sub_key = f"{m.field}::{m.subfield}"
        sub_id = subfield_ids[sub_key]
        disc_key = f"{m.field}::{m.subfield}::{m.discipline}"
        disc_id = discipline_ids[disc_key]

        node_path = [root_id, field_id, sub_id, disc_id]

        span_id = m.doc_id
        source_id = m.pdf_path if m.pdf_path is not None else m.doc_id

        meta: Dict[str, object] = {
            "doc_id": m.doc_id,
            "field": m.field,
            "subfield": m.subfield,
            "discipline": m.discipline,
        }
        if m.pdf_path is not None:
            meta["pdf_path"] = m.pdf_path
        if m.source is not None:
            meta["source"] = m.source

        spans.append(
            {
                "span_id": span_id,
                "text": m.text,
                "source_id": source_id,
                "node_path": node_path,
                "meta": meta,
            }
        )

    return spans


def write_spans_jsonl(spans: List[Dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in spans:
            f.write(json.dumps(rec) + "\n")


def write_level_sizes(
    field_ids: Dict[str, int],
    subfield_ids: Dict[str, int],
    discipline_ids: Dict[str, int],
    out_path: Path,
) -> None:
    # We always have exactly one root.
    level_sizes = [
        1,  # root
        len(field_ids),
        len(subfield_ids),
        len(discipline_ids),
    ]
    out = {"level_sizes": level_sizes}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build ResearchHierarchy spans JSONL and ontology metadata from WOS / arXiv metadata."
        )
    )
    ap.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="CSV or JSONL file with at least: doc_id, field, subfield, discipline, text.",
    )
    ap.add_argument(
        "--spans_out",
        type=str,
        required=True,
        help="Output path for spans JSONL file.",
    )
    ap.add_argument(
        "--nodes_out",
        type=str,
        default="",
        help="Optional output path for nodes JSONL (ontology nodes).",
    )
    ap.add_argument(
        "--level_sizes_out",
        type=str,
        default="",
        help="Optional output path for level_sizes JSON file.",
    )

    # Column name overrides for flexibility.
    ap.add_argument("--id_col", type=str, default="doc_id", help="Column name for document ID.")
    ap.add_argument("--field_col", type=str, default="field", help="Column name for level-1 field.")
    ap.add_argument(
        "--subfield_col",
        type=str,
        default="subfield",
        help="Column name for level-2 subfield.",
    )
    ap.add_argument(
        "--discipline_col",
        type=str,
        default="discipline",
        help="Column name for level-3 discipline.",
    )
    ap.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Column name containing the paper text (e.g., title + abstract).",
    )
    ap.add_argument(
        "--pdf_path_col",
        type=str,
        default="pdf_path",
        help="Optional column name for the PDF path; if missing, source_id falls back to doc_id.",
    )
    ap.add_argument(
        "--source_col",
        type=str,
        default="source",
        help="Optional column name for corpus tag (e.g., 'wos', 'arxiv').",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    meta_path = Path(args.metadata_file)
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata_file does not exist or is not a file: {meta_path}")

    metas = load_metadata(
        meta_path,
        id_col=args.id_col,
        field_col=args.field_col,
        subfield_col=args.subfield_col,
        discipline_col=args.discipline_col,
        text_col=args.text_col,
        pdf_path_col=args.pdf_path_col or None,
        source_col=args.source_col or None,
    )
    if not metas:
        raise RuntimeError(f"No valid paper records were loaded from metadata file: {meta_path}")

    field_ids, subfield_ids, discipline_ids = build_ontology(metas)

    spans = build_span_records(
        metas=metas,
        field_ids=field_ids,
        subfield_ids=subfield_ids,
        discipline_ids=discipline_ids,
    )

    spans_out = Path(args.spans_out)
    spans_out.parent.mkdir(parents=True, exist_ok=True)
    write_spans_jsonl(spans, spans_out)

    if args.nodes_out:
        nodes_out = Path(args.nodes_out)
        nodes_out.parent.mkdir(parents=True, exist_ok=True)
        write_nodes_jsonl(nodes_out, field_ids, subfield_ids, discipline_ids)

    if args.level_sizes_out:
        ls_out = Path(args.level_sizes_out)
        ls_out.parent.mkdir(parents=True, exist_ok=True)
        write_level_sizes(field_ids, subfield_ids, discipline_ids, ls_out)


if __name__ == "__main__":
    main()


