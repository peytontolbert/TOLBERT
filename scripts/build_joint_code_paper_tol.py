"""
Build a joint code+paper Tree-of-Life (ToL) from existing per-domain spans.

This script is the missing glue described in the paper/docs: it takes
separate code and paper span files (each with its own local ontology and
`node_path` ids) and produces:

- A single, unified `nodes.jsonl` Tree-of-Life with:
    level 0: root
    level 1: domain nodes (e.g., "Code", "Papers")
    level 2+: remapped nodes from the original per-domain ontologies
- Rewritten span files for code and papers where `node_path` has been
  updated to refer to the *joint* node ID space.
- A `level_sizes` JSON helper compatible with `TOLBERTConfig.level_sizes`.

Design notes
============

This script deliberately **does not attempt to infer a semantic mapping**
between code categories (e.g., "Web", "ML", "Systems") and paper fields
("Computer Science", "Physics", ...). Doing that well would require
domain-specific heuristics or manual curation.

Instead, it:

- Adds explicit top-level domain nodes (by default: `"Code"` and `"Papers"`)
  at level 1 under a single shared root.
- Shifts each original ontology **one level deeper** in the tree:
    - Original root (id 0) is discarded; we introduce a fresh shared root.
    - Original level-1 nodes become level-2 under their domain node.
    - Original level-2 nodes become level-3, and so on.
- Remaps all node IDs into a **single global ID space** while preserving
  parent/child relations within each domain.

This gives you:

- A concrete, unified Tree-of-Life suitable for pretraining a single
  TOLBERT model across both domains.
- A starting point for more sophisticated, semantics-aware unification
  (e.g., manually or automatically clustering level-2 nodes across domains).

Inputs
======

- --code_spans:
    Spans JSONL for the code side (e.g., from `build_codehierarchy_spans.py`),
    with records of the form:
        {"span_id": "...", "text": "...", "source_id": "...", "node_path": [...], ...}

- --code_nodes:
    Ontology `nodes.jsonl` corresponding to `code_spans`. Expected fields:
        node_id, level, type, parent_id, name, attributes

- --paper_spans:
    Spans JSONL for the paper side (e.g., from `build_wos_spans.py` or
    `build_researchhierarchy_spans.py`).

- --paper_nodes:
    Ontology `nodes.jsonl` corresponding to `paper_spans`.

Outputs
=======

- --out_code_spans:
    Code spans JSONL with updated, joint `node_path`.

- --out_paper_spans:
    Paper spans JSONL with updated, joint `node_path`.

- --out_nodes:
    Joint ontology `nodes.jsonl` with a single root and shared ID space.

- --out_level_sizes:
    Small JSON file with:
        {"level_sizes": {1: num_level1, 2: num_level2, ...}}
    which you can plug into your TOLBERT training config.

Usage (example)
===============

1) Build per-domain datasets:

    python scripts/build_codehierarchy_spans.py \
        --repos_root /path/to/repos \
        --metadata_file /path/to/code_meta.csv \
        --spans_out data/code/spans_train.jsonl \
        --nodes_out data/code/nodes.jsonl

    python scripts/build_wos_spans.py \
        --input-csv /path/to/wos_train.csv \
        --spans-out data/wos/spans_train.jsonl \
        --nodes-out data/wos/nodes.jsonl

2) Join into a single Tree-of-Life:

    python scripts/build_joint_code_paper_tol.py \
        --code_spans data/code/spans_train.jsonl \
        --code_nodes data/code/nodes.jsonl \
        --paper_spans data/wos/spans_train.jsonl \
        --paper_nodes data/wos/nodes.jsonl \
        --out_code_spans data/joint/code_spans_train.jsonl \
        --out_paper_spans data/joint/paper_spans_train.jsonl \
        --out_nodes data/joint/nodes.jsonl \
        --out_level_sizes data/joint/level_sizes.json

You can then train TOLBERT on the union of `out_code_spans` and
`out_paper_spans` (e.g., via the `spans_files` list in your config) with
`level_sizes` taken from `out_level_sizes`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class NodeRecord:
    node_id: int
    level: int
    type: str
    parent_id: int | None
    name: str
    attributes: Dict[str, object]


def _load_nodes(path: Path) -> Dict[int, NodeRecord]:
    nodes: Dict[int, NodeRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            nid = int(obj["node_id"])
            level = int(obj.get("level", 0))
            node_type = str(obj.get("type", "node"))
            parent_id = obj.get("parent_id")
            if parent_id is not None:
                parent_id = int(parent_id)
            name = str(obj.get("name", f"node_{nid}"))
            attributes = obj.get("attributes") or {}
            nodes[nid] = NodeRecord(
                node_id=nid,
                level=level,
                type=node_type,
                parent_id=parent_id,
                name=name,
                attributes=attributes,
            )
    return nodes


def _iter_spans(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_spans(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _build_joint_ontology(
    *,
    code_nodes: Dict[int, NodeRecord],
    paper_nodes: Dict[int, NodeRecord],
    code_domain_name: str = "Code",
    paper_domain_name: str = "Papers",
) -> Tuple[
    Dict[int, NodeRecord],
    Dict[int, int],
    Dict[int, int],
    Dict[int, int],
]:
    """
    Construct a joint ontology and return:

    - joint_nodes: dict[new_id -> NodeRecord]
    - code_id_map: mapping old_code_node_id -> new_global_node_id
    - paper_id_map: mapping old_paper_node_id -> new_global_node_id
    - level_counts: mapping level -> number of nodes at that level (excluding root)

    Strategy:
      - Create a fresh root (id 0).
      - Create two domain nodes at level 1: Code, Papers.
      - For each original ontology:
          * Drop its root node (level 0).
          * Shift all other nodes one level deeper (level' = level + 1).
          * Rewire parents so that:
              - old parent == 0  → new parent = corresponding domain node id
              - otherwise        → new parent = mapped parent id
    """
    joint_nodes: Dict[int, NodeRecord] = {}
    level_counts: Dict[int, int] = {}

    # Root (level 0)
    root_id = 0
    joint_nodes[root_id] = NodeRecord(
        node_id=root_id,
        level=0,
        type="root",
        parent_id=None,
        name="Root",
        attributes={},
    )

    next_id = 1

    # Domain nodes at level 1
    code_domain_id = next_id
    next_id += 1
    paper_domain_id = next_id
    next_id += 1

    joint_nodes[code_domain_id] = NodeRecord(
        node_id=code_domain_id,
        level=1,
        type="domain",
        parent_id=root_id,
        name=code_domain_name,
        attributes={"source": "code"},
    )
    joint_nodes[paper_domain_id] = NodeRecord(
        node_id=paper_domain_id,
        level=1,
        type="domain",
        parent_id=root_id,
        name=paper_domain_name,
        attributes={"source": "paper"},
    )
    level_counts[1] = 2

    code_id_map: Dict[int, int] = {}
    paper_id_map: Dict[int, int] = {}

    def _remap_domain(
        *,
        src_nodes: Dict[int, NodeRecord],
        domain_node_id: int,
        is_code: bool,
    ) -> Dict[int, int]:
        nonlocal next_id
        id_map: Dict[int, int] = {}

        # Sort by (level, node_id) for stable assignment.
        for old_id, rec in sorted(src_nodes.items(), key=lambda kv: (kv[1].level, kv[0])):
            if rec.level == 0:
                # Skip local root; we replace it with shared root + domain nodes.
                continue

            new_level = rec.level + 1  # shift deeper by one level
            new_id = next_id
            next_id += 1

            # Determine parent in joint space.
            if rec.parent_id is None or rec.parent_id == 0:
                new_parent = domain_node_id
            else:
                if rec.parent_id not in id_map:
                    raise ValueError(
                        f"Parent node_id {rec.parent_id} for node {old_id} has not been remapped yet."
                    )
                new_parent = id_map[rec.parent_id]

            # Merge attributes and tag source.
            attrs = dict(rec.attributes or {})
            attrs.setdefault("source", "code" if is_code else "paper")

            joint_nodes[new_id] = NodeRecord(
                node_id=new_id,
                level=new_level,
                type=rec.type,
                parent_id=new_parent,
                name=rec.name,
                attributes=attrs,
            )

            id_map[old_id] = new_id
            level_counts[new_level] = level_counts.get(new_level, 0) + 1

        return id_map

    code_id_map = _remap_domain(src_nodes=code_nodes, domain_node_id=code_domain_id, is_code=True)
    paper_id_map = _remap_domain(
        src_nodes=paper_nodes, domain_node_id=paper_domain_id, is_code=False
    )

    return joint_nodes, code_id_map, paper_id_map, level_counts


def _rewrite_spans(
    spans_path: Path,
    *,
    domain_node_id: int,
    id_map: Dict[int, int],
) -> List[Dict[str, object]]:
    """
    Rewrite node_path for spans from a single domain into the joint ID space.
    """
    out: List[Dict[str, object]] = []
    root_id = 0

    for rec in _iter_spans(spans_path):
        old_path = rec.get("node_path")
        if not isinstance(old_path, list) or not old_path:
            # Leave record untouched if it lacks a path.
            out.append(rec)
            continue

        # Old paths are expected to be [root_id, ... local node ids ...].
        # We drop the old root and map each remaining id via id_map.
        new_path: List[int] = [root_id, domain_node_id]
        for old_id in old_path[1:]:
            new_id = id_map.get(int(old_id))
            if new_id is None:
                # If we don't know this node, skip it; better a shorter path
                # than a broken one.
                continue
            new_path.append(new_id)

        # Ensure at least [root, domain] is present.
        if len(new_path) < 2:
            new_path = [root_id, domain_node_id]

        rec = dict(rec)
        rec["node_path"] = new_path
        out.append(rec)

    return out


def _write_nodes_jsonl(nodes: Dict[int, NodeRecord], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for nid, rec in sorted(nodes.items(), key=lambda kv: kv[0]):
            f.write(
                json.dumps(
                    {
                        "node_id": rec.node_id,
                        "level": rec.level,
                        "type": rec.type,
                        "parent_id": rec.parent_id,
                        "name": rec.name,
                        "attributes": rec.attributes,
                    }
                )
                + "\n"
            )


def _write_level_sizes(level_counts: Dict[int, int], out_path: Path) -> None:
    # Exclude root (level 0) from level_sizes; TOLBERT heads start at level 1.
    level_sizes = {level: count for level, count in level_counts.items() if level > 0}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"level_sizes": level_sizes}, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a joint code+paper Tree-of-Life from per-domain spans and nodes.",
    )
    ap.add_argument("--code_spans", type=str, required=True, help="Code spans JSONL file.")
    ap.add_argument("--code_nodes", type=str, required=True, help="Code nodes JSONL file.")
    ap.add_argument("--paper_spans", type=str, required=True, help="Paper spans JSONL file.")
    ap.add_argument("--paper_nodes", type=str, required=True, help="Paper nodes JSONL file.")

    ap.add_argument(
        "--out_code_spans",
        type=str,
        required=True,
        help="Output path for code spans JSONL with joint node_path.",
    )
    ap.add_argument(
        "--out_paper_spans",
        type=str,
        required=True,
        help="Output path for paper spans JSONL with joint node_path.",
    )
    ap.add_argument(
        "--out_nodes",
        type=str,
        required=True,
        help="Output path for joint nodes JSONL.",
    )
    ap.add_argument(
        "--out_level_sizes",
        type=str,
        required=True,
        help="Output path for joint level_sizes JSON helper JSON.",
    )

    ap.add_argument(
        "--code_domain_name",
        type=str,
        default="Code",
        help="Name for the code domain node at level 1.",
    )
    ap.add_argument(
        "--paper_domain_name",
        type=str,
        default="Papers",
        help="Name for the paper domain node at level 1.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    code_nodes_path = Path(args.code_nodes)
    paper_nodes_path = Path(args.paper_nodes)
    code_spans_path = Path(args.code_spans)
    paper_spans_path = Path(args.paper_spans)

    if not code_nodes_path.is_file():
        raise FileNotFoundError(f"code_nodes does not exist or is not a file: {code_nodes_path}")
    if not paper_nodes_path.is_file():
        raise FileNotFoundError(
            f"paper_nodes does not exist or is not a file: {paper_nodes_path}"
        )
    if not code_spans_path.is_file():
        raise FileNotFoundError(f"code_spans does not exist or is not a file: {code_spans_path}")
    if not paper_spans_path.is_file():
        raise FileNotFoundError(
            f"paper_spans does not exist or is not a file: {paper_spans_path}"
        )

    code_nodes = _load_nodes(code_nodes_path)
    paper_nodes = _load_nodes(paper_nodes_path)

    joint_nodes, code_id_map, paper_id_map, level_counts = _build_joint_ontology(
        code_nodes=code_nodes,
        paper_nodes=paper_nodes,
        code_domain_name=args.code_domain_name,
        paper_domain_name=args.paper_domain_name,
    )

    # Rewrite spans into the joint ID space.
    # Domain node IDs are always 1 (code) and 2 (paper) per _build_joint_ontology.
    code_domain_id = 1
    paper_domain_id = 2
    code_spans_joint = _rewrite_spans(
        code_spans_path,
        domain_node_id=code_domain_id,
        id_map=code_id_map,
    )
    paper_spans_joint = _rewrite_spans(
        paper_spans_path,
        domain_node_id=paper_domain_id,
        id_map=paper_id_map,
    )

    out_code_spans_path = Path(args.out_code_spans)
    out_paper_spans_path = Path(args.out_paper_spans)
    out_nodes_path = Path(args.out_nodes)
    out_level_sizes_path = Path(args.out_level_sizes)

    out_code_spans_path.parent.mkdir(parents=True, exist_ok=True)
    out_paper_spans_path.parent.mkdir(parents=True, exist_ok=True)
    out_nodes_path.parent.mkdir(parents=True, exist_ok=True)
    out_level_sizes_path.parent.mkdir(parents=True, exist_ok=True)

    _write_spans(out_code_spans_path, code_spans_joint)
    _write_spans(out_paper_spans_path, paper_spans_joint)
    _write_nodes_jsonl(joint_nodes, out_nodes_path)
    _write_level_sizes(level_counts, out_level_sizes_path)


if __name__ == "__main__":
    main()


