"""
Build a simple Tree-of-Life taxonomy and span records from a single repository
using the language-agnostic RepoGraph.

This is a concrete, graph-driven implementation of the code-side part of
`docs/tree_of_life.md`:

  - Nodes:
      level 0: root
      level 1: language nodes (Python, Java, C/C++, Go, JS/TS, ...)
      level 2: repo node
      level 3: top-level directory nodes (subtrees within the repo)
      level 4: file nodes
      level 5: symbol nodes (functions / methods / classes), when available

  - Edges:
      parent_id / child_id edges forming a tree over the above nodes.

  - Spans:
      one span per file, with a node_path from root → language → repo →
      top-level directory → file.

You can use the resulting nodes.jsonl / edges.jsonl / spans.jsonl as direct
inputs to the training pipeline described in `docs/tree_of_life.md` and
`docs/training.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.repo_graph import RepoGraph


def _primary_language_for_file_entity(labels: List[str]) -> Optional[str]:
    for lab in labels:
        if lab.startswith("lang:"):
            return lab.split(":", 1)[1]
    return None


def build_tree_for_repo(
    repo_root: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Return (nodes, edges, spans) lists.
    """
    g = RepoGraph(str(repo_root))
    ents = list(g.entities())

    # Node id assignment
    next_id = 0

    def alloc_id() -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    spans: List[Dict[str, object]] = []

    # Level 0: root
    root_id = alloc_id()
    nodes.append(
        {
            "node_id": root_id,
            "level": 0,
            "type": "root",
            "parent_id": None,
            "name": "Root",
            "attributes": {},
        }
    )

    # Language nodes (level 1)
    lang_to_id: Dict[str, int] = {}

    # Single repo node (level 2)
    repo_name = os.path.basename(os.path.abspath(repo_root))
    # We will pick a "primary language" below for the repo parent.

    # File / symbol entities from RepoGraph
    file_ents: List[Tuple[str, object]] = []  # (entity_id, entity)
    symbol_ents: List[Tuple[str, object]] = []
    for e in ents:
        if getattr(e, "kind", "") == "file":
            file_ents.append((e.id, e))
        elif getattr(e, "kind", "") in ("function", "class", "method"):
            symbol_ents.append((e.id, e))

    # Collect languages observed in files
    file_langs: Dict[str, int] = {}
    for _, e in file_ents:
        lang = _primary_language_for_file_entity(getattr(e, "labels", []))
        if not lang:
            continue
        file_langs[lang] = file_langs.get(lang, 0) + 1

    # Create language nodes
    for lang in sorted(file_langs.keys()):
        nid = alloc_id()
        lang_to_id[lang] = nid
        nodes.append(
            {
                "node_id": nid,
                "level": 1,
                "type": "language",
                "parent_id": root_id,
                "name": lang,
                "attributes": {},
            }
        )
        edges.append({"parent_id": root_id, "child_id": nid})

    # Choose a primary language for the repo (highest file count; fallback: None)
    primary_lang: Optional[str] = None
    if file_langs:
        primary_lang = max(file_langs.items(), key=lambda kv: kv[1])[0]

    repo_parent = lang_to_id.get(primary_lang, root_id)
    repo_id = alloc_id()
    nodes.append(
        {
            "node_id": repo_id,
            "level": 2,
            "type": "repo",
            "parent_id": repo_parent,
            "name": repo_name,
            "attributes": {"root": str(repo_root)},
        }
    )
    edges.append({"parent_id": repo_parent, "child_id": repo_id})

    # Level 3: top-level directories / subtrees
    subdir_to_id: Dict[str, int] = {}
    for _, e in file_ents:
        rel = str(getattr(e, "attributes", {}).get("rel_path", ""))
        top = rel.split("/", 1)[0] if "/" in rel else ""
        key = top or "(root)"
        if key in subdir_to_id:
            continue
        nid = alloc_id()
        subdir_to_id[key] = nid
        nodes.append(
            {
                "node_id": nid,
                "level": 3,
                "type": "subdir",
                "parent_id": repo_id,
                "name": key,
                "attributes": {},
            }
        )
        edges.append({"parent_id": repo_id, "child_id": nid})

    # Level 4: files
    fileid_to_nodeid: Dict[str, int] = {}
    for eid, e in file_ents:
        rel = str(getattr(e, "attributes", {}).get("rel_path", ""))
        top = rel.split("/", 1)[0] if "/" in rel else ""
        key = top or "(root)"
        parent_id = subdir_to_id[key]
        nid = alloc_id()
        fileid_to_nodeid[eid] = nid
        nodes.append(
            {
                "node_id": nid,
                "level": 4,
                "type": "file",
                "parent_id": parent_id,
                "name": rel,
                "attributes": {"artifact_uri": getattr(e, "artifact_uri", None)},
            }
        )
        edges.append({"parent_id": parent_id, "child_id": nid})

    # Level 5: symbols (functions / methods / classes).
    for _, e in symbol_ents:
        art_uri = getattr(e, "artifact_uri", None)
        if not art_uri:
            continue
        # Find the file entity that owns this symbol via artifact_uri match.
        parent_node_id: Optional[int] = None
        for fe_id, fe in file_ents:
            if getattr(fe, "artifact_uri", None) == art_uri:
                parent_node_id = fileid_to_nodeid.get(fe_id)
                break
        if parent_node_id is None:
            continue
        nid = alloc_id()
        nodes.append(
            {
                "node_id": nid,
                "level": 5,
                "type": getattr(e, "kind", "symbol"),
                "parent_id": parent_node_id,
                "name": str(getattr(e, "attributes", {}).get("name", "")),
                "attributes": {
                    "artifact_uri": art_uri,
                    "span": {
                        "start_line": getattr(getattr(e, "span", None), "start_line", None),  # type: ignore[attr-defined]  # noqa: E501
                        "end_line": getattr(getattr(e, "span", None), "end_line", None),  # type: ignore[attr-defined]  # noqa: E501
                    },
                },
            }
        )
        edges.append({"parent_id": parent_node_id, "child_id": nid})

    # Spans: one per file, rooted at file nodes.
    for eid, e in file_ents:
        rel = str(getattr(e, "attributes", {}).get("rel_path", ""))
        abs_path = repo_root / rel
        try:
            text = abs_path.read_text(encoding="utf-8")
        except Exception:
            continue
        file_node_id = fileid_to_nodeid[eid]
        # Node path: root → language (primary) → repo → subdir → file.
        lang_id = repo_parent if repo_parent != root_id else None
        top = rel.split("/", 1)[0] if "/" in rel else ""
        key = top or "(root)"
        subdir_id = subdir_to_id[key]
        path: List[int] = [root_id]
        if lang_id is not None:
            path.append(lang_id)
        path.extend([repo_id, subdir_id, file_node_id])
        spans.append(
            {
                "span_id": rel,
                "text": text,
                "source_id": rel,
                "node_path": path,
                "meta": {},
            }
        )

    return nodes, edges, spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", type=str, help="Path to a single repository root")
    ap.add_argument("--nodes-out", type=str, required=True, help="Output nodes.jsonl")
    ap.add_argument("--edges-out", type=str, required=True, help="Output edges.jsonl")
    ap.add_argument(
        "--spans-out",
        type=str,
        default=None,
        help="Optional spans.jsonl output (one span per file)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    nodes, edges, spans = build_tree_for_repo(repo_root)

    nodes_path = Path(args.nodes_out)
    edges_path = Path(args.edges_out)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)

    with nodes_path.open("w", encoding="utf-8") as f:
        for row in nodes:
            f.write(json.dumps(row) + "\n")

    with edges_path.open("w", encoding="utf-8") as f:
        for row in edges:
            f.write(json.dumps(row) + "\n")

    if args.spans_out is not None:
        spans_path = Path(args.spans_out)
        spans_path.parent.mkdir(parents=True, exist_ok=True)
        with spans_path.open("w", encoding="utf-8") as f:
            for row in spans:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()


