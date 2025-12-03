"""
Build CodeHierarchy-style JSONL spans and simple ontology metadata.

This is a *reference* implementation of the dataset construction described
in the TOLBERT paper for the CodeHierarchy benchmark. It does not fetch or
ship any GitHub data; instead, it expects you to point it at a directory of
local repositories plus a small metadata file that assigns each repo to a
language and coarse category.

Input:
  - --repos_root:
      Directory that contains one subdirectory per repository, e.g.
        repos_root/
          repo1/
            ...
          repo2/
            ...
  - --metadata_file:
      A CSV or JSONL file with, at minimum, the fields:
        repo_name, language, category
      where:
        - repo_name matches the subdirectory name under repos_root
        - language is a string label (e.g. "Python", "Java", "C++")
        - category is a coarse repo category (e.g. "Web", "ML", "Systems")

Outputs:
  - --spans_out:
      JSONL file with one record per *file* (not per function) of the form:
        {
          "span_id": "repo/file.py",
          "text": "... full file contents ...",
          "source_id": "repo/file.py",
          "node_path": [root_id, lang_id, cat_id, repo_id],
          "meta": { ... optional extra metadata ... }
        }

  - --nodes_out (optional):
      JSONL file describing ontology nodes with fields:
        node_id, level, type, parent_id, name

  - --level_sizes_out (optional):
      Small JSON file with:
        {"level_sizes": [num_level0, num_level1, num_level2, num_level3]}
      which you can copy into your training config as `level_sizes`.

This script is intentionally lightweight. It is meant to make the paper's
CodeHierarchy setup reproducible given local mirrors of the repositories and
a small metadata file, without hard-coding any GitHub-specific details.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RepoMeta:
    repo_name: str
    language: str
    category: str


def load_metadata(path: Path) -> List[RepoMeta]:
    metas: List[RepoMeta] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                metas.append(
                    RepoMeta(
                        repo_name=obj["repo_name"],
                        language=obj["language"],
                        category=obj["category"],
                    )
                )
        return metas

    # Default: CSV with header
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metas.append(
                RepoMeta(
                    repo_name=row["repo_name"],
                    language=row["language"],
                    category=row["category"],
                )
            )
    return metas


def build_ontology(metas: List[RepoMeta]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build simple integer ID mappings:
      - language -> lang_node_id
      - (language, category) -> cat_node_id
      - repo_name -> repo_node_id

    Level conventions:
      level 0: root (id 0)
      level 1: language nodes
      level 2: category nodes (per language)
      level 3: repo nodes
    """
    next_id = 0
    # Root
    root_id = next_id
    next_id += 1

    lang_ids: Dict[str, int] = {}
    cat_ids: Dict[str, int] = {}
    repo_ids: Dict[str, int] = {}

    for m in metas:
        if m.language not in lang_ids:
            lang_ids[m.language] = next_id
            next_id += 1
        cat_key = f"{m.language}::{m.category}"
        if cat_key not in cat_ids:
            cat_ids[cat_key] = next_id
            next_id += 1
        if m.repo_name not in repo_ids:
            repo_ids[m.repo_name] = next_id
            next_id += 1

    return lang_ids, cat_ids, repo_ids


def write_nodes_jsonl(
    out_path: Path,
    lang_ids: Dict[str, int],
    cat_ids: Dict[str, int],
    repo_ids: Dict[str, int],
) -> None:
    """
    Emit a minimal nodes.jsonl compatible with docs/tree_of_life.md.
    """
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

        # Languages (level 1)
        for lang, nid in lang_ids.items():
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 1,
                        "type": "language",
                        "parent_id": 0,
                        "name": lang,
                        "attributes": {},
                    }
                )
                + "\n"
            )

        # Categories (level 2)
        for key, nid in cat_ids.items():
            lang, cat = key.split("::", 1)
            parent_id = lang_ids[lang]
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 2,
                        "type": "category",
                        "parent_id": parent_id,
                        "name": cat,
                        "attributes": {"language": lang},
                    }
                )
                + "\n"
            )

        # Repos (level 3)
        for repo, nid in repo_ids.items():
            # We do not encode parent here; it can be inferred from metadata if needed.
            f.write(
                json.dumps(
                    {
                        "node_id": nid,
                        "level": 3,
                        "type": "repo",
                        "parent_id": None,
                        "name": repo,
                        "attributes": {},
                    }
                )
                + "\n"
            )


def build_span_records(
    repos_root: Path,
    metas: List[RepoMeta],
    lang_ids: Dict[str, int],
    cat_ids: Dict[str, int],
    repo_ids: Dict[str, int],
) -> List[Dict[str, object]]:
    """
    Iterate over repositories and files, generating one span per file.
    """
    root_id = 0
    spans: List[Dict[str, object]] = []

    meta_by_repo: Dict[str, RepoMeta] = {m.repo_name: m for m in metas}

    for repo_name, repo_id in repo_ids.items():
        repo_dir = repos_root / repo_name
        if not repo_dir.is_dir():
            continue

        m = meta_by_repo.get(repo_name)
        if m is None:
            continue

        lang_id = lang_ids[m.language]
        cat_key = f"{m.language}::{m.category}"
        cat_id = cat_ids[cat_key]

        for dirpath, _, filenames in os.walk(repo_dir):
            for fname in filenames:
                # Basic filter: only include source-like files
                if not fname.endswith((".py", ".java", ".cpp", ".cc", ".cxx", ".h", ".hpp")):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    text = fpath.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                rel_path = fpath.relative_to(repos_root).as_posix()
                span_id = rel_path
                node_path = [root_id, lang_id, cat_id, repo_id]

                spans.append(
                    {
                        "span_id": span_id,
                        "text": text,
                        "source_id": rel_path,
                        "node_path": node_path,
                        "meta": {
                            "repo_name": repo_name,
                            "language": m.language,
                            "category": m.category,
                        },
                    }
                )

    return spans


def write_spans_jsonl(spans: List[Dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in spans:
            f.write(json.dumps(rec) + "\n")


def write_level_sizes(
    lang_ids: Dict[str, int],
    cat_ids: Dict[str, int],
    repo_ids: Dict[str, int],
    out_path: Path,
) -> None:
    """
    Write a small JSON helper with level_sizes for convenience.

    Note: TOLBERTConfig.level_sizes expects a dict[int, int] mapping
    level index -> number of nodes at that level (excluding the root
    level, which typically has no head). For CodeHierarchy we use:
      level 1: languages
      level 2: categories
      level 3: repos

    You can either:
      - Paste this dict into a YAML config (which preserves int keys), or
      - Load it manually and convert keys to ints before constructing
        TOLBERTConfig if you prefer JSON configs.
    """
    level_sizes = {
        1: len(lang_ids),
        2: len(cat_ids),
        3: len(repo_ids),
    }
    out = {"level_sizes": level_sizes}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CodeHierarchy spans JSONL and ontology metadata.")
    ap.add_argument("--repos_root", type=str, required=True, help="Directory containing local repos.")
    ap.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="CSV or JSONL file with columns: repo_name, language, category.",
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    repos_root = Path(args.repos_root)
    if not repos_root.is_dir():
        raise FileNotFoundError(f"repos_root does not exist or is not a directory: {repos_root}")

    meta_path = Path(args.metadata_file)
    metas = load_metadata(meta_path)

    lang_ids, cat_ids, repo_ids = build_ontology(metas)

    spans = build_span_records(
        repos_root=repos_root,
        metas=metas,
        lang_ids=lang_ids,
        cat_ids=cat_ids,
        repo_ids=repo_ids,
    )

    spans_out = Path(args.spans_out)
    spans_out.parent.mkdir(parents=True, exist_ok=True)
    write_spans_jsonl(spans, spans_out)

    if args.nodes_out:
        nodes_out = Path(args.nodes_out)
        nodes_out.parent.mkdir(parents=True, exist_ok=True)
        write_nodes_jsonl(nodes_out, lang_ids, cat_ids, repo_ids)

    if args.level_sizes_out:
        ls_out = Path(args.level_sizes_out)
        ls_out.parent.mkdir(parents=True, exist_ok=True)
        write_level_sizes(lang_ids, cat_ids, repo_ids, ls_out)


if __name__ == "__main__":
    main()


