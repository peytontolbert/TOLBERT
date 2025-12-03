from __future__ import annotations

import os
import ast
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set


@dataclass(frozen=True)
class FileSpan:
    file: str           # absolute path
    start_line: int     # 1-based inclusive
    end_line: int       # 1-based inclusive


@dataclass
class CGEntity:
    id: str             # stable id (e.g., fqn)
    kind: str           # module|function|class|test
    name: str
    file: str           # absolute path
    owner: Optional[str]
    start_line: int
    end_line: int


@dataclass
class CGEdge:
    src: str            # CGEntity.id
    dst: str            # CGEntity.id
    type: str           # imports|calls|owns|tests


class CodeGraph:
    def __init__(self, repo_root: str, ignore: Optional[List[str]] = None):
        self.root = os.path.abspath(repo_root)
        self.ignore_rules = [s for s in (ignore or []) if s]
        self.entities_by_id: Dict[str, CGEntity] = {}
        self.edges_list: List[CGEdge] = []
        self._file_hash: Dict[str, str] = {}
        self._id_by_module: Dict[str, str] = {}
        self._ids_by_file: Dict[str, List[str]] = {}
        self._index_identifiers: Dict[str, List[str]] = {}

    # Build
    def build(self) -> "CodeGraph":
        py_files = self._discover_py_files(self.root, self.ignore_rules)
        for abs_fp in py_files:
            mod = self._module_name_for(abs_fp)
            mid = f"py:{mod}"
            self._id_by_module[mod] = mid
            ent = CGEntity(
                id=mid, kind="module", name=mod, file=abs_fp, owner=None,
                start_line=1, end_line=self._safe_count_lines(abs_fp),
            )
            self.entities_by_id[mid] = ent
            self._ids_by_file.setdefault(abs_fp, []).append(mid)
            self._index_identifiers.setdefault(mod.lower(), []).append(mid)
            # Parse AST for defs/imports/calls
            try:
                with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except Exception:
                tree = None
            if tree is None:
                continue
            # functions/classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = getattr(node, "name", "")
                    a = int(getattr(node, "lineno", 1))
                    b = int(getattr(node, "end_lineno", a))
                    fid = f"py:{mod}.{name}"
                    self.entities_by_id[fid] = CGEntity(
                        id=fid, kind="function", name=name, file=abs_fp, owner=mid,
                        start_line=a, end_line=b,
                    )
                    self._ids_by_file.setdefault(abs_fp, []).append(fid)
                    self.edges_list.append(CGEdge(src=mid, dst=fid, type="owns"))
                    self._index_identifiers.setdefault(name.lower(), []).append(fid)
                elif isinstance(node, ast.ClassDef):
                    name = getattr(node, "name", "")
                    a = int(getattr(node, "lineno", 1))
                    b = int(getattr(node, "end_lineno", a))
                    cid = f"py:{mod}.{name}"
                    self.entities_by_id[cid] = CGEntity(
                        id=cid, kind="class", name=name, file=abs_fp, owner=mid,
                        start_line=a, end_line=b,
                    )
                    self._ids_by_file.setdefault(abs_fp, []).append(cid)
                    self.edges_list.append(CGEdge(src=mid, dst=cid, type="owns"))
                    self._index_identifiers.setdefault(name.lower(), []).append(cid)
            # imports (module-level)
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            im = alias.name
                            if not im:
                                continue
                            tgt_mod = im
                            tid = f"py:{tgt_mod}"
                            self.edges_list.append(CGEdge(src=mid, dst=tid, type="imports"))
                    elif isinstance(node, ast.ImportFrom):
                        im = node.module or ""
                        if not im:
                            continue
                        tgt_mod = im
                        tid = f"py:{tgt_mod}"
                        self.edges_list.append(CGEdge(src=mid, dst=tid, type="imports"))
            except Exception:
                pass
            # calls (best-effort): record identifiers used in Call nodes
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        fn = getattr(node, "func", None)
                        name = None
                        if isinstance(fn, ast.Attribute):
                            name = getattr(fn, "attr", None)
                        elif isinstance(fn, ast.Name):
                            name = fn.id
                        if name:
                            lid = str(name).lower()
                            for cand in self._index_identifiers.get(lid, []):
                                self.edges_list.append(CGEdge(src=mid, dst=cand, type="calls"))
            except Exception:
                pass
            # tests tag
            base = os.path.basename(abs_fp)
            if base.startswith("test_") or base.endswith("_test.py"):
                self.entities_by_id[mid].kind = "test_module"
        # finalize file hashes
        self._precompute_hashes(py_files)
        return self

    # Public accessors
    def entities(self) -> Iterable[CGEntity]:
        return self.entities_by_id.values()

    def edges(self) -> Iterable[CGEdge]:
        # Filter edges whose endpoints are known (post totality)
        known = set(self.entities_by_id.keys())
        for e in self.edges_list:
            if (e.src in known) and (e.dst in known):
                yield e

    def file_hash(self, abs_path: str) -> str:
        return self._file_hash.get(abs_path) or ""

    def ids_for_file(self, abs_path: str) -> List[str]:
        return list(self._ids_by_file.get(abs_path, []))

    def find_identifier_ids(self, token: str) -> List[str]:
        return list(self._index_identifiers.get(token.lower(), []))

    # Helpers
    def _discover_py_files(self, root: str, ignore: List[str]) -> List[str]:
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # naive ignore: drop segments that contain any ignore pattern
            if any(ig in dirpath for ig in ignore):
                continue
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                ap = os.path.abspath(os.path.join(dirpath, fn))
                out.append(ap)
        return out

    def _module_name_for(self, abs_file: str) -> str:
        # repo-relative without extension, path with dots
        rel = os.path.relpath(abs_file, self.root).replace("\\", "/")
        if rel.endswith(".py"):
            rel = rel[:-3]
        parts = [p for p in rel.split("/") if p and p != "__init__"]
        return ".".join(parts) or os.path.splitext(os.path.basename(abs_file))[0]

    def _safe_count_lines(self, abs_file: str) -> int:
        try:
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 1

    def _precompute_hashes(self, files: List[str]) -> None:
        for fp in files:
            try:
                with open(fp, "rb") as fh:
                    raw = fh.read()
                h = hashlib.sha256(raw).hexdigest()
            except Exception:
                h = ""
            self._file_hash[fp] = h


