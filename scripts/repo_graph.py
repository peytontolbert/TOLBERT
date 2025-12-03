from __future__ import annotations

import os
import re
import hashlib
from typing import Iterable, List, Tuple, Dict, Optional, Set, Any

from modules.program_graph import ProgramGraph, Entity, Edge, Artifact, Span, ResolvedAnchor, EntityId


def program_id_for_repo(repo_root: str) -> str:
    base = os.path.basename(os.path.abspath(repo_root)) or "repo"
    return base


def artifact_uri(program_id: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"program://{program_id}/artifact/{rel}"


def parse_program_uri(uri: str) -> Tuple[str, str, str, Optional[Tuple[int, int]]]:
    m = re.match(r"^program://([^/]+)/([^/]+)/(.+?)(?:#L(\d+)-L(\d+))?$", uri)
    if not m:
        raise ValueError(f"invalid program uri: {uri}")
    pid, kind, res, a, b = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    span = (int(a), int(b)) if (a and b) else None
    return pid, kind, res, span


class RepoGraph(ProgramGraph):
    def __init__(self, repo_root: str, ignore: Optional[List[str]] = None):
        self.repo_root = os.path.abspath(repo_root)
        self.program_id = program_id_for_repo(self.repo_root)
        self.ignore_rules = [s for s in (ignore or []) if s]
        self._file_hash: Dict[str, str] = {}
        # Cached graph views
        self._entities: Dict[EntityId, Entity] = {}
        self._edges: List[Edge] = []
        self._built: bool = False
        # Convenience indices
        #   - absolute file path → file-entity id
        #   - symbol name (lowercased) → list of entity ids
        self._file_entity_for_abs: Dict[str, EntityId] = {}
        self._symbols_by_name: Dict[str, List[EntityId]] = {}

    # ProgramGraph: core views
    def entities(self) -> Iterable[Entity]:
        self._ensure_built()
        return self._entities.values()

    def edges(self) -> Iterable[Edge]:
        self._ensure_built()
        return list(self._edges)

    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
        return []

    def subgraph(self, seeds: List[EntityId], radius: int) -> "ProgramGraph":
        if not seeds or radius <= 0:
            return self
        # Generic BFS over current edges view
        adj: Dict[str, List[str]] = {}
        for e in self.edges():
            adj.setdefault(e.src, []).append(e.dst)
            adj.setdefault(e.dst, []).append(e.src)
        cur = set(seeds)
        seen = set(cur)
        for _ in range(max(1, radius)):
            nxt: Set[str] = set()
            for s in list(cur):
                for nb in adj.get(s, []):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            cur = nxt
        # For now, RepoGraph exposes a single global view; callers that need
        # an actual induced subgraph can post-filter entities/edges.
        return self  # pragma: no cover - view semantics only

    def artifacts(self, kind: str) -> Iterable[Artifact]:
        if kind not in ("artifact", "source"):
            return []
        self._ensure_built()
        out: List[Artifact] = []
        for fp in self._discover_files(self.repo_root, self.ignore_rules):
            rel = os.path.relpath(fp, self.repo_root).replace("\\", "/")
            out.append(
                Artifact(
                    uri=artifact_uri(self.program_id, rel),
                    type="source",
                    hash=self._hash_for(fp),
                    span=None,
                )
            )
        return out

    def resolve(self, uri: str) -> ResolvedAnchor:
        pid, kind, res, span = parse_program_uri(uri)
        if pid != self.program_id:
            raise ValueError(f"program id mismatch: {pid} != {self.program_id}")
        if kind == "artifact":
            abs_fp = os.path.abspath(os.path.join(self.repo_root, res))
            if not os.path.isfile(abs_fp):
                raise FileNotFoundError(f"artifact not found: {abs_fp}")
            a = int(span[0]) if span else 1
            b = int(span[1]) if span else self._safe_count_lines(abs_fp)
            rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
            return ResolvedAnchor(
                artifact_uri=artifact_uri(self.program_id, rel),
                span=Span(start_line=a, end_line=b),
                hash=self._hash_for(abs_fp),
            )
        # Let subclass handle entity URIs
        return self._resolve_entity_uri(kind, res, span)

    # Hooks for subclasses
    def _resolve_entity_uri(self, kind: str, resource: str, span: Optional[Tuple[int, int]]) -> ResolvedAnchor:
        raise KeyError(f"unrecognized entity uri for kind={kind}, resource={resource}")

    # Build / utilities
    def _ensure_built(self) -> None:
        if self._built:
            return
        self._build_graph()
        self._built = True

    def _build_graph(self) -> None:
        """
        Populate entity and edge sets for this repository.

        This is intentionally language-agnostic at the core:
          - Always creates a repo-level entity.
          - Always creates file-level entities for discovered source artifacts.

        Language-specific structure (imports, calls, packages / modules) is
        layered on top via simple heuristics and optional backends.
        """
        # Repo entity
        repo_eid: EntityId = f"repo:{self.program_id}"
        if repo_eid not in self._entities:
            self._entities[repo_eid] = Entity(
                id=repo_eid,
                kind="repo",
                uri=f"program://{self.program_id}/repo",
                artifact_uri=None,
                span=None,
                labels=["kind:repo"],
                attributes={"root": self.repo_root},
            )

        # Discover files and create file-level entities
        files = self._discover_files(self.repo_root, self.ignore_rules)
        for fp in files:
            rel = os.path.relpath(fp, self.repo_root).replace("\\", "/")
            uri = artifact_uri(self.program_id, rel)
            labels = ["kind:file"] + self._language_labels_for(rel)
            eid: EntityId = uri  # stable id tied to artifact URI
            if eid not in self._entities:
                self._entities[eid] = Entity(
                    id=eid,
                    kind="file",
                    uri=uri,
                    artifact_uri=uri,
                    span=None,
                    labels=labels,
                    attributes={"rel_path": rel},
                )
            self._file_entity_for_abs[os.path.abspath(fp)] = eid
            # Repo "owns" file
            self._edges.append(
                Edge(
                    src=repo_eid,
                    dst=eid,
                    type="owns",
                    attributes={},
                )
            )

        # Language-specific edges and entities.
        self._build_language_edges_and_entities(files)

    def _build_language_edges_and_entities(self, files: List[str]) -> None:
        """
        Add language-specific structure for supported stacks.

        - C / C++: file-level imports from #include relationships.
        - Go: package entities and import edges.
        - Java: imported type entities and import edges.
        - JS / TS: module/file imports from ES modules / require().
        - Python: optional integration via scripts.codegraph_core.CodeGraph
          for module / function / class entities and imports / calls.
        """
        # Best-effort C / C++ / Go / Java / JS import graphs
        for abs_fp in files:
            rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
            _, ext = os.path.splitext(rel.lower())
            if ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"):
                self._add_c_includes(abs_fp)
            elif ext == ".go":
                self._add_go_imports(abs_fp)
            elif ext == ".java":
                self._add_java_imports(abs_fp)
            elif ext in (".js", ".jsx", ".mjs", ".ts", ".tsx"):
                self._add_js_like_imports(abs_fp)

        # Optional: richer Python graph via scripts.codegraph_core
        try:
            from scripts.codegraph_core import CodeGraph as _PyCodeGraph  # type: ignore
        except Exception:
            _PyCodeGraph = None  # type: ignore

        if _PyCodeGraph is not None:
            try:
                cg = _PyCodeGraph(self.repo_root, ignore=self.ignore_rules).build()
            except Exception:
                cg = None
            if cg is not None:
                self._ingest_python_codegraph(cg)

        # Non-Python symbol entities + heuristic call graph.
        self._add_non_python_symbols_and_calls(files)

    def _discover_files(self, root: str, ignore: List[str]) -> List[str]:
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            if any(ig and ig in dirpath for ig in ignore):
                continue
            for fn in filenames:
                ap = os.path.abspath(os.path.join(dirpath, fn))
                out.append(ap)
        return out

    def _language_labels_for(self, rel_path: str) -> List[str]:
        """
        Best-effort language tags for a repo-relative path based on extension.

        These are intentionally coarse and are only used to annotate entities
        (e.g., file-level entities) so that downstream tools can filter or
        group by language family when needed.
        """
        _, ext = os.path.splitext(rel_path.lower())
        labels: List[str] = []
        if ext == ".py":
            labels.append("lang:python")
        elif ext in (".c", ".h"):
            labels.append("lang:c")
        elif ext in (".cc", ".cpp", ".cxx", ".hpp"):
            labels.append("lang:cpp")
        elif ext in (".js", ".jsx", ".mjs"):
            labels.append("lang:js")
        elif ext in (".ts", ".tsx"):
            labels.append("lang:ts")
        elif ext == ".go":
            labels.append("lang:go")
        elif ext == ".java":
            labels.append("lang:java")
        elif ext == ".md":
            labels.append("lang:markdown")
        return labels

    def _safe_count_lines(self, abs_file: str) -> int:
        try:
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 1

    def _hash_for(self, abs_file: str) -> str:
        if abs_file in self._file_hash:
            return self._file_hash[abs_file]
        try:
            with open(abs_file, "rb") as fh:
                raw = fh.read()
            h = hashlib.sha256(raw).hexdigest()
        except Exception:
            h = ""
        self._file_hash[abs_file] = h
        return h

    # --- Language-specific helpers ---

    def _add_symbol_entity(
        self,
        abs_fp: str,
        name: str,
        kind: str,
        lang_label: str,
        start_line: int,
        end_line: int,
    ) -> Optional[EntityId]:
        abs_fp = os.path.abspath(abs_fp)
        file_eid = self._file_entity_for_abs.get(abs_fp)
        if not file_eid:
            return None
        # Stable id: lang-specific prefix + file rel path + symbol name.
        rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
        sym_id: EntityId = f"{lang_label}:{rel}:{name}"
        if sym_id in self._entities:
            return sym_id
        file_ent = self._entities[file_eid]
        span = Span(start_line=int(start_line), end_line=int(end_line))
        uri = f"program://{self.program_id}/sym/{lang_label}/{rel}#{name}"
        labels = [f"kind:{kind}", f"lang:{lang_label}"]
        self._entities[sym_id] = Entity(
            id=sym_id,
            kind=kind,
            uri=uri,
            artifact_uri=file_ent.artifact_uri,
            span=span,
            labels=labels,
            attributes={"name": name, "file": rel},
        )
        # file "owns" symbol
        self._edges.append(
            Edge(
                src=file_eid,
                dst=sym_id,
                type="owns",
                attributes={"lang": lang_label},
            )
        )
        self._symbols_by_name.setdefault(name.lower(), []).append(sym_id)
        return sym_id

    def _add_c_includes(self, abs_fp: str) -> None:
        src_eid = self._file_entity_for_abs.get(os.path.abspath(abs_fp))
        if not src_eid:
            return
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            return
        # Match #include "local.h" and #include <local.h>
        rx = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)
        for m in rx.finditer(text):
            target = m.group(1).strip()
            if not target:
                continue
            # Resolve relative to current directory; ignore obvious system headers.
            if "/" not in target and "\\" not in target and "." not in target:
                continue
            cand = os.path.abspath(os.path.join(os.path.dirname(abs_fp), target))
            dst_eid = self._file_entity_for_abs.get(cand)
            if not dst_eid:
                continue
            self._edges.append(
                Edge(
                    src=src_eid,
                    dst=dst_eid,
                    type="imports",
                    attributes={"lang": "c_cpp"},
                )
            )

    def _add_go_imports(self, abs_fp: str) -> None:
        src_eid = self._file_entity_for_abs.get(os.path.abspath(abs_fp))
        if not src_eid:
            return
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return
        in_block = False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("import "):
                # Single-line: import "pkg/path"
                in_block = "(" in stripped and not stripped.rstrip().endswith(")")
                m = re.search(r'["`](.+?)["`]', stripped)
                if m:
                    pkg = m.group(1).strip()
                    self._add_go_import_edge(src_eid, pkg)
            elif in_block:
                if stripped.startswith(")"):
                    in_block = False
                    continue
                m = re.search(r'["`](.+?)["`]', stripped)
                if m:
                    pkg = m.group(1).strip()
                    self._add_go_import_edge(src_eid, pkg)

    def _add_go_import_edge(self, src_eid: EntityId, pkg: str) -> None:
        if not pkg:
            return
        eid: EntityId = f"go:pkg:{pkg}"
        if eid not in self._entities:
            self._entities[eid] = Entity(
                id=eid,
                kind="package",
                uri=f"program://{self.program_id}/go_pkg/{pkg}",
                artifact_uri=None,
                span=None,
                labels=["kind:package", "lang:go"],
                attributes={"package": pkg},
            )
        self._edges.append(
            Edge(
                src=src_eid,
                dst=eid,
                type="imports",
                    attributes={"lang": "go"},
            )
        )

    def _add_java_imports(self, abs_fp: str) -> None:
        src_eid = self._file_entity_for_abs.get(os.path.abspath(abs_fp))
        if not src_eid:
            return
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            return
        rx = re.compile(r'^\s*import\s+(static\s+)?([a-zA-Z0-9_.]+)\s*;', re.MULTILINE)
        for m in rx.finditer(text):
            fqn = m.group(2)
            if not fqn:
                continue
            eid: EntityId = f"java:import:{fqn}"
            if eid not in self._entities:
                self._entities[eid] = Entity(
                    id=eid,
                    kind="type",
                    uri=f"program://{self.program_id}/java_type/{fqn}",
                    artifact_uri=None,
                    span=None,
                    labels=["kind:type", "lang:java"],
                    attributes={"fqn": fqn},
                )
            self._edges.append(
                Edge(
                    src=src_eid,
                    dst=eid,
                    type="imports",
                    attributes={"lang": "java"},
                )
            )

    def _add_js_like_imports(self, abs_fp: str) -> None:
        src_eid = self._file_entity_for_abs.get(os.path.abspath(abs_fp))
        if not src_eid:
            return
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            return
        rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
        base_dir = os.path.dirname(os.path.abspath(abs_fp))

        specs: Set[str] = set()
        # import x from 'spec';  import 'spec';
        for m in re.finditer(r'^\s*import\s+(?:.+?\s+from\s+)?[\'"](.+?)[\'"]', text, re.MULTILINE):
            specs.add(m.group(1).strip())
        # require('spec')
        for m in re.finditer(r'require\(\s*[\'"](.+?)[\'"]\s*\)', text):
            specs.add(m.group(1).strip())

        for spec in specs:
            if not spec:
                continue
            if spec.startswith("."):
                # Local module → resolve to file entity if possible.
                dst_eid = self._resolve_local_js_like_spec(base_dir, spec)
                if dst_eid:
                    self._edges.append(
                        Edge(
                            src=src_eid,
                            dst=dst_eid,
                            type="imports",
                            attributes={"lang": "js_ts"},
                        )
                    )
            else:
                # External package
                eid: EntityId = f"js:pkg:{spec}"
                if eid not in self._entities:
                    self._entities[eid] = Entity(
                        id=eid,
                        kind="package",
                        uri=f"program://{self.program_id}/js_pkg/{spec}",
                        artifact_uri=None,
                        span=None,
                        labels=["kind:package", "lang:js"],
                        attributes={"package": spec},
                    )
                self._edges.append(
                    Edge(
                        src=src_eid,
                        dst=eid,
                        type="imports",
                        attributes={"lang": "js_ts"},
                    )
                )

    def _resolve_local_js_like_spec(self, base_dir: str, spec: str) -> Optional[EntityId]:
        # Resolve "./foo" style specifiers to actual files if present.
        # Try with and without common JS/TS extensions.
        exts = ["", ".js", ".jsx", ".mjs", ".ts", ".tsx"]
        for ext in exts:
            cand = spec
            if not cand.endswith(ext):
                cand = spec + ext
            abs_cand = os.path.abspath(os.path.join(base_dir, cand))
            eid = self._file_entity_for_abs.get(abs_cand)
            if eid:
                return eid
        return None

    def _ingest_python_codegraph(self, cg: Any) -> None:
        """
        Integrate a scripts.codegraph_core.CodeGraph instance as one backend.

        This adds:
          - Module / function / class entities for Python.
          - "owns" edges from file entities to these symbols.
          - "imports" / "calls" / "tests" edges as provided by the backend.
        """
        # Entities
        for ent in cg.entities():
            abs_file = os.path.abspath(getattr(ent, "file", ""))
            file_eid = self._file_entity_for_abs.get(abs_file)
            if not file_eid:
                continue
            eid: EntityId = getattr(ent, "id")
            if eid in self._entities:
                continue
            span = Span(start_line=int(ent.start_line), end_line=int(ent.end_line))
            uri = f"program://{self.program_id}/py_entity/{eid}"
            labels = [f"kind:{ent.kind}", "lang:python"]
            self._entities[eid] = Entity(
                id=eid,
                kind=str(ent.kind),
                uri=uri,
                artifact_uri=self._entities[file_eid].artifact_uri,
                span=span,
                labels=labels,
                attributes={"name": ent.name},
            )
            self._symbols_by_name.setdefault(ent.name.lower(), []).append(eid)
            # file "owns" symbol
            self._edges.append(
                Edge(
                    src=file_eid,
                    dst=eid,
                    type="owns",
                    attributes={"lang": "python"},
                )
            )

        # Edges from backend (imports, calls, owns, tests, ...)
        for e in cg.edges():
            if (e.src not in self._entities) or (e.dst not in self._entities):
                continue
            self._edges.append(
                Edge(
                    src=e.src,
                    dst=e.dst,
                    type=e.type,
                    attributes={"lang": "python"},
                )
            )

    def _add_non_python_symbols_and_calls(self, files: List[str]) -> None:
        """
        Add best-effort function/class symbols and a simple call graph for
        non-Python languages (C/C++, Go, Java, JS/TS).

        The call graph is heuristic: we scan for identifier`(` patterns in
        each file and connect the *file entity* to any known symbol entity
        with a matching name.
        """
        # Pass 1: symbols
        for abs_fp in files:
            rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
            _, ext = os.path.splitext(rel.lower())
            if ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"):
                self._index_c_cpp_symbols(abs_fp)
            elif ext == ".go":
                self._index_go_symbols(abs_fp)
            elif ext == ".java":
                self._index_java_symbols(abs_fp)
            elif ext in (".js", ".jsx", ".mjs", ".ts", ".tsx"):
                self._index_js_ts_symbols(abs_fp)

        # Pass 2: heuristic calls (file-level callers)
        call_rx = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        for abs_fp in files:
            src_eid = self._file_entity_for_abs.get(os.path.abspath(abs_fp))
            if not src_eid:
                continue
            try:
                with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            for m in call_rx.finditer(text):
                name = m.group(1)
                if not name:
                    continue
                cands = self._symbols_by_name.get(name.lower())
                if not cands:
                    continue
                for dst_id in cands:
                    self._edges.append(
                        Edge(
                            src=src_eid,
                            dst=dst_id,
                            type="calls",
                            attributes={"heuristic": "name_scan"},
                        )
                    )

    def _index_c_cpp_symbols(self, abs_fp: str) -> None:
        """
        Very coarse C/C++ function and class detector based on regexes.
        """
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return
        # Function definitions: return-type name(args) { ... }
        func_rx = re.compile(
            r"^[\t ]*(?:[A-Za-z_][\w\s\*\:&<>\[\]]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{"
        )
        # Class/struct definitions
        class_rx = re.compile(
            r"^[\t ]*(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        )
        for idx, line in enumerate(lines, start=1):
            m_func = func_rx.match(line)
            if m_func:
                name = m_func.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="function", lang_label="c_cpp", start_line=idx, end_line=idx
                )
            m_cls = class_rx.match(line)
            if m_cls:
                name = m_cls.group(2)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="class", lang_label="c_cpp", start_line=idx, end_line=idx
                )

    def _index_go_symbols(self, abs_fp: str) -> None:
        """
        Coarse Go function and method detector.
        """
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return
        # func name(...) or func (recv) name(...)
        func_rx = re.compile(
            r"^[\t ]*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
        )
        for idx, line in enumerate(lines, start=1):
            m = func_rx.match(line)
            if m:
                name = m.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="function", lang_label="go", start_line=idx, end_line=idx
                )

    def _index_java_symbols(self, abs_fp: str) -> None:
        """
        Coarse Java class and method detector.
        """
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return
        class_rx = re.compile(
            r"^[\t ]*(?:public|protected|private|abstract|final|static|\s)*\s*"
            r"(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
        )
        method_rx = re.compile(
            r"^[\t ]*(?:public|protected|private|static|final|synchronized|\s)+"
            r"[A-Za-z_\$][\w\<\>\[\]]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{"
        )
        for idx, line in enumerate(lines, start=1):
            m_cls = class_rx.match(line)
            if m_cls:
                name = m_cls.group(2)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="class", lang_label="java", start_line=idx, end_line=idx
                )
            m_m = method_rx.match(line)
            if m_m:
                name = m_m.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="method", lang_label="java", start_line=idx, end_line=idx
                )

    def _index_js_ts_symbols(self, abs_fp: str) -> None:
        """
        Coarse JS/TS function and class detector.
        """
        try:
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return
        func_rx = re.compile(r"^[\t ]*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        arrow_rx = re.compile(
            r"^[\t ]*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\([^)]*\)\s*=>"
        )
        class_rx = re.compile(r"^[\t ]*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        for idx, line in enumerate(lines, start=1):
            m_fn = func_rx.match(line)
            if m_fn:
                name = m_fn.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="function", lang_label="js_ts", start_line=idx, end_line=idx
                )
            m_arrow = arrow_rx.match(line)
            if m_arrow:
                name = m_arrow.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="function", lang_label="js_ts", start_line=idx, end_line=idx
                )
            m_cls = class_rx.match(line)
            if m_cls:
                name = m_cls.group(1)
                self._add_symbol_entity(
                    abs_fp, name=name, kind="class", lang_label="js_ts", start_line=idx, end_line=idx
                )


