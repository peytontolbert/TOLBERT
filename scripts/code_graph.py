import os
import ast
import re
import json
import time  # noqa: F401
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
try:
    import pathspec  # type: ignore
except Exception:  # pragma: no cover
    pathspec = None  # type: ignore


@dataclass
class Symbol:
    fqn: str
    name: str
    qualname: str
    kind: str  # module|class|function|variable
    module: str
    file: str
    line: int
    end_line: int
    doc: Optional[str] = None
    signature: Optional[str] = None
    returns: Optional[str] = None


@dataclass
class ModuleInfo:
    module: str
    file: str
    is_test: bool = False
    imports: Dict[str, str] = field(
        default_factory=dict
    )  # alias -> target (module or module.symbol)
    defs: List[str] = field(default_factory=list)  # list of symbol FQNs
    exports: List[str] = field(default_factory=list)  # names from __all__


class CodeGraph:
    def __init__(self, root: str, *, ignore: Optional[List[str]] = None) -> None:
        self.root = os.path.abspath(root)
        # Ignore patterns (relative to root) or glob-like; simple prefix/glob matching
        self._ignore: List[str] = []
        if ignore:
            # normalize to forward-slash relative prefixes for matching
            for pat in ignore:
                if not pat:
                    continue
                p = os.path.normpath(pat)
                # store both relative and absolute forms for convenience
                self._ignore.append(p)
        # Load .gitignore as pathspec if available
        self._pspec = None
        try:
            gi = os.path.join(self.root, ".gitignore")
            if pathspec is not None and os.path.exists(gi):
                with open(gi, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [ln.rstrip("\n") for ln in fh]
                self._pspec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        except Exception:
            self._pspec = None
        self.symbols_by_fqn: Dict[str, Symbol] = {}
        self.symbols_by_name: Dict[str, List[str]] = {}
        self.modules: Dict[str, ModuleInfo] = {}
        self.indexed_files: List[str] = []
        self.calls: List[Tuple[str, str]] = []  # (caller_fqn, callee_fqn_or_key)
        self.module_to_tests: Dict[str, List[str]] = {}
        self.coverage_files: Dict[str, set[int]] = {}
        self.symbol_coverage: Dict[str, float] = {}
        self.module_imports: Dict[str, List[str]] = {}
        self.module_star_imports: Dict[str, List[str]] = {}
        self.pytest_nodes_by_module: Dict[str, List[str]] = {}
        self._cached_mtimes: Dict[str, int] = {}
        self._cached_hashes: Dict[str, str] = {}

    def _is_ignored(self, rel: str) -> bool:
        try:
            r = rel.replace(os.sep, "/")
            # pathspec first
            if self._pspec is not None:
                if self._pspec.match_file(r):
                    return True
            # fallback: prefix match
            for pat in self._ignore:
                pp = pat.replace(os.sep, "/")
                if r == pp or r.startswith(pp + "/"):
                    return True
            return False
        except Exception:
            return False

    @classmethod
    def load_or_build(cls, root: str, *, ignore_cache: bool = False, ignore: Optional[List[str]] = None) -> "CodeGraph":
        g = cls(root=root, ignore=ignore)
        g.build(ignore_cache=ignore_cache)
        return g

    def build(self, ignore_cache: bool = False) -> None:
        cache_path = os.path.join(self.root, ".codegraph.json")
        if (not ignore_cache) and self._load_cache_relaxed(cache_path):
            # Incremental: reindex changed and dependents
            changed, removed = self._detect_changed_files(
                self._cached_mtimes, self._cached_hashes
            )
            if not changed and not removed:
                return
            self._incremental_reindex(changed, removed)
            self._expand_star_imports()
            self._post_resolve_calls()
            self._save_cache(cache_path)
            return
        for dirpath, dirnames, filenames in os.walk(self.root):
            # prune ignored directories in-place
            dir_rel = os.path.relpath(dirpath, self.root)
            # remove child dirs that are ignored
            dirnames[:] = [d for d in dirnames if not self._is_ignored(os.path.join(dir_rel, d))]
            if self._is_ignored(dir_rel):
                continue
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fn)
                if self._is_ignored(os.path.relpath(fpath, self.root)):
                    continue
                try:
                    src = open(fpath, "r", encoding="utf-8").read()
                except Exception:
                    continue
                try:
                    tree = ast.parse(src)
                except Exception:
                    continue
                self.indexed_files.append(fpath)
                self._index_module(fpath, tree)
        # Build test mapping from imports in test modules
        self._build_test_mapping()
        # Expand star imports and post-resolve call targets
        self._expand_star_imports()
        self._post_resolve_calls()
        self._save_cache(cache_path)

    def _add_symbol(self, sym: Symbol) -> None:
        self.symbols_by_fqn[sym.fqn] = sym
        self.symbols_by_name.setdefault(sym.name, []).append(sym.fqn)
        mi = self.modules.setdefault(
            sym.module, ModuleInfo(module=sym.module, file=sym.file)
        )
        if sym.fqn not in mi.defs:
            mi.defs.append(sym.fqn)

    def _index_module(self, path: str, tree: ast.AST) -> None:
        module = self._module_name_for_path(path)
        is_test = ("/tests/" in path) or (os.path.basename(path).startswith("test_"))
        self.modules.setdefault(
            module, ModuleInfo(module=module, file=path, is_test=is_test)
        )
        # Add module symbol
        mod_fqn = module
        mod_name = module.split(".")[-1]
        self._add_symbol(
            Symbol(
                fqn=mod_fqn,
                name=mod_name,
                qualname="",
                kind="module",
                module=module,
                file=path,
                line=1,
                end_line=1,
            )
        )
        # Visit
        visitor = _ModuleVisitor(module, path)
        visitor.visit(tree)
        # Register imports
        self.modules[module].imports.update(visitor.imports)
        # Module dependency edges
        self.module_imports[module] = sorted(visitor.import_modules)
        # Record star imports for later expansion
        self.module_star_imports[module] = list(getattr(visitor, "star_imports", []))
        # Record __all__ exports
        self.modules[module].exports = list(getattr(visitor, "exports", []))
        # Register defs
        for sym in visitor.symbols:
            self._add_symbol(sym)
        # Register calls
        for caller, callee_key in visitor.calls:
            callee_fqn = self._resolve_callee(module, callee_key, visitor)
            self.calls.append((caller, callee_fqn or callee_key))
        # Collect pytest nodes if test module
        if is_test:
            rel = os.path.relpath(path, self.root)
            self.pytest_nodes_by_module[module] = self._collect_pytest_nodes(tree, rel)

    def owners_of(self, symbol: str) -> List[str]:
        fqns = self.symbols_by_name.get(symbol, [])
        return sorted(
            {os.path.relpath(self.symbols_by_fqn[f].file, self.root) for f in fqns}
        )

    def find_symbol(self, name: str) -> List[Symbol]:
        return [self.symbols_by_fqn[f] for f in self.symbols_by_name.get(name, [])]

    def defs_in(self, module: str) -> List[str]:
        mi = self.modules.get(module)
        return list(mi.defs) if mi else []

    def calls_of(self, fqn: str) -> List[str]:
        return [c for (caller, c) in self.calls if caller == fqn]

    def who_calls(self, fqn: str) -> List[str]:
        target_short = fqn.split(".")[-1]
        out: List[str] = []
        for caller, callee in self.calls:
            if callee == fqn or callee.split(".")[-1] == target_short:
                out.append(caller)
        return out

    def search_refs(self, pattern: str) -> List[Tuple[str, int, str]]:
        """Ripgrep-based raw reference search (file, line_no, text)."""
        try:
            import subprocess

            out = subprocess.check_output(["rg", "-n", pattern, self.root], text=True)
            rows: List[Tuple[str, int, str]] = []
            for line in out.splitlines():
                try:
                    fp, ln, txt = line.split(":", 2)
                    rows.append((os.path.relpath(fp, self.root), int(ln), txt))
                except Exception:
                    continue
            return rows
        except Exception:
            # Fallback: simple Python regex over indexed .py files
            rows: List[Tuple[str, int, str]] = []
            try:
                rx = re.compile(pattern)
            except Exception:
                # If pattern is not a valid regex, escape it
                rx = re.compile(re.escape(pattern))
            for fpath in self.indexed_files:
                rel = os.path.relpath(fpath, self.root)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as rf:
                        for i, ln in enumerate(rf, start=1):
                            if rx.search(ln):
                                rows.append((rel, i, ln.rstrip("\n")))
                except Exception:
                    continue
            return rows

    # --- Helpers --- #

    def module_for_file(self, path: str) -> Optional[str]:
        p = path
        if not os.path.isabs(p):
            p = os.path.abspath(os.path.join(self.root, path))
        for mod, mi in self.modules.items():
            if os.path.abspath(mi.file) == p:
                return mod
        return None

    def file_for_module(self, module: str) -> Optional[str]:
        mi = self.modules.get(module)
        return mi.file if mi else None

    def tests_for_module(self, module: str) -> List[str]:
        base = module.split(".")[0]
        out = set(self.module_to_tests.get(base, []))
        # include direct module key if present
        out.update(self.module_to_tests.get(module, []))
        return sorted(out)

    def tests_for_symbol(self, fqn: str) -> List[str]:
        mod = fqn.rsplit(".", 1)[0] if "." in fqn else fqn
        return self.tests_for_module(mod)

    def refs_of(self, fqn: str) -> List[Tuple[str, str]]:
        """Return (caller_fqn, callee_match) entries that reference fqn or its short name."""
        target_short = fqn.split(".")[-1]
        out: List[Tuple[str, str]] = []
        for caller, callee in self.calls:
            if callee == fqn or callee.split(".")[-1] == target_short:
                out.append((caller, callee))
        return out

    def export_json(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "files": [os.path.relpath(p, self.root) for p in self.indexed_files],
            "symbols": [self._sym_to_dict(s) for s in self.symbols_by_fqn.values()],
            "modules": {k: self._mi_to_dict(v) for k, v in self.modules.items()},
            "calls": self.calls,
            "module_to_tests": self.module_to_tests,
            "coverage_files": {
                os.path.relpath(k, self.root): sorted(list(v))
                for k, v in self.coverage_files.items()
            },
            "symbol_coverage": self.symbol_coverage,
            "module_imports": self.module_imports,
        }

    def export_sqlite(self, db_path: str) -> None:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS modules(module TEXT PRIMARY KEY, file TEXT, is_test INT);
            CREATE TABLE IF NOT EXISTS symbols(
              fqn TEXT PRIMARY KEY, name TEXT, qualname TEXT, kind TEXT, module TEXT,
              file TEXT, line INT, end_line INT, doc TEXT, signature TEXT, returns TEXT
            );
            CREATE TABLE IF NOT EXISTS calls(caller TEXT, callee TEXT);
            CREATE TABLE IF NOT EXISTS tests_map(module TEXT, test_module TEXT);
            CREATE TABLE IF NOT EXISTS coverage(file TEXT, line INT);
            CREATE TABLE IF NOT EXISTS mod_deps(module TEXT, dep TEXT);
            """
        )
        cur.executemany(
            "INSERT OR IGNORE INTO files(path) VALUES(?)",
            [(os.path.relpath(f, self.root),) for f in self.indexed_files],
        )
        cur.executemany(
            "INSERT OR REPLACE INTO modules(module,file,is_test) VALUES(?,?,?)",
            [
                (m, os.path.relpath(mi.file, self.root), 1 if mi.is_test else 0)
                for m, mi in self.modules.items()
            ],
        )
        cur.executemany(
            "INSERT OR REPLACE INTO symbols VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    s.fqn,
                    s.name,
                    s.qualname,
                    s.kind,
                    s.module,
                    os.path.relpath(s.file, self.root),
                    int(s.line),
                    int(s.end_line),
                    s.doc or "",
                    s.signature or "",
                    s.returns or "",
                )
                for s in self.symbols_by_fqn.values()
            ],
        )
        if self.calls:
            cur.executemany(
                "INSERT INTO calls(caller,callee) VALUES(?,?)", list(self.calls)
            )
        rows = []
        for mod, tests in self.module_to_tests.items():
            for t in tests:
                rows.append((mod, t))
        if rows:
            cur.executemany(
                "INSERT INTO tests_map(module,test_module) VALUES(?,?)", rows
            )
        cov_rows = []
        for f, lines in self.coverage_files.items():
            rel = os.path.relpath(f, self.root)
            cov_rows.extend([(rel, int(n)) for n in lines])
        if cov_rows:
            cur.executemany("INSERT INTO coverage(file,line) VALUES(?,?)", cov_rows)
        dep_rows = []
        for m, deps in self.module_imports.items():
            for d in deps:
                dep_rows.append((m, d))
        if dep_rows:
            cur.executemany("INSERT INTO mod_deps(module,dep) VALUES(?,?)", dep_rows)
        conn.commit()
        conn.close()

    def _module_name_for_path(self, path: str) -> str:
        rel = os.path.relpath(path, self.root)
        no_ext = rel[:-3] if rel.endswith(".py") else rel
        parts = no_ext.split(os.sep)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(p for p in parts if p)

    def _resolve_callee(
        self, module: str, callee_key: str, visitor: "_ModuleVisitor"
    ) -> Optional[str]:
        if "." in callee_key and ":" not in callee_key:
            return callee_key

        if ":" in callee_key:
            mod_alias, name = callee_key.split(":", 1)
            target = visitor.imports.get(mod_alias)
            if target:
                return f"{target}.{name}" if not target.endswith(f".{name}") else target

        mi = self.modules.get(module)
        if mi:
            # Prefer any def with same suffix name (matches within class or function)
            for f in mi.defs:
                if f.split(".")[-1] == callee_key:
                    return f

        tgt = visitor.imports.get(callee_key)
        if tgt:
            return tgt
        return None

    def _build_test_mapping(self) -> None:
        for mod, mi in self.modules.items():
            if not mi.is_test:
                continue
            for alias, target in mi.imports.items():
                # target may be module or module.symbol
                m = target.split(".")[0]
                self.module_to_tests.setdefault(m, []).append(mod)

    def _expand_star_imports(self) -> None:
        for mod, stars in self.module_star_imports.items():
            mi = self.modules.get(mod)
            if not mi:
                continue
            for star_mod in stars:
                defs = [
                    f
                    for f in self.modules.get(
                        star_mod, ModuleInfo(module=star_mod, file="")
                    ).defs
                ]
                exports = set(
                    self.modules.get(
                        star_mod, ModuleInfo(module=star_mod, file="")
                    ).exports
                    or []
                )
                for fqn in defs:
                    name = fqn.split(".")[-1]
                    if exports:
                        if name not in exports:
                            continue
                    elif name.startswith("_"):
                        continue
                    if name not in mi.imports:
                        mi.imports[name] = f"{star_mod}.{name}"

    def _post_resolve_calls(self) -> None:
        # After imports expanded, try to resolve unresolved simple names
        new_calls: List[Tuple[str, str]] = []
        for caller, callee in self.calls:
            if "." in callee:
                new_calls.append((caller, callee))
                continue
            # Find caller module
            caller_mod = caller.rsplit(".", 1)[0] if "." in caller else caller
            imports = self.modules.get(
                caller_mod, ModuleInfo(module=caller_mod, file="")
            ).imports
            tgt = imports.get(callee)
            if tgt:
                new_calls.append((caller, tgt))
            else:
                # leave as-is
                new_calls.append((caller, callee))
        self.calls = new_calls

    def unresolved_calls(self) -> List[Tuple[str, str]]:
        return [
            (a, c)
            for (a, c) in self.calls
            if "." not in c and not self._is_builtin_name(c)
        ]

    def _collect_pytest_nodes(self, tree: ast.AST, rel_path: str) -> List[str]:
        nodes: List[str] = []
        # top-level test_* functions
        for n in getattr(tree, "body", []) or []:
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
                nodes.extend(self._expand_parametrize(rel_path, None, n))
            if isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
                cls = n.name
                for m in getattr(n, "body", []) or []:
                    if isinstance(m, ast.FunctionDef) and m.name.startswith("test_"):
                        nodes.extend(self._expand_parametrize(rel_path, cls, m))
        return nodes

    def _expand_parametrize(
        self, rel_path: str, cls: Optional[str], fn: ast.FunctionDef
    ) -> List[str]:
        base = f"{rel_path}::" + (f"{cls}::" if cls else "") + fn.name
        # Look for @pytest.mark.parametrize("arg", [vals])
        total: List[str] = []
        params: List[int] = []
        try:
            for dec in getattr(fn, "decorator_list", []) or []:
                # pytest.mark.parametrize(...)
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "parametrize"
                ):
                    # estimate number of cases from second arg list length
                    if len(dec.args) >= 2 and isinstance(
                        dec.args[1], (ast.List, ast.Tuple)
                    ):
                        params.append(len(dec.args[1].elts))
        except Exception:
            pass
        if params:
            count: int = 1
            for k in params:
                try:
                    count *= int(k)
                except Exception:
                    count = max(count, 1)
            for i in range(count):
                total.append(f"{base}[{i}]")
            return total
        return [base]

    # --- Cache --- #

    def _try_load_cache(self, cache_path: str) -> bool:
        try:
            if not os.path.exists(cache_path):
                return False
            data = json.loads(open(cache_path, "r", encoding="utf-8").read())
            if str(data.get("version", "")) != "3":
                return False
            # Verify mtimes and hashes
            files = data.get("indexed_files", [])
            mt = data.get("mtimes", {})
            hh = data.get("hashes", {})
            for f in files:
                if not os.path.exists(f):
                    return False
                if int(os.path.getmtime(f)) != int(mt.get(f, 0)):
                    return False
                if self._file_hash(f) != str(hh.get(f, "")):
                    return False
            # Load
            self.indexed_files = files
            for s in data.get("symbols", []):
                sym = Symbol(**s)
                self._add_symbol(sym)
            for mod, mi in data.get("modules", {}).items():
                self.modules[mod] = ModuleInfo(**mi)
            self.calls = [tuple(x) for x in data.get("calls", [])]
            self.module_to_tests = data.get("module_to_tests", {})
            self.module_imports = data.get("module_imports", {})
            self._cached_mtimes = {k: int(v) for k, v in (mt or {}).items()}
            self._cached_hashes = {k: str(v) for k, v in (hh or {}).items()}
            return True
        except Exception:
            return False

    def _load_cache_relaxed(self, cache_path: str) -> bool:
        try:
            if not os.path.exists(cache_path):
                return False
            data = json.loads(open(cache_path, "r", encoding="utf-8").read())
            if str(data.get("version", "")) != "3":
                return False
            self.indexed_files = data.get("indexed_files", [])
            for s in data.get("symbols", []):
                sym = Symbol(**s)
                self._add_symbol(sym)
            for mod, mi in data.get("modules", {}).items():
                self.modules[mod] = ModuleInfo(**mi)
            self.calls = [tuple(x) for x in data.get("calls", [])]
            self.module_to_tests = data.get("module_to_tests", {})
            self.module_imports = data.get("module_imports", {})
            self._cached_mtimes = {
                k: int(v) for k, v in (data.get("mtimes", {}) or {}).items()
            }
            self._cached_hashes = {
                k: str(v) for k, v in (data.get("hashes", {}) or {}).items()
            }
            return True
        except Exception:
            return False

    def _detect_changed_files(
        self, old_mt: Dict[str, int], old_hh: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        curr_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dir_rel = os.path.relpath(dirpath, self.root)
            dirnames[:] = [d for d in dirnames if not self._is_ignored(os.path.join(dir_rel, d))]
            if self._is_ignored(dir_rel):
                continue
            for fn in filenames:
                if fn.endswith(".py"):
                    fp = os.path.join(dirpath, fn)
                    if self._is_ignored(os.path.relpath(fp, self.root)):
                        continue
                    curr_files.append(fp)
        curr = set(curr_files)
        prev = set(self.indexed_files or [])
        removed = list(prev - curr)
        added = list(curr - prev)
        changed: List[str] = list(added)
        for f in curr & prev:
            try:
                mt = int(os.path.getmtime(f))
                hh = self._file_hash(f)
            except Exception:
                changed.append(f)
                continue
            if old_mt.get(f) != mt or old_hh.get(f) != hh:
                changed.append(f)
        self.indexed_files = sorted(list(curr))
        return sorted(set(changed)), sorted(removed)

    def _incremental_reindex(
        self, changed_files: List[str], removed_files: List[str]
    ) -> None:
        # purge removed modules
        for f in removed_files:
            m = self._module_name_for_path(f)
            if m in self.modules:
                to_remove = [
                    fqn for fqn, s in list(self.symbols_by_fqn.items()) if s.module == m
                ]
                for fqn in to_remove:
                    s = self.symbols_by_fqn.pop(fqn, None)
                    if s:
                        self.symbols_by_name[s.name] = [
                            x for x in self.symbols_by_name.get(s.name, []) if x != fqn
                        ]
                self.calls = [
                    (a, b) for (a, b) in self.calls if not a.startswith(m + ".")
                ]
                self.modules.pop(m, None)
                self.module_imports.pop(m, None)
                self.module_star_imports.pop(m, None)
        mods = {self._module_name_for_path(f) for f in changed_files}
        # include dependents via reverse import graph
        rev = self._reverse_imports()
        queue = list(mods)
        seen = set(mods)
        while queue:
            m = queue.pop(0)
            for dep in rev.get(m, []):
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
        for m in seen:
            self._reindex_module(m)

    def _reverse_imports(self) -> Dict[str, List[str]]:
        rev: Dict[str, List[str]] = {}
        for m, deps in self.module_imports.items():
            for d in deps:
                rev.setdefault(d, []).append(m)
        return rev

    def _reindex_module(self, module: str) -> None:
        mi = self.modules.get(module)
        if not mi:
            return
        # remove existing symbols and calls for this module
        to_remove = [
            fqn for fqn, s in self.symbols_by_fqn.items() if s.module == module
        ]
        for fqn in to_remove:
            s = self.symbols_by_fqn.pop(fqn, None)
            if s:
                lst = self.symbols_by_name.get(s.name, [])
                self.symbols_by_name[s.name] = [x for x in lst if x != fqn]
        self.calls = [(a, b) for (a, b) in self.calls if not a.startswith(module + ".")]
        # reset import maps for this module
        self.modules[module].imports = {}
        self.module_imports[module] = []
        self.module_star_imports[module] = []
        # re-parse
        try:
            src = open(mi.file, "r", encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            return
        visitor = _ModuleVisitor(module, mi.file)
        visitor.visit(tree)
        self.modules[module].imports.update(visitor.imports)
        self.module_imports[module] = sorted(visitor.import_modules)
        self.module_star_imports[module] = list(getattr(visitor, "star_imports", []))
        self.modules[module].exports = list(getattr(visitor, "exports", []))
        for sym in visitor.symbols:
            self._add_symbol(sym)
        for caller, callee_key in visitor.calls:
            callee_fqn = self._resolve_callee(module, callee_key, visitor)
            self.calls.append((caller, callee_fqn or callee_key))

    def _save_cache(self, cache_path: str) -> None:
        try:
            mt = {f: int(os.path.getmtime(f)) for f in self.indexed_files}
            hh = {f: self._file_hash(f) for f in self.indexed_files}
            data = {
                "version": "3",
                "indexed_files": self.indexed_files,
                "mtimes": mt,
                "hashes": hh,
                "symbols": [self._sym_to_dict(s) for s in self.symbols_by_fqn.values()],
                "modules": {k: self._mi_to_dict(v) for k, v in self.modules.items()},
                "calls": self.calls,
                "module_to_tests": self.module_to_tests,
                "module_imports": self.module_imports,
            }
            open(cache_path, "w", encoding="utf-8").write(json.dumps(data))
        except Exception:
            pass

    def _sym_to_dict(self, s: Symbol) -> Dict[str, Any]:
        return {
            "fqn": s.fqn,
            "name": s.name,
            "qualname": s.qualname,
            "kind": s.kind,
            "module": s.module,
            "file": s.file,
            "line": s.line,
            "end_line": s.end_line,
            "doc": s.doc,
            "signature": s.signature,
            "returns": s.returns,
        }

    def _mi_to_dict(self, mi: ModuleInfo) -> Dict[str, Any]:
        return {
            "module": mi.module,
            "file": mi.file,
            "is_test": mi.is_test,
            "imports": mi.imports,
            "defs": mi.defs,
            "exports": mi.exports,
        }

    def _is_builtin_name(self, name: str) -> bool:
        try:
            import builtins as _bi  # type: ignore

            if hasattr(_bi, name):
                return True
        except Exception:
            pass
        return name in {
            "super",
            "property",
            "globals",
            "locals",
            "__import__",
            "print",
            "len",
            "range",
            "dict",
            "list",
            "set",
            "tuple",
            "int",
            "float",
            "bool",
            "max",
            "min",
            "sum",
            "open",
            "enumerate",
            "zip",
            "map",
            "filter",
            "round",
            "any",
            "all",
            "sorted",
            "hasattr",
            "getattr",
            "setattr",
            "isinstance",
            "issubclass",
        }

    def _file_hash(self, path: str) -> str:
        try:
            import hashlib

            with open(path, "rb") as rf:
                return hashlib.sha1(rf.read()).hexdigest()
        except Exception:
            return ""

    # --- Coverage --- #

    def attach_coverage_from_xml(self, xml_path: str) -> None:
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(xml_path)
            root = tree.getroot()
            files_hits: Dict[str, set[int]] = {}
            # coverage.py XML: <class filename="path"> ... <lines><line number="N" hits="H"/></lines>
            for cls in root.findall(".//class"):
                fn = cls.attrib.get("filename", "")
                if not fn:
                    continue
                # Normalize to absolute path
                f_abs = (
                    fn
                    if os.path.isabs(fn)
                    else os.path.abspath(os.path.join(self.root, fn))
                )
                hits = files_hits.setdefault(f_abs, set())
                for ln in cls.findall(".//line"):
                    try:
                        num = int(ln.attrib.get("number", "0"))
                        h = int(ln.attrib.get("hits", "0"))
                        if h > 0:
                            hits.add(num)
                    except Exception:
                        continue
            # Some coverage.xml variants place <file> nodes
            if not files_hits:
                for fnode in root.findall(".//file"):
                    fn = fnode.attrib.get("filename", "")
                    if not fn:
                        continue
                    f_abs = (
                        fn
                        if os.path.isabs(fn)
                        else os.path.abspath(os.path.join(self.root, fn))
                    )
                    hits = files_hits.setdefault(f_abs, set())
                    for ln in fnode.findall(".//line"):
                        try:
                            num = int(ln.attrib.get("number", "0"))
                            h = int(ln.attrib.get("hits", "0"))
                            if h > 0:
                                hits.add(num)
                        except Exception:
                            continue
            self.coverage_files = files_hits
            # Compute per-symbol coverage
            sym_cov: Dict[str, float] = {}
            for fqn, sym in self.symbols_by_fqn.items():
                covered = files_hits.get(sym.file, set())
                a = int(sym.line)
                b = int(sym.end_line) if int(sym.end_line) >= a else a
                span = list(range(a, b + 1))
                if not span:
                    sym_cov[fqn] = 0.0
                    continue
                hits = sum(1 for x in span if x in covered)
                sym_cov[fqn] = hits / float(len(span))
            self.symbol_coverage = sym_cov
        except Exception:
            # Leave coverage empty on error
            self.coverage_files = {}
            self.symbol_coverage = {}

    def coverage_of(self, fqn: str) -> Optional[float]:
        return self.symbol_coverage.get(fqn)


def _cli() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="./repo")
    p.add_argument("--ignore", action="append", default=None, help="Relative paths to ignore (repeatable)")
    p.add_argument("--owners-of", dest="owners_of", default=None)
    p.add_argument("--search", dest="search", default=None)
    p.add_argument("--defs-in", dest="defs_in", default=None)
    p.add_argument("--calls-of", dest="calls_of", default=None)
    p.add_argument("--who-calls", dest="who_calls", default=None)
    p.add_argument("--dump", dest="dump", action="store_true")
    p.add_argument("--coverage-xml", dest="coverage_xml", default=None)
    p.add_argument("--coverage-of", dest="coverage_of", default=None)
    p.add_argument("--refs-of", dest="refs_of", default=None)
    p.add_argument("--tests-for", dest="tests_for", default=None)
    p.add_argument("--tests-for-module", dest="tests_for_module", default=None)
    p.add_argument("--export", dest="export", default=None)
    p.add_argument("--no-cache", dest="no_cache", action="store_true")
    p.add_argument("--export-sqlite", dest="export_sqlite", default=None)
    p.add_argument("--pytest-nodes", dest="pytest_nodes", default=None)
    p.add_argument("--module-deps", dest="module_deps", default=None)
    p.add_argument("--unresolved", dest="unresolved", action="store_true")
    args = p.parse_args()
    g = CodeGraph.load_or_build(args.root, ignore_cache=bool(args.no_cache), ignore=[s for s in (args.ignore or []) if s])
    if args.coverage_xml:
        g.attach_coverage_from_xml(args.coverage_xml)
        # fall through to other queries if provided
    if args.owners_of:
        print(json.dumps(g.owners_of(args.owners_of)))
        return
    if args.search:
        print(json.dumps(g.search_refs(args.search)))
        return
    if args.defs_in:
        print(json.dumps(g.defs_in(args.defs_in)))
        return
    if args.calls_of:
        print(json.dumps(g.calls_of(args.calls_of)))
        return
    if args.who_calls:
        print(json.dumps(g.who_calls(args.who_calls)))
        return
    if args.coverage_of:
        print(json.dumps(g.coverage_of(args.coverage_of)))
        return
    if args.refs_of:
        print(json.dumps(g.refs_of(args.refs_of)))
        return
    if args.tests_for:
        print(json.dumps(g.tests_for_symbol(args.tests_for)))
        return
    if args.tests_for_module:
        print(json.dumps(g.tests_for_module(args.tests_for_module)))
        return
    if args.export:
        obj = g.export_json()
        if args.export == "-":
            print(json.dumps(obj))
        else:
            open(args.export, "w", encoding="utf-8").write(json.dumps(obj))
            print(args.export)
        return
    if args.export_sqlite:
        g.export_sqlite(args.export_sqlite)
        print(args.export_sqlite)
        return
    if args.pytest_nodes:
        mod = args.pytest_nodes
        print(json.dumps(g.pytest_nodes_by_module.get(mod, [])))
        return
    if args.module_deps:
        print(json.dumps(g.module_imports.get(args.module_deps, [])))
        return
    if args.unresolved:
        print(json.dumps(g.unresolved_calls()))
        return
    if args.dump:
        print(
            json.dumps(
                {
                    "files": len(g.indexed_files),
                    "symbols": len(g.symbols_by_fqn),
                    "modules": len(g.modules),
                    "calls": len(g.calls),
                    "coverage_files": len(g.coverage_files),
                }
            )
        )
        return
    # Dump summary
    print(json.dumps({"files": len(g.indexed_files), "symbols": len(g.symbols_by_fqn)}))


class _ModuleVisitor(ast.NodeVisitor):
    def __init__(self, module: str, path: str) -> None:
        self.module = module
        self.path = path
        self.symbols: List[Symbol] = []
        self.calls: List[Tuple[str, str]] = []  # (caller_fqn, callee_key)
        self.stack: List[str] = []  # qualname stack
        self.class_stack: List[str] = []
        self.imports: Dict[str, str] = {}
        self.import_modules: List[str] = []
        self.star_imports: List[str] = []
        self.exports: List[str] = []

    def _cur_qualname(self) -> str:
        return ".".join(self.stack)

    def _cur_class(self) -> Optional[str]:
        return self.class_stack[-1] if self.class_stack else None

    def _fqn(self, name: str) -> str:
        q = self._cur_qualname()
        return f"{self.module}.{q + ('.' if q else '')}{name}"

    def visit_Import(self, node: ast.Import) -> Any:  # type: ignore[override]
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[-1]
            self.imports[asname] = alias.name
            self.import_modules.append(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:  # type: ignore[override]
        # Resolve relative imports: from .x import y
        if node.level and node.module:
            base = self.module.split(".")
            up = max(0, int(node.level))
            prefix = base[:-up] if up > 0 else base
            mod = ".".join([p for p in prefix if p] + [node.module])
        elif node.level and not node.module:
            base = self.module.split(".")
            up = max(0, int(node.level))
            mod = ".".join(base[:-up])
        else:
            mod = node.module or ""
        for alias in node.names:
            # star import
            if getattr(alias, "name", "") == "*":
                if mod:
                    self.star_imports.append(mod)
                continue
            asname = alias.asname or alias.name
            self.imports[asname] = f"{mod}.{alias.name}" if mod else alias.name
        if mod:
            self.import_modules.append(mod.split(".")[0])
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # type: ignore[override]
        fqn = self._fqn(node.name)
        try:
            doc_s = ast.get_docstring(node) or None
        except Exception:
            doc_s = None
        sym = Symbol(
            fqn=fqn,
            name=node.name,
            qualname=self._cur_qualname(),
            kind="class",
            module=self.module,
            file=self.path,
            line=getattr(node, "lineno", 1),
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            doc=doc_s,
        )
        self.symbols.append(sym)
        self.stack.append(node.name)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
        self._visit_func_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # type: ignore[override]
        self._visit_func_like(node)

    def _visit_func_like(self, node: Any) -> None:
        fqn = self._fqn(node.name)
        # Signature & returns
        sig_s, ret_s = None, None
        try:
            params = []
            for a in getattr(node, "args", None).args or []:
                nm = getattr(a, "arg", "")
                ann = getattr(a, "annotation", None)
                params.append(f"{nm}:{ast.unparse(ann)}" if ann is not None else nm)
            ret = getattr(node, "returns", None)
            ret_s = ast.unparse(ret) if ret is not None else None
            sig_s = f"({', '.join(params)})"
        except Exception:
            sig_s, ret_s = None, None
        try:
            doc_s = ast.get_docstring(node) or None
        except Exception:
            doc_s = None
        sym = Symbol(
            fqn=fqn,
            name=node.name,
            qualname=self._cur_qualname(),
            kind="function",
            module=self.module,
            file=self.path,
            line=getattr(node, "lineno", 1),
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            doc=doc_s,
            signature=sig_s,
            returns=ret_s,
        )
        self.symbols.append(sym)
        self.stack.append(node.name)
        # Traverse body to collect calls
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee_key = self._extract_callee_key(sub.func)
                if callee_key:
                    self.calls.append((fqn, callee_key))
        # Decorators as calls
        for dec in getattr(node, "decorator_list", []) or []:
            callee_key = self._extract_callee_key(dec)
            if callee_key:
                self.calls.append((fqn, callee_key))
        self.stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:  # type: ignore[override]
        for t in getattr(node, "targets", []) or []:
            if isinstance(t, ast.Name):
                fqn = self._fqn(t.id)
                sym = Symbol(
                    fqn=fqn,
                    name=t.id,
                    qualname=self._cur_qualname(),
                    kind="variable",
                    module=self.module,
                    file=self.path,
                    line=getattr(node, "lineno", 1),
                    end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                )
                self.symbols.append(sym)
        # capture __all__ = ["..."]
        try:
            names = []
            is_all = any(
                (isinstance(t, ast.Name) and t.id == "__all__") for t in node.targets
            )
            if is_all and isinstance(node.value, (ast.List, ast.Tuple)):
                for el in node.value.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        names.append(el.value)
            if names:
                self.exports.extend(names)
        except Exception:
            pass
        self.generic_visit(node)

    def _extract_callee_key(self, fn: ast.AST) -> Optional[str]:
        # simple name
        if isinstance(fn, ast.Name):
            return fn.id

        # super().method()
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Call)
            and isinstance(fn.value.func, ast.Name)
            and fn.value.func.id == "super"
        ):
            meth = fn.attr
            cur_cls = self._cur_class()
            if cur_cls:
                return f"{self.module}.{cur_cls}.{meth}"
            return meth

        # obj.attr chain
        if isinstance(fn, ast.Attribute):
            parts: List[str] = []
            cur = fn
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            parts.reverse()

            if isinstance(cur, ast.Name):
                base = cur.id
                if base in ("self", "cls"):
                    cur_cls = self._cur_class()
                    if cur_cls and parts:
                        return f"{self.module}.{cur_cls}.{parts[-1]}"
                    return f"{self.module}.{cur_cls}" if cur_cls else parts[-1]
                if base in self.imports:
                    return f"{base}:{parts[-1]}" if parts else base
                return f"{self.module}.{base}.{parts[-1]}" if parts else base
        # getattr(module, "name") heuristic
        if (
            isinstance(fn, ast.Call)
            and isinstance(fn.func, ast.Name)
            and fn.func.id == "getattr"
            and fn.args
            and len(fn.args) >= 2
            and isinstance(fn.args[0], ast.Name)
            and isinstance(fn.args[1], ast.Constant)
            and isinstance(fn.args[1].value, str)
        ):
            base = fn.args[0].id
            name = fn.args[1].value
            if base in self.imports:
                return f"{self.imports[base]}.{name}"
            cur_cls = self._cur_class()
            if base in ("self", "cls") and cur_cls:
                return f"{self.module}.{cur_cls}.{name}"
            return f"{self.module}.{base}.{name}"
        # importlib.import_module("pkg.mod") heuristic
        if (
            isinstance(fn, ast.Call)
            and isinstance(fn.func, ast.Attribute)
            and isinstance(fn.func.value, ast.Name)
            and fn.func.value.id == "importlib"
            and fn.func.attr == "import_module"
            and fn.args
            and isinstance(fn.args[0], ast.Constant)
            and isinstance(fn.args[0].value, str)
        ):
            mod = str(fn.args[0].value)
            return mod
        return None


if __name__ == "__main__":
    _cli()
