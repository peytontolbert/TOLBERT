from pathlib import Path
import sys
from typing import List

import torch

# Ensure project root is on sys.path so we import the local `scripts` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_repo_tree_of_life


class _DummySpan:
    def __init__(self, start_line: int, end_line: int) -> None:
        self.start_line = start_line
        self.end_line = end_line


class _DummyEntity:
    def __init__(self, kind: str, rel_path: str, labels: List[str] | None = None):
        self.kind = kind
        self.attributes = {"rel_path": rel_path, "name": rel_path}
        self.labels = labels or []
        self.id = rel_path
        self.artifact_uri = f"program://dummy/artifact/{rel_path}"
        self.span = _DummySpan(1, 10)


class _DummyRepoGraph:
    """
    Minimal stand-in for RepoGraph.entities() to exercise build_tree_for_repo.
    """

    def __init__(self, root: str) -> None:
        self.root = root

    def entities(self):
        # One Python file and one C++ file, plus a function symbol in the Python file.
        return [
            _DummyEntity("file", "pkg/file1.py", labels=["lang:python"]),
            _DummyEntity("file", "pkg/file2.cpp", labels=["lang:cpp"]),
            _DummyEntity("function", "pkg/file1.py::fn", labels=[]),
        ]


def test_build_tree_for_repo_with_dummy_graph(tmp_path: Path, monkeypatch):
    """
    Smoke test for build_tree_for_repo:
      - uses a dummy RepoGraph to avoid filesystem scanning,
      - verifies that nodes, edges, and spans have the expected basic structure.
    """
    # Patch RepoGraph used inside build_repo_tree_of_life to our dummy.
    monkeypatch.setattr(build_repo_tree_of_life, "RepoGraph", _DummyRepoGraph)

    repo_root = tmp_path / "repo"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "file1.py").write_text("print('hello')\n", encoding="utf-8")
    (repo_root / "pkg" / "file2.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    nodes, edges, spans = build_repo_tree_of_life.build_tree_for_repo(repo_root)

    # Basic sanity checks.
    assert nodes
    assert edges
    assert spans

    # There should be a root node at level 0.
    root_nodes = [n for n in nodes if n.get("level") == 0 and n.get("type") == "root"]
    assert len(root_nodes) == 1

    # There should be language nodes at level 1 and a repo node at level 2.
    lang_nodes = [n for n in nodes if n.get("level") == 1 and n.get("type") == "language"]
    assert lang_nodes
    repo_nodes = [n for n in nodes if n.get("level") == 2 and n.get("type") == "repo"]
    assert len(repo_nodes) == 1

    # Spans should have a node_path starting at root and including the repo node id.
    repo_id = repo_nodes[0]["node_id"]
    for s in spans:
        path = s["node_path"]
        assert isinstance(path, list)
        assert path[0] == root_nodes[0]["node_id"]
        assert repo_id in path


