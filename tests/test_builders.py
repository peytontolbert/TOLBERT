import csv
import json
from pathlib import Path
import sys
from typing import Dict

# Ensure project root is on sys.path so we import the local `scripts` package
# instead of any third-party package named `scripts`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import (
    build_wos_spans,
    build_researchhierarchy_spans,
    build_codehierarchy_spans,
    build_joint_code_paper_tol,
)


def test_build_wos_spans_helpers(tmp_path: Path):
    """
    Smoke test for build_wos_spans:
      - load_wos_csv parses a small CSV,
      - build_ontology assigns ids at each level,
      - build_spans produces node_path with expected structure.
    """
    csv_path = tmp_path / "wos_small.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "level1", "level2", "level3"])
        writer.writeheader()
        writer.writerow(
            {
                "text": "doc about machine learning",
                "level1": "Computer Science",
                "level2": "Artificial Intelligence",
                "level3": "Machine Learning",
            }
        )
        writer.writerow(
            {
                "text": "doc about physics",
                "level1": "Physics",
                "level2": "Quantum",
                "level3": "Quantum Mechanics",
            }
        )

    rows = build_wos_spans.load_wos_csv(
        path=csv_path,
        text_col="text",
        l1_col="level1",
        l2_col="level2",
        l3_col="level3",
    )
    assert len(rows) == 2

    l1_ids, l2_ids, l3_ids = build_wos_spans.build_ontology(rows)
    assert len(l1_ids) == 2
    assert len(l2_ids) == 2
    assert len(l3_ids) == 2

    spans = build_wos_spans.build_spans(rows, l1_ids, l2_ids, l3_ids)
    assert len(spans) == 2
    for rec in spans:
        path = rec["node_path"]
        # Root + three levels if all labels are present.
        assert isinstance(path, list)
        assert len(path) == 4
        assert path[0] == 0


def test_build_researchhierarchy_spans_helpers(tmp_path: Path):
    """
    Smoke test for build_researchhierarchy_spans helpers:
      - load_metadata parses a small CSV,
      - build_ontology assigns ids,
      - build_span_records produces the expected node_path layout.
    """
    csv_path = tmp_path / "research_small.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doc_id", "field", "subfield", "discipline", "text"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doc_id": "D1",
                "field": "Computer Science",
                "subfield": "AI",
                "discipline": "ML",
                "text": "title and abstract about ML",
            }
        )

    metas = build_researchhierarchy_spans.load_metadata(
        path=csv_path,
        id_col="doc_id",
        field_col="field",
        subfield_col="subfield",
        discipline_col="discipline",
        text_col="text",
        pdf_path_col=None,
        source_col=None,
    )
    assert len(metas) == 1

    field_ids, subfield_ids, discipline_ids = build_researchhierarchy_spans.build_ontology(
        metas
    )
    assert len(field_ids) == 1
    assert len(subfield_ids) == 1
    assert len(discipline_ids) == 1

    spans = build_researchhierarchy_spans.build_span_records(
        metas, field_ids, subfield_ids, discipline_ids
    )
    assert len(spans) == 1
    rec = spans[0]
    path = rec["node_path"]
    assert isinstance(path, list)
    # root + field + subfield + discipline
    assert len(path) == 4
    assert path[0] == 0


def test_build_codehierarchy_spans_helpers(tmp_path: Path):
    """
    Smoke test for build_codehierarchy_spans helpers:
      - load_metadata parses a small metadata file,
      - build_ontology assigns ids for languages, categories, repos,
      - build_span_records produces spans with expected node_path layout.
    """
    # Create a fake repos_root with two repos and one file each.
    repos_root = tmp_path / "repos"
    repo1 = repos_root / "repo1"
    repo2 = repos_root / "repo2"
    repo1.mkdir(parents=True)
    repo2.mkdir(parents=True)
    (repo1 / "file1.py").write_text("print('hello')\n", encoding="utf-8")
    (repo2 / "file2.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    meta_path = tmp_path / "meta.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repo_name", "language", "category"])
        writer.writeheader()
        writer.writerow({"repo_name": "repo1", "language": "Python", "category": "ML"})
        writer.writerow({"repo_name": "repo2", "language": "C++", "category": "Systems"})

    metas = build_codehierarchy_spans.load_metadata(meta_path)
    assert len(metas) == 2

    lang_ids, cat_ids, repo_ids = build_codehierarchy_spans.build_ontology(metas)
    assert len(lang_ids) == 2
    assert len(cat_ids) == 2
    assert len(repo_ids) == 2

    spans = build_codehierarchy_spans.build_span_records(
        repos_root=repos_root,
        metas=metas,
        lang_ids=lang_ids,
        cat_ids=cat_ids,
        repo_ids=repo_ids,
    )
    # One span per file.
    assert len(spans) == 2
    for rec in spans:
        path = rec["node_path"]
        assert isinstance(path, list)
        # root + language + category + repo
        assert len(path) == 4
        assert path[0] == 0


def test_build_joint_code_paper_tol_helpers(tmp_path: Path):
    """
    Smoke test for build_joint_code_paper_tol core helpers:
      - _build_joint_ontology merges two small ontologies,
      - _rewrite_spans remaps node_path ids into the joint space,
      - _write_level_sizes produces a dict[level -> count] excluding root.
    """
    # Build minimal code and paper nodes.jsonl
    code_nodes_path = tmp_path / "code_nodes.jsonl"
    paper_nodes_path = tmp_path / "paper_nodes.jsonl"

    def _write_nodes(path: Path, levels: Dict[int, int]) -> None:
        with path.open("w", encoding="utf-8") as f:
            # local root
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
            nid = 1
            for lvl, count in levels.items():
                for i in range(count):
                    parent_id = 0 if lvl == 1 else nid - 1
                    f.write(
                        json.dumps(
                            {
                                "node_id": nid,
                                "level": lvl,
                                "type": f"lvl{lvl}",
                                "parent_id": parent_id,
                                "name": f"n{lvl}_{i}",
                                "attributes": {},
                            }
                        )
                        + "\n"
                    )
                    nid += 1

    _write_nodes(code_nodes_path, {1: 1, 2: 1})
    _write_nodes(paper_nodes_path, {1: 1, 2: 1})

    code_nodes = build_joint_code_paper_tol._load_nodes(code_nodes_path)
    paper_nodes = build_joint_code_paper_tol._load_nodes(paper_nodes_path)

    joint_nodes, code_map, paper_map, level_counts = build_joint_code_paper_tol._build_joint_ontology(
        code_nodes=code_nodes,
        paper_nodes=paper_nodes,
        code_domain_name="Code",
        paper_domain_name="Papers",
    )
    # Root + 2 domain nodes + remapped nodes from both ontologies.
    assert 0 in joint_nodes
    assert level_counts[1] == 2  # Code, Papers
    # There should be some nodes at deeper levels.
    assert any(lvl > 1 for lvl in level_counts.keys())
    assert code_map and paper_map

    # Create minimal spans that refer to old node ids and ensure they get remapped.
    code_spans_path = tmp_path / "code_spans.jsonl"
    paper_spans_path = tmp_path / "paper_spans.jsonl"
    with code_spans_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"span_id": "c1", "text": "code", "node_path": [0, 1, 2]}) + "\n")
    with paper_spans_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"span_id": "p1", "text": "paper", "node_path": [0, 1, 2]}) + "\n")

    # Use internal helpers directly.
    code_domain_id = 1  # first domain under root in joint ontology
    paper_domain_id = 2  # second domain
    remapped_code_spans = build_joint_code_paper_tol._rewrite_spans(
        code_spans_path, domain_node_id=code_domain_id, id_map=code_map
    )
    remapped_paper_spans = build_joint_code_paper_tol._rewrite_spans(
        paper_spans_path, domain_node_id=paper_domain_id, id_map=paper_map
    )

    assert remapped_code_spans[0]["node_path"][0] == 0
    assert remapped_code_spans[0]["node_path"][1] == code_domain_id
    assert remapped_paper_spans[0]["node_path"][0] == 0
    assert remapped_paper_spans[0]["node_path"][1] == paper_domain_id

    # Check level_sizes writing.
    level_sizes_path = tmp_path / "level_sizes.json"
    build_joint_code_paper_tol._write_level_sizes(level_counts, level_sizes_path)
    contents = json.loads(level_sizes_path.read_text(encoding="utf-8"))
    assert "level_sizes" in contents
    # Root (0) should not be present.
    assert 0 not in contents["level_sizes"]



