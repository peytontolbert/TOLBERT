import json
from pathlib import Path
import sys

import torch

# Ensure project root is on sys.path so local `tolbert` and `scripts` packages
# are importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tolbert.data import TreeOfLifeDataset, collate_tree_of_life_batch
from scripts import eval_retrieval


def test_tree_of_life_dataset_and_collate_single_path(tmp_path: Path):
    """
    Verify that TreeOfLifeDataset:
      - reads node_path into per-level targets,
      - produces paths suitable for contrastive / path losses,
      - and that collate_tree_of_life_batch produces the expected tensors.
    """
    spans_path = tmp_path / "spans_single.jsonl"
    records = [
        {
            "span_id": "s1",
            "text": "hello world",
            "node_path": [0, 1, 3],
        },
        {
            "span_id": "s2",
            "text": "another span",
            "node_path": [0, 2, 4],
        },
    ]
    with spans_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    class _DummyTokenizer:
        def __init__(self):
            self.vocab_size = 100
            # Provide a MASK token id for MLM replacement logic.
            self.mask_token_id = 99

        def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
            return [0] * len(ids)

        def __call__(self, text, return_tensors, truncation, padding, max_length):
            # Simple fixed-length encoding: map each char to an integer.
            ids = list(range(1, min(len(text) + 1, max_length - 2)))
            # Pad / truncate to max_length.
            ids = ids + [0] * (max_length - len(ids))
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attention_mask = (input_ids != 0).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer = _DummyTokenizer()
    ds = TreeOfLifeDataset(
        spans_file=str(spans_path),
        tokenizer=tokenizer,  # type: ignore[arg-type]
        max_length=8,
        mask_probability=0.15,
    )

    assert len(ds) == 2
    sample0 = ds[0]
    assert "input_ids" in sample0 and "labels_mlm" in sample0
    assert "level_targets" in sample0
    assert "paths" in sample0
    # Level targets should have entries for levels 1 and 2.
    assert sample0["level_targets"][1] == 1
    assert sample0["level_targets"][2] == 3

    batch = collate_tree_of_life_batch([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert set(batch["level_targets"].keys()) == {1, 2}
    # Paths should be a list of node_path lists (wrapped in an extra list per example).
    assert "paths" in batch
    assert batch["paths"][0] == [[0, 1, 3]]
    assert batch["paths"][1] == [[0, 2, 4]]


def test_tree_of_life_dataset_with_node_paths_multi(tmp_path: Path):
    """
    Verify that when `node_paths` (multiple valid paths) is provided:
      - the first path is used as canonical `level_targets`,
      - all paths are preserved in the `paths` field for DAG-aware losses.
    """
    spans_path = tmp_path / "spans_multi.jsonl"
    records = [
        {
            "span_id": "s1",
            "text": "multi path span",
            "node_paths": [
                [0, 1, 3],
                [0, 1, 4],
            ],
        },
    ]
    with spans_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    class _DummyTokenizer:
        def __init__(self):
            self.vocab_size = 100
            self.mask_token_id = 99

        def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
            return [0] * len(ids)

        def __call__(self, text, return_tensors, truncation, padding, max_length):
            ids = list(range(1, min(len(text) + 1, max_length - 2)))
            ids = ids + [0] * (max_length - len(ids))
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attention_mask = (input_ids != 0).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer = _DummyTokenizer()
    ds = TreeOfLifeDataset(
        spans_file=str(spans_path),
        tokenizer=tokenizer,  # type: ignore[arg-type]
        max_length=8,
        mask_probability=0.15,
    )

    assert len(ds) == 1
    sample = ds[0]
    # Canonical targets come from the first path [0,1,3].
    assert sample["level_targets"][1] == 1
    assert sample["level_targets"][2] == 3
    # All paths are preserved.
    assert sample["paths"] == [[0, 1, 3], [0, 1, 4]]


def test_eval_retrieval_helpers_basic():
    """
    Sanity-check compute_relevance_mask, eval_retrieval, and
    compute_branch_consistency_at_k on a tiny toy example.
    """
    # Two index items, two queries.
    index_embs = torch.eye(2, dtype=torch.float32)
    query_embs = torch.eye(2, dtype=torch.float32)

    # Paths: query 0 matches index 0 at level 1; query 1 matches index 1.
    index_paths = [[0, 1], [0, 2]]
    query_paths = [[0, 1], [0, 2]]

    rel = eval_retrieval.compute_relevance_mask(
        query_paths=query_paths,
        index_paths=index_paths,
        min_level=1,
    )
    assert rel == [[True, False], [False, True]]

    mrr, p_at_1 = eval_retrieval.eval_retrieval(
        query_embs=query_embs,
        index_embs=index_embs,
        relevant=rel,
        k=1,
    )
    # Each query retrieves its exact match at rank 1.
    assert abs(mrr - 1.0) < 1e-6
    assert abs(p_at_1 - 1.0) < 1e-6

    bc = eval_retrieval.compute_branch_consistency_at_k(
        query_embs=query_embs,
        index_embs=index_embs,
        query_paths=query_paths,
        index_paths=index_paths,
        k=1,
    )
    # At depth 1, all retrieved items share the same node as the query.
    assert 1 in bc
    assert abs(bc[1] - 1.0) < 1e-6


