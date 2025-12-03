"""
Evaluate embeddings on retrieval-style tasks.

This script computes:
  - MRR (Mean Reciprocal Rank)
  - Precision@K

for a simple setup where:
  - The index consists of spans from one spans_file (e.g., code files).
  - The queries are spans from another spans_file (e.g., paper abstracts).
  - "Relevance" is defined by sharing one or more levels in the hierarchy.

By default we use "share at least level 2" as the relevance criterion, but
this can be adjusted via flags.

Usage (Paper2Code-style example):

  python -m scripts.eval_retrieval \\
      --config configs/codehierarchy_example.yaml \\
      --checkpoint checkpoints/tolbert_epoch5.pt \\
      --index-spans data/codehierarchy/spans_code.jsonl \\
      --query-spans data/wos/spans_papers.jsonl \\
      --relevant-min-level 2 \\
      --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_tolbert_model(cfg: Dict[str, Any], checkpoint: str, device: torch.device) -> TOLBERT:
    model_cfg = TOLBERTConfig(
        base_model_name=cfg["base_model_name"],
        level_sizes=cfg["level_sizes"],
        proj_dim=cfg.get("proj_dim", 256),
    )
    model = TOLBERT(model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def encode_spans_tolbert(
    model: TOLBERT,
    tokenizer,
    spans_file: str,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Encode all spans in `spans_file` into TOLBERT embeddings (proj head) and
    return (emb_mat, raw_records).

    This mirrors the logic in scripts/retrieval_sandbox.py but returns the full
    embedding matrix for evaluation purposes.
    """
    dataset = TreeOfLifeDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    embs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    for rec in dataset._records:  # type: ignore[attr-defined]
        tokens = tokenizer(
            rec.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        embs.append(out["proj"].squeeze(0).cpu())
        metas.append(rec.raw)

    if not embs:
        return torch.empty(0), metas
    return torch.stack(embs, dim=0), metas


def encode_spans_hf_encoder(
    model: AutoModel,
    tokenizer,
    spans_file: str,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Encode all spans using a vanilla HF encoder (e.g., BERT / CodeBERT /
    ModernBERT) and return CLS embeddings.

    This is used for baseline retrieval comparisons where we do not have
    TOLBERT's projection head and simply use the encoder's [CLS] representation.
    """
    dataset = TreeOfLifeDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    embs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    for rec in dataset._records:  # type: ignore[attr-defined]
        tokens = tokenizer(
            rec.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        # Standard HF encoder output: last_hidden_state[:, 0, :] is [CLS]
        cls = out.last_hidden_state[:, 0, :]
        embs.append(cls.squeeze(0).cpu())
        metas.append(rec.raw)

    if not embs:
        return torch.empty(0), metas
    return torch.stack(embs, dim=0), metas


def compute_relevance_mask(
    query_paths: Sequence[Sequence[int]],
    index_paths: Sequence[Sequence[int]],
    min_level: int,
) -> List[List[bool]]:
    """
    Build a boolean matrix R where R[i][j] is True if query i and index j
    share at least one node at depth >= min_level in their node_path.

    Paths are lists of node ids [root, c1, c2, ...].
    """
    rel: List[List[bool]] = []
    for q_path in query_paths:
        row: List[bool] = []
        for idx_path in index_paths:
            # shared depth = deepest k where q_path[k] == idx_path[k]
            shared = False
            max_shared_level = min(len(q_path), len(idx_path)) - 1
            for lvl in range(min_level, max_shared_level + 1):
                if q_path[lvl] == idx_path[lvl]:
                    shared = True
                    break
            row.append(shared)
        rel.append(row)
    return rel


def eval_retrieval(
    query_embs: torch.Tensor,
    index_embs: torch.Tensor,
    relevant: List[List[bool]],
    k: int,
) -> Tuple[float, float]:
    """
    Compute MRR and Precision@k given:
      - query_embs: (Q, D)
      - index_embs: (N, D)
      - relevant:   QxN boolean matrix (relevance marks)
    """
    if query_embs.numel() == 0 or index_embs.numel() == 0:
        return 0.0, 0.0

    # Normalize embeddings for cosine similarity
    query_embs = F.normalize(query_embs, dim=-1)
    index_embs = F.normalize(index_embs, dim=-1)

    # Similarity matrix (Q, N)
    sims = torch.matmul(query_embs, index_embs.T)

    Q, N = sims.shape
    k = min(k, N)

    mrr = 0.0
    prec_at_k = 0.0

    for i in range(Q):
        rel_row = relevant[i]
        if not any(rel_row):
            continue  # skip queries with no relevant items

        scores = sims[i]
        topk_vals, topk_idx = torch.topk(scores, k=k)
        topk_idx_list = topk_idx.tolist()

        # MRR: find rank of first relevant
        rr = 0.0
        for rank, j in enumerate(topk_idx_list, start=1):
            if rel_row[j]:
                rr = 1.0 / rank
                break
        mrr += rr

        # Precision@k: fraction of top-k that are relevant
        num_rel_in_topk = sum(1 for j in topk_idx_list if rel_row[j])
        prec_at_k += num_rel_in_topk / float(k)

    # Normalize by number of queries that had at least one relevant item.
    num_queries_with_rel = sum(1 for row in relevant if any(row))
    if num_queries_with_rel == 0:
        return 0.0, 0.0

    mrr /= num_queries_with_rel
    prec_at_k /= num_queries_with_rel
    return mrr, prec_at_k


def compute_branch_consistency_at_k(
    query_embs: torch.Tensor,
    index_embs: torch.Tensor,
    query_paths: Sequence[Sequence[int]],
    index_paths: Sequence[Sequence[int]],
    k: int,
) -> Dict[int, float]:
    """
    Compute branch-consistency@k per hierarchy depth.

    For each query and each depth level ℓ (index in node_path, 0=root),
    we measure the fraction of the top-k retrieved items whose node_path
    matches the query's node at depth ℓ. Results are averaged over all
    queries that have a node defined at that depth.
    """
    if query_embs.numel() == 0 or index_embs.numel() == 0:
        return {}

    # Normalize embeddings for cosine similarity
    query_embs = F.normalize(query_embs, dim=-1)
    index_embs = F.normalize(index_embs, dim=-1)

    sims = torch.matmul(query_embs, index_embs.T)  # (Q, N)
    Q, N = sims.shape
    k = min(k, N)

    # Determine maximum depth observed across all paths (0-based index).
    max_depth = 0
    for p in list(query_paths) + list(index_paths):
        if p:
            max_depth = max(max_depth, len(p) - 1)

    # Accumulators: depth -> (sum_fraction, num_queries_with_node)
    num_levels = max_depth + 1
    sum_frac: List[float] = [0.0 for _ in range(num_levels)]
    count_q: List[int] = [0 for _ in range(num_levels)]

    for qi in range(Q):
        q_path = list(query_paths[qi]) if qi < len(query_paths) else []
        if not q_path:
            continue

        scores = sims[qi]
        topk_vals, topk_idx = torch.topk(scores, k=k)
        topk_indices = topk_idx.tolist()

        for depth in range(1, num_levels):  # skip depth 0 (root) by default
            if len(q_path) <= depth:
                continue
            q_node = q_path[depth]
            # Count how many of the top-k share the same node at this depth.
            matches = 0
            valid_cands = 0
            for j in topk_indices:
                if j >= len(index_paths):
                    continue
                idx_path = list(index_paths[j])
                if len(idx_path) <= depth:
                    continue
                valid_cands += 1
                if idx_path[depth] == q_node:
                    matches += 1
            if valid_cands == 0:
                continue
            frac = matches / float(valid_cands)
            sum_frac[depth] += frac
            count_q[depth] += 1

    consistency: Dict[int, float] = {}
    for depth in range(1, num_levels):
        if count_q[depth] == 0:
            continue
        consistency[depth] = sum_frac[depth] / float(count_q[depth])
    return consistency


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate retrieval (MRR, Precision@K, branch consistency).")
    ap.add_argument("--config", type=str, required=True, help="Training config used for the model.")
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "For mode=tolbert: path to .pt checkpoint (state_dict). "
            "For mode=hf_encoder: HF model name or directory to load via AutoModel."
        ),
    )
    ap.add_argument(
        "--index-spans",
        type=str,
        required=True,
        help="Spans JSONL file used as the retrieval index (e.g., code).",
    )
    ap.add_argument(
        "--query-spans",
        type=str,
        required=True,
        help="Spans JSONL file used as queries (e.g., papers).",
    )
    ap.add_argument(
        "--relevant-min-level",
        type=int,
        default=2,
        help="Minimum level depth (in node_path) that must match to count as relevant.",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for Precision@K and MRR computation (default: 5).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="tolbert",
        choices=["tolbert", "hf_encoder"],
        help=(
            "Embedding backend: 'tolbert' (default, uses TOLBERT proj head) "
            "or 'hf_encoder' (vanilla HF encoder CLS for baselines like BERT, "
            "SciBERT, CodeBERT, ModernBERT)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_tolbert_config(args.config)
    device = torch.device(args.device)

    index_spans_path = Path(args.index_spans)
    query_spans_path = Path(args.query_spans)
    if not index_spans_path.exists():
        raise FileNotFoundError(f"index_spans not found: {index_spans_path}")
    if not query_spans_path.exists():
        raise FileNotFoundError(f"query_spans not found: {query_spans_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"] if args.mode == "tolbert" else args.checkpoint,
        cache_dir="/data/checkpoints/",  # noqa: E501
    )

    if args.mode == "tolbert":
        model = build_tolbert_model(cfg, checkpoint=args.checkpoint, device=device)
        encode_fn = encode_spans_tolbert
    else:
        # Vanilla HF encoder baseline (BERT, SciBERT, CodeBERT, ModernBERT, etc.)
        model = AutoModel.from_pretrained(
            args.checkpoint,
            cache_dir="/data/checkpoints/",  # noqa: E501
        )
        model.to(device)
        model.eval()
        encode_fn = encode_spans_hf_encoder

    print(f"Encoding index spans from {index_spans_path} ...")
    index_embs, index_metas = encode_fn(
        model=model,
        tokenizer=tokenizer,
        spans_file=str(index_spans_path),
        max_length=cfg.get("max_length", 256),
        device=device,
    )

    print(f"Encoding query spans from {query_spans_path} ...")
    query_embs, query_metas = encode_fn(
        model=model,
        tokenizer=tokenizer,
        spans_file=str(query_spans_path),
        max_length=cfg.get("max_length", 256),
        device=device,
    )

    # Extract node_path lists from metas.
    index_paths: List[Sequence[int]] = [
        rec.get("node_path", []) for rec in index_metas  # type: ignore[assignment]
    ]
    query_paths: List[Sequence[int]] = [
        rec.get("node_path", []) for rec in query_metas  # type: ignore[assignment]
    ]

    relevant = compute_relevance_mask(
        query_paths=query_paths,
        index_paths=index_paths,
        min_level=args.relevant_min_level,
    )

    mrr, p_at_k = eval_retrieval(
        query_embs=query_embs,
        index_embs=index_embs,
        relevant=relevant,
        k=args.k,
    )

    print("=== Retrieval Evaluation ===")
    print(f"MRR:        {mrr:.4f}")
    print(f"Precision@{args.k}: {p_at_k:.4f}")

    # Ontology-aware metric: branch-consistency@k per depth level.
    branch_consistency = compute_branch_consistency_at_k(
        query_embs=query_embs,
        index_embs=index_embs,
        query_paths=query_paths,
        index_paths=index_paths,
        k=args.k,
    )
    if branch_consistency:
        print("\nBranch-consistency@{k} by depth (node_path index):".format(k=args.k))
        for depth in sorted(branch_consistency.keys()):
            # depth 0 is the root; depths >=1 correspond to actual hierarchy levels.
            print(f"  depth {depth}: {branch_consistency[depth]:.4f}")


if __name__ == "__main__":
    main()


