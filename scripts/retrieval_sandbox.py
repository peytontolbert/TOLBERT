"""
Simple retrieval sandbox for TOLBERT.

This script:
  - loads a trained TOLBERT checkpoint and tokenizer,
  - encodes spans from a `spans_file`,
  - builds an in-memory index of embeddings,
  - lets you run ad-hoc queries over that index from the command line.

It is intended for small-scale experimentation and debugging, not production.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_model(cfg: Dict[str, Any], checkpoint: str, device: torch.device) -> TOLBERT:
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


def encode_all_spans(
    model: TOLBERT,
    tokenizer,
    spans_file: str,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Encode all spans in `spans_file` into a single tensor of embeddings,
    plus a parallel list of raw records.
    """
    dataset = TreeOfLifeDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    embs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    for rec in dataset._records:  # type: ignore[attr-defined]
        # Directly reuse dataset internals to avoid extra tokenization logic.
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

    emb_mat = torch.stack(embs, dim=0) if embs else torch.empty(0)
    return emb_mat, metas


def retrieve(
    query_emb: torch.Tensor,
    index_embs: torch.Tensor,
    k: int = 5,
) -> List[int]:
    """
    Return indices of top-k most similar embeddings using cosine similarity.
    """
    if index_embs.numel() == 0:
        return []
    sims = F.cosine_similarity(query_emb.unsqueeze(0), index_embs, dim=1)
    topk = torch.topk(sims, k=min(k, sims.numel()))
    return topk.indices.tolist()


def interactive_loop(
    model: TOLBERT,
    tokenizer,
    index_embs: torch.Tensor,
    metas: List[Dict[str, Any]],
    max_length: int,
    device: torch.device,
) -> None:
    print("Entering interactive retrieval loop. Type 'exit' to quit.")
    while True:
        text = input("Query> ")
        if not text or text.strip().lower() in {"exit", "quit"}:
            break
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        with torch.no_grad():
            out = model(
                input_ids=tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device),
            )
        q_emb = out["proj"].squeeze(0).cpu()
        idxs = retrieve(q_emb, index_embs, k=5)
        for i in idxs:
            meta = metas[i]
            snippet = meta.get("text", "")[:200].replace("\n", " ")
            print(f"- span_id={meta.get('span_id')} text={snippet!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TOLBERT retrieval sandbox.")
    ap.add_argument("--config", type=str, required=True, help="Path to training config.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
    ap.add_argument(
        "--spans_file",
        type=str,
        required=True,
        help="Spans JSONL file to index (e.g., code spans or paper spans).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_tolbert_config(args.config)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"],
        cache_dir="/data/checkpoints/",  # noqa: E501
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg, str(checkpoint_path), device=device)

    print("Encoding index spans...")
    index_embs, metas = encode_all_spans(
        model=model,
        tokenizer=tokenizer,
        spans_file=args.spans_file,
        max_length=cfg.get("max_length", 256),
        device=device,
    )
    print(f"Indexed {index_embs.size(0)} spans.")

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        index_embs=index_embs,
        metas=metas,
        max_length=cfg.get("max_length", 256),
        device=device,
    )


if __name__ == "__main__":
    main()


