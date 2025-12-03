"""
Agent / LLM stack skeleton: PR -> paper recommendations via TOLBERT.

This script is a high-level wiring of the "worked scenario" from `docs/usage.md`:
  1. Ingest a pull request (PR) as text spans.
  2. Encode spans with TOLBERT.
  3. Locate the PR within the Tree-of-Life (domain/subdomain).
  4. Retrieve relevant paper spans from a pre-built index.
  5. Use an LLM to summarize / explain the recommendations.

You are expected to plug in:
  - real PR ingestion,
  - a proper paper-span index (vector DB, search service, etc.),
  - and your own LLM backend in `call_llm`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.modeling import TOLBERT, TOLBERTConfig


def call_llm(prompt: str) -> str:
    """
    Placeholder for your LLM call.

    Replace this with your own LLM client (OpenAI, local model, etc.).
    """
    raise NotImplementedError("Integrate your LLM client here.")


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


def simple_pr_spans(pr_text: str) -> List[str]:
    """
    Very simple PR spanizer:
      - splits on double newlines,
      - keeps reasonably sized chunks as candidate spans.

    You likely want to replace this with something that understands diffs
    and function boundaries.
    """
    parts = [p.strip() for p in pr_text.split("\n\n") if p.strip()]
    return parts


def encode_spans(
    model: TOLBERT,
    tokenizer,
    spans: List[str],
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    embs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    for i, text in enumerate(spans):
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
        embs.append(out["proj"].squeeze(0).cpu())

        # Record predicted hierarchy information for analysis.
        level_logits = out["level_logits"]
        level_preds = {
            int(level): logits.argmax(dim=-1).item()
            for level, logits in level_logits.items()
        }

        metas.append(
            {
                "span_index": i,
                "text": text,
                "level_predictions": level_preds,
            }
        )

    if not embs:
        return torch.empty(0), metas
    return torch.stack(embs, dim=0), metas


def load_paper_index(index_path: str) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Load a pre-built paper-span index.

    This function is intentionally schematic. You might, for example, store:
      - embeddings as a `.pt` tensor,
      - metadata as a JSON/JSONL file with the same length.
    """
    index_dir = Path(index_path)
    emb_path = index_dir / "embeddings.pt"
    meta_path = index_dir / "metadata.jsonl"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Expected embeddings at {emb_path} and metadata at {meta_path}."
        )

    embs = torch.load(emb_path, map_location="cpu")
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        import json

        for line in f:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))

    if embs.size(0) != len(metas):
        raise ValueError("Mismatch between number of embeddings and metadata entries.")

    return embs, metas


def retrieve_top_papers(
    pr_embs: torch.Tensor,
    paper_embs: torch.Tensor,
    paper_metas: List[Dict[str, Any]],
    top_k_per_span: int = 5,
    top_k_papers: int = 10,
) -> List[Dict[str, Any]]:
    """
    Simple multi-span retrieval:
      - for each PR span embedding, retrieve top_k_per_span paper spans,
      - aggregate scores per paper id,
      - return top_k_papers papers with aggregated scores.
    """
    if pr_embs.numel() == 0 or paper_embs.numel() == 0:
        return []

    scores: Dict[str, float] = {}
    for i in range(pr_embs.size(0)):
        q = pr_embs[i : i + 1]
        sims = F.cosine_similarity(q, paper_embs, dim=1)
        topk = torch.topk(sims, k=min(top_k_per_span, sims.numel()))
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            meta = paper_metas[idx]
            paper_id = str(meta.get("paper_id", idx))
            scores[paper_id] = scores.get(paper_id, 0.0) + float(score)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k_papers]
    results: List[Dict[str, Any]] = []
    for paper_id, score in ranked:
        # find first meta entry with this paper_id
        meta = next((m for m in paper_metas if str(m.get("paper_id")) == paper_id), None)
        if meta is None:
            continue
        results.append(
            {
                "paper_id": paper_id,
                "score": score,
                "title": meta.get("title"),
                "abstract": meta.get("abstract"),
            }
        )
    return results


def build_prompt(pr_text: str, recommendations: List[Dict[str, Any]]) -> str:
    """
    Construct a prompt for the LLM given:
      - the PR text,
      - a set of candidate papers with scores and metadata.
    """
    lines = [
        "You are an expert AI code reviewer and research assistant.",
        "Given the following pull request and a set of candidate research papers,",
        "explain which papers are most relevant and why.",
        "",
        "PULL REQUEST:",
        pr_text,
        "",
        "CANDIDATE PAPERS:",
    ]
    for i, rec in enumerate(recommendations, start=1):
        lines.append(
            f"{i}. [paper_id={rec['paper_id']}] title={rec.get('title')!r} "
            f"(score={rec['score']:.3f})"
        )
        if rec.get("abstract"):
            lines.append(f"   abstract={rec['abstract']!r}")
    lines.append("")
    lines.append(
        "For each paper above, give a short explanation (2-4 sentences) "
        "of how it relates to the PR, and suggest how the PR could benefit "
        "from ideas in these papers."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PR -> paper recommendation agent (skeleton).")
    ap.add_argument("--config", type=str, required=True, help="Path to TOLBERT config.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    ap.add_argument(
        "--paper_index",
        type=str,
        required=True,
        help="Path to directory with paper-span embeddings and metadata.",
    )
    ap.add_argument(
        "--pr_file",
        type=str,
        required=True,
        help="Path to a text file with PR description / diff.",
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"],
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    model = build_model(cfg, str(checkpoint_path), device=device)

    # Load and span-ize the PR
    pr_text = Path(args.pr_file).read_text(encoding="utf-8")
    pr_spans = simple_pr_spans(pr_text)

    pr_embs, span_metas = encode_spans(
        model=model,
        tokenizer=tokenizer,
        spans=pr_spans,
        max_length=cfg.get("max_length", 256),
        device=device,
    )

    # Load paper index and retrieve recommendations
    paper_embs, paper_metas = load_paper_index(args.paper_index)
    recommendations = retrieve_top_papers(
        pr_embs=pr_embs,
        paper_embs=paper_embs,
        paper_metas=paper_metas,
        top_k_per_span=cfg.get("top_k_per_span", 5),
        top_k_papers=cfg.get("top_k_papers", 10),
    )

    if not recommendations:
        print("No paper recommendations found (empty embeddings or index).")
        return

    prompt = build_prompt(pr_text, recommendations)

    # Hand off to LLM (you implement this)
    print("=== LLM PROMPT BEGIN ===")
    print(prompt)
    print("=== LLM PROMPT END ===")
    print()
    print("Now call your LLM client with the prompt above (see `call_llm`).")


if __name__ == "__main__":
    main()


