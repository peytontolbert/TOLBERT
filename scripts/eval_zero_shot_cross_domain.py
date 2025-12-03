"""
Simple cross-domain zero-shot evaluation script for TOLBERT.

This wraps the hierarchical classification evaluator to support the scenario:

  - Train TOLBERT on domain A (e.g., CodeHierarchy).
  - Evaluate the same checkpoint on domain B (e.g., WOS) without further training.

Usage:

  # Evaluate a code-trained model on WOS spans
  python -m scripts.eval_zero_shot_cross_domain \\
      --config configs/codehierarchy_example.yaml \\
      --checkpoint checkpoints/codehierarchy/tolbert_epoch5.pt \\
      --target-config configs/wos_example.yaml \\
      --target-spans data/wos/spans_test.jsonl

The script uses:
  - base_model_name and model head structure from the *source* config
    (the one used for training),
  - but evaluation data and tokenizer settings (e.g., max_length) from
    the *target* config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset, collate_tree_of_life_batch
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_model(src_cfg: Dict[str, Any], checkpoint: str, device: torch.device) -> TOLBERT:
    model_cfg = TOLBERTConfig(
        base_model_name=src_cfg["base_model_name"],
        level_sizes=src_cfg["level_sizes"],
        proj_dim=src_cfg.get("proj_dim", 256),
        lambda_hier=src_cfg.get("lambda_hier", 1.0),
        lambda_path=src_cfg.get("lambda_path", 0.0),
        lambda_contrast=0.0,
    )
    model = TOLBERT(model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Zero-shot cross-domain eval for TOLBERT.")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Source (training) config for the model (defines heads).",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint on source domain.",
    )
    ap.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Target domain config (used for tokenizer and data params).",
    )
    ap.add_argument(
        "--target-spans",
        type=str,
        required=True,
        help="Spans JSONL file from the target domain (with node_path labels).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
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

    src_cfg = load_tolbert_config(args.config)
    tgt_cfg = load_tolbert_config(args.target_config)
    device = torch.device(args.device)

    spans_path = Path(args.target_spans)
    if not spans_path.exists():
        raise FileNotFoundError(f"target_spans not found: {spans_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tgt_cfg["base_model_name"],
        cache_dir="/data/checkpoints/",  # noqa: E501
    )

    dataset = TreeOfLifeDataset(
        spans_file=str(spans_path),
        tokenizer=tokenizer,
        max_length=tgt_cfg.get("max_length", 256),
        mask_probability=tgt_cfg.get("mask_probability", 0.15),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=tgt_cfg.get("num_workers", 0),
        collate_fn=collate_tree_of_life_batch,
    )

    model = build_model(src_cfg, checkpoint=args.checkpoint, device=device)

    level_correct: Dict[int, int] = {}
    level_total: Dict[int, int] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            level_targets = {
                level: targets.to(device) for level, targets in batch["level_targets"].items()
            }

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                level_targets=level_targets,
            )
            level_logits: Dict[str, torch.Tensor] = out["level_logits"]

            for level_int, targets in level_targets.items():
                logits = level_logits.get(str(level_int))
                if logits is None:
                    continue
                preds = logits.argmax(dim=-1)

                mask = targets != -100
                if mask.sum().item() == 0:
                    continue

                correct = (preds == targets) & mask
                num_correct = correct.sum().item()
                num_total = mask.sum().item()

                level_correct[level_int] = level_correct.get(level_int, 0) + num_correct
                level_total[level_int] = level_total.get(level_int, 0) + num_total

    print("=== Zero-shot Cross-domain Classification ===")
    for level in sorted(level_total.keys()):
        acc = level_correct[level] / max(1, level_total[level])
        print(f"Level {level}: accuracy={acc:.4f} (n={level_total[level]})")


if __name__ == "__main__":
    main()


