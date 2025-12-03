"""
Evaluate a trained TOLBERT checkpoint on a labeled spans_file.

This script implements the hierarchical classification metrics discussed
in the TOLBERT paper:

  - per-level accuracy
  - per-level micro F1 (for single-label multiclass this equals accuracy)
  - path accuracy (all supervised levels correct for a span)

Usage (CodeHierarchy example):

  python -m scripts.eval_hierarchical_classification \\
      --config configs/codehierarchy_example.yaml \\
      --checkpoint checkpoints/codehierarchy/tolbert_epoch5.pt \\
      --spans-file data/codehierarchy/spans_test.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset, collate_tree_of_life_batch
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_model(cfg: Dict[str, Any], checkpoint: str, device: torch.device) -> TOLBERT:
    model_cfg = TOLBERTConfig(
        base_model_name=cfg["base_model_name"],
        level_sizes=cfg["level_sizes"],
        proj_dim=cfg.get("proj_dim", 256),
        lambda_hier=cfg.get("lambda_hier", 1.0),
        lambda_path=cfg.get("lambda_path", 0.0),
        lambda_contrast=0.0,
    )
    model = TOLBERT(model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate hierarchical classification for TOLBERT.")
    ap.add_argument("--config", type=str, required=True, help="Training config used for the model.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt checkpoint.")
    ap.add_argument("--spans-file", type=str, required=True, help="Labeled spans JSONL file for eval.")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64).",
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

    spans_path = Path(args.spans_file)
    if not spans_path.exists():
        raise FileNotFoundError(f"spans_file not found: {spans_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"],
        cache_dir="/data/checkpoints/",  # noqa: E501
    )

    dataset = TreeOfLifeDataset(
        spans_file=str(spans_path),
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 256),
        mask_probability=cfg.get("mask_probability", 0.15),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_tree_of_life_batch,
    )

    model = build_model(cfg, checkpoint=args.checkpoint, device=device)

    # Metrics accumulators: per-level correct / total
    level_correct: Dict[int, int] = {}
    level_total: Dict[int, int] = {}

    # For path accuracy we track per-example correctness across supervised levels.
    path_correct = 0
    path_total = 0

    # For hierarchical precision/recall/F1 (Kiritchenko-style, single-path case),
    # we treat each (level, node_id) pair as one "label". Since every supervised
    # level has exactly one node, this reduces to counting how many levels are
    # correct across all examples.
    hier_total_true = 0
    hier_total_pred = 0
    hier_total_correct = 0

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
                paths=batch.get("paths"),
            )
            level_logits: Dict[str, torch.Tensor] = out["level_logits"]

            batch_size = input_ids.size(0)
            # Track per-example path correctness over all available levels.
            per_example_all_correct: List[bool] = [True] * batch_size

            for level_int, targets in level_targets.items():
                logits = level_logits[str(level_int)]
                preds = logits.argmax(dim=-1)

                # Ignore positions where target == -100 (unknown/masked level)
                mask = targets != -100
                if mask.sum().item() == 0:
                    continue

                correct = (preds == targets) & mask
                num_correct = correct.sum().item()
                num_total = mask.sum().item()

                level_correct[level_int] = level_correct.get(level_int, 0) + num_correct
                level_total[level_int] = level_total.get(level_int, 0) + num_total

                # For hierarchical precision/recall we count each supervised
                # (example, level) as one "true" label and one "predicted" label.
                # A label is correct iff the level prediction matches the target.
                hier_total_true += num_total
                hier_total_pred += num_total
                hier_total_correct += num_correct

                # Update per-example correctness for path accuracy
                # Only consider examples where this level is supervised.
                for i in range(batch_size):
                    if not mask[i]:
                        continue
                    if not bool(correct[i]):
                        per_example_all_correct[i] = False

            # Update path metrics: any example that had at least one supervised level
            # contributes to the denominator; it counts as correct only if all such
            # supervised levels were predicted correctly.
            for i in range(batch_size):
                # A sample is included if it has any supervised level
                has_any_level = any(
                    batch["level_targets"][lvl][i].item() != -100
                    for lvl in batch["level_targets"]
                )
                if not has_any_level:
                    continue
                path_total += 1
                if per_example_all_correct[i]:
                    path_correct += 1

    print("=== Hierarchical Classification Evaluation ===")
    for level in sorted(level_total.keys()):
        acc = level_correct[level] / max(1, level_total[level])
        # For single-label multiclass, micro-F1 equals accuracy; we report it explicitly.
        micro_f1 = acc
        print(
            f"Level {level}: "
            f"accuracy={acc:.4f} "
            f"micro_F1={micro_f1:.4f} "
            f"(n={level_total[level]})"
        )

    path_acc = path_correct / max(1, path_total)
    print(f"Path accuracy (all supervised levels correct): {path_acc:.4f} (n={path_total})")

    # Global hierarchical precision/recall/F1 over the full path, adapted from
    # Kiritchenko et al. to the single-path, single-label-per-level setting.
    # Here:
    #   - "true labels"  = all supervised (example, level) pairs,
    #   - "pred labels"  = one prediction per supervised level,
    #   - "correct"      = prediction matches the ground-truth node at that level.
    if hier_total_true > 0 and hier_total_pred > 0:
        hier_prec = hier_total_correct / hier_total_pred
        hier_rec = hier_total_correct / hier_total_true
        if hier_prec + hier_rec > 0.0:
            hier_f1 = 2.0 * hier_prec * hier_rec / (hier_prec + hier_rec)
        else:
            hier_f1 = 0.0
        print(
            "Hierarchical P/R/F1 over paths "
            f"(micro-style over all levels): "
            f"precision={hier_prec:.4f} recall={hier_rec:.4f} F1={hier_f1:.4f}"
        )
    else:
        print("Hierarchical P/R/F1 over paths: undefined (no supervised labels).")


if __name__ == "__main__":
    main()


