"""
Evaluate a flat (leaf-level) baseline classifier on a labeled spans_file.

This computes standard leaf-level accuracy for the "BERT-flat" style baselines
trained with scripts/train_flat_baseline.py. It does NOT reconstruct full
hierarchical paths; it simply evaluates how well the model predicts the leaf
node id (last element of node_path).

Example:

  python -m scripts.eval_flat_baseline \\
      --config configs/codehierarchy_example.yaml \\
      --checkpoint-dir checkpoints/codehierarchy_bert_flat \\
      --spans-file /data/tolbert/data/codehierarchy/spans_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
import os

from tolbert.config import load_tolbert_config


class FlatLeafEvalDataset(Dataset):
    """
    Identical to FlatLeafDataset but without any training-specific behavior.
    """

    def __init__(
        self,
        spans_file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.spans_file = spans_file
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._records: List[Dict[str, Any]] = []
        with open(spans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj or "node_path" not in obj:
                    continue
                if not obj["node_path"]:
                    continue
                self._records.append(obj)

    def __len__(self) -> int:
        return len(self._records)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]
        tokens = self._tokenize(rec["text"])
        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))
        node_path: List[int] = rec["node_path"]
        label = int(node_path[-1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_flat_eval_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate flat (leaf-level) baseline classifier.")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config with max_length / batch_size; level_sizes is used to sanity-check num_labels.",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned baseline model (save_pretrained output).",
    )
    ap.add_argument(
        "--spans-file",
        type=str,
        required=True,
        help="Labeled spans JSONL file for evaluation (e.g., *_test.jsonl).",
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

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    # Leaf class count from config (for sanity checking).
    level_sizes: Dict[int, int] = cfg["level_sizes"]
    leaf_level = max(level_sizes.keys())
    num_labels_expected = level_sizes[leaf_level]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    if model.num_labels != num_labels_expected:
        print(
            f"Warning: model.num_labels={model.num_labels} "
            f"but config expects {num_labels_expected} leaf classes."
        )

    model.to(device)
    model.eval()

    dataset = FlatLeafEvalDataset(
        spans_file=str(spans_path),
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 256),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_flat_eval_batch,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(1, total)
    print("=== Flat Leaf-Level Classification Evaluation ===")
    print(f"Accuracy (leaf node id): {acc:.4f} (n={total})")


if __name__ == "__main__":
    main()


