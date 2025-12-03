"""
Train a flat (leaf-level) baseline classifier on spans_file.

This script is intended to reproduce the "BERT-flat" style baselines from
Section 4 using standard HuggingFace models (e.g. BERT, SciBERT, CodeBERT,
ModernBERT) on the same JSONL spans files used for TOLBERT.

Key characteristics:
  - Single-label, flat classification on the *leaf* node of each path
    (last element of `node_path`).
  - Uses AutoModelForSequenceClassification with a single softmax head.
  - Ignores intermediate hierarchy levels entirely.

Example (CodeHierarchy, BERT-base flat):

  python -m scripts.train_flat_baseline \\
      --config configs/codehierarchy_example.yaml \\
      --output-dir checkpoints/codehierarchy_bert_flat \\
      --base-model-name bert-base-uncased

Example (WOS, SciBERT flat):

  python -m scripts.train_flat_baseline \\
      --config configs/wos_example.yaml \\
      --output-dir checkpoints/wos_scibert_flat \\
      --base-model-name allenai/scibert_scivocab_uncased

Example (CodeHierarchy, CodeBERT flat):

  python -m scripts.train_flat_baseline \\
      --config configs/codehierarchy_example.yaml \\
      --output-dir checkpoints/codehierarchy_codebert_flat \\
      --base-model-name microsoft/codebert-base

Example (mixed-domain, ModernBERT flat):

  python -m scripts.train_flat_baseline \\
      --config configs/codehierarchy_example.yaml \\
      --output-dir checkpoints/codehierarchy_modernbert_flat \\
      --base-model-name answerdotai/ModernBERT-base
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


class FlatLeafDataset(Dataset):
    """
    Minimal dataset for flat (leaf-level) classification.

    Expects the same JSONL format as TreeOfLifeDataset, but uses only:
      - "text": span text
      - "node_path": [root_id, ..., leaf_id]

    The label is taken to be the *last* element of node_path (leaf).
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
                    # Skip spans that do not have both text and a path.
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
        # Leaf = last element of node_path
        node_path: List[int] = rec["node_path"]
        label = int(node_path[-1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_flat_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a flat (leaf-level) baseline classifier.")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML/JSON config with at least spans_file, level_sizes, batch_size, num_epochs, lr.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned baseline model (HF save_pretrained format).",
    )
    ap.add_argument(
        "--base-model-name",
        type=str,
        required=True,
        help=(
            "HF model name or path for the baseline backbone "
            "(e.g., bert-base-uncased, allenai/scibert_scivocab_uncased, "
            "microsoft/codebert-base, answerdotai/ModernBERT-base)."
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_tolbert_config(args.config)

    device = torch.device(args.device)

    spans_file = cfg["spans_file"]
    if not Path(spans_file).exists():
        raise FileNotFoundError(f"spans_file not found: {spans_file}")

    # Determine number of leaf classes from level_sizes: leaf = max level index.
    level_sizes: Dict[int, int] = cfg["level_sizes"]
    leaf_level = max(level_sizes.keys())
    num_labels = level_sizes[leaf_level]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    dataset = FlatLeafDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 256),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_flat_batch,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=num_labels,
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))
    num_epochs = cfg.get("num_epochs", 1)
    log_every = cfg.get("log_every", 50)

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            global_step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
            optimizer.step()

            if global_step % log_every == 0:
                print(f"[epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved flat baseline model to {out_dir}")


if __name__ == "__main__":
    main()


