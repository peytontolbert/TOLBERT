"""
Minimal training skeleton for TOLBERT.

This script wires together:
  - config loading
  - tokenizer + dataset
  - model + optimizer
  - a simple training loop with optional contrastive loss

It assumes you have prepared a `spans_file` as described in `docs/tree_of_life.md`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset, collate_tree_of_life_batch
from tolbert.losses import tree_contrastive_loss
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_model(cfg: Dict[str, Any]) -> TOLBERT:
    model_cfg = TOLBERTConfig(
        base_model_name=cfg["base_model_name"],
        level_sizes=cfg["level_sizes"],
        proj_dim=cfg.get("proj_dim", 256),
        lambda_hier=cfg.get("lambda_hier", 1.0),
        lambda_path=cfg.get("lambda_path", 0.0),
        lambda_contrast=cfg.get("lambda_contrast", 0.0),
    )
    return TOLBERT(model_cfg)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a TOLBERT model (skeleton).")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file describing model and training params.",
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

    # Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    spans_file = cfg["spans_file"]
    if not Path(spans_file).exists():
        raise FileNotFoundError(f"spans_file not found: {spans_file}")

    dataset = TreeOfLifeDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 256),
        mask_probability=cfg.get("mask_probability", 0.15),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_tree_of_life_batch,
    )

    # Model and optimizer
    model = build_model(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))

    num_epochs = cfg.get("num_epochs", 1)
    use_contrastive = cfg.get("lambda_contrast", 0.0) > 0.0
    contrast_temp = cfg.get("contrast_temperature", 0.07)

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_mlm = batch["labels_mlm"].to(device)
            level_targets = {
                level: targets.to(device) for level, targets in batch["level_targets"].items()
            }

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_mlm=labels_mlm,
                level_targets=level_targets,
            )

            loss = out["loss"]

            if use_contrastive and "paths" in batch:
                contrast_loss = tree_contrastive_loss(
                    embeddings=out["proj"],
                    paths=batch["paths"],
                    temperature=contrast_temp,
                )
                loss = loss + cfg["lambda_contrast"] * contrast_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
            optimizer.step()

            if step % cfg.get("log_every", 50) == 0:
                loss_items = {k: v.item() for k, v in out["loss_components"].items()}
                print(
                    f"[epoch {epoch+1} step {step}] "
                    f"loss={loss.item():.4f} components={loss_items}"
                )

        # Simple checkpointing at end of each epoch
        out_dir = Path(cfg.get("output_dir", "checkpoints"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"tolbert_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()


