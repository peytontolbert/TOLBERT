"""
Minimal training skeleton for TOLBERT **with curriculum and multi-domain support**.

This script wires together:
  - config loading
  - tokenizer + dataset(s)
  - model + optimizer
  - a training loop with optional curriculum over hierarchical / path / contrastive losses

It assumes you have prepared one or more `spans_file`(s) as described in `docs/tree_of_life.md`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
import os

from tolbert.config import load_tolbert_config
from tolbert.data import TreeOfLifeDataset, collate_tree_of_life_batch
from tolbert.losses import tree_contrastive_loss
from tolbert.modeling import TOLBERT, TOLBERTConfig


def build_model(cfg: Dict[str, Any]) -> TOLBERT:
    """
    Build a TOLBERT model from config.

    Notes:
      - `lambda_hier` and `lambda_path` are used as *default* weights inside
        the model when no curriculum is configured.
      - When a curriculum is enabled, per-step weights are applied in the
        training loop based on `loss_components`, and these config values act
        only as fallbacks.
    """
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
    ap = argparse.ArgumentParser(description="Train a TOLBERT model.")
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


def _build_dataset(cfg: Dict[str, Any], tokenizer: Any) -> torch.utils.data.Dataset:
    """
    Build a (possibly multi-domain) dataset from config.

    Supported config patterns:
      - `spans_file: /path/to/spans.jsonl`
      - `spans_files: [ /path/to/code.jsonl, /path/to/papers.jsonl, ... ]`

    In the multi-file case we simply concatenate datasets; mixed-domain
    batches then come from the union of underlying corpora, as described
    in the paper.
    """
    max_length = cfg.get("max_length", 256)
    mask_probability = cfg.get("mask_probability", 0.15)

    if "spans_files" in cfg and cfg["spans_files"] is not None:
        spans_files_cfg = cfg["spans_files"]
        if isinstance(spans_files_cfg, str):
            spans_files = [spans_files_cfg]
        else:
            spans_files = list(spans_files_cfg)

        datasets = []
        for spans_path in spans_files:
            if not Path(spans_path).exists():
                raise FileNotFoundError(f"spans_file not found: {spans_path}")
            datasets.append(
                TreeOfLifeDataset(
                    spans_file=spans_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    mask_probability=mask_probability,
                )
            )

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    # Fallback: single-file setup (backwards-compatible).
    spans_file = cfg["spans_file"]
    if not Path(spans_file).exists():
        raise FileNotFoundError(f"spans_file not found: {spans_file}")

    return TreeOfLifeDataset(
        spans_file=spans_file,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_probability=mask_probability,
    )


def _get_curriculum_stage(
    curriculum_cfg: Optional[Dict[str, Any]],
    global_step: int,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the active curriculum stage for the given step.

    Expected config shape:
      curriculum:
        enabled: true
        stages:
          - name: warmup
            start_step: 0
            end_step: 10000
            max_supervised_level: 2
            lambda_hier: 1.0
            lambda_path: 0.0
            lambda_contrast: 0.0
          - name: deep
            start_step: 10000
            end_step: 20000
            max_supervised_level: 3
            lambda_hier: 1.0
            lambda_path: 0.0
            lambda_contrast: 0.0
          - name: contrastive
            start_step: 20000
            end_step: 30000
            max_supervised_level: 3
            lambda_hier: 1.0
            lambda_path: 0.0
            lambda_contrast: 0.05
          - name: path
            start_step: 30000
            end_step: null    # or omit to mean "until end"
            max_supervised_level: 3
            lambda_hier: 1.0
            lambda_path: 0.05
            lambda_contrast: 0.05

    If `curriculum.enabled` is false or missing, returns None.
    """
    if not curriculum_cfg or not curriculum_cfg.get("enabled", False):
        return None

    stages = curriculum_cfg.get("stages") or []
    if not stages:
        return None

    for stage in stages:
        start = int(stage.get("start_step", 0))
        end_raw = stage.get("end_step", None)
        end: Optional[int] = None if end_raw is None else int(end_raw)

        if global_step < start:
            continue
        if end is not None and global_step >= end:
            continue
        return stage

    # If we fall past all defined ranges, use the last stage as a default.
    return stages[-1]


def main() -> None:
    args = parse_args()
    cfg = load_tolbert_config(args.config)

    device = torch.device(args.device)

    # Tokenizer and dataset(s)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"],
        cache_dir="/data/checkpoints/",  # noqa: E501
    )
    dataset = _build_dataset(cfg, tokenizer)

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
    contrast_temp = cfg.get("contrast_temperature", 0.07)
    curriculum_cfg: Optional[Dict[str, Any]] = cfg.get("curriculum")

    # Fallback static weights when curriculum is disabled.
    static_lambda_hier = float(cfg.get("lambda_hier", 1.0))
    static_lambda_path = float(cfg.get("lambda_path", 0.0))
    static_lambda_contrast = float(cfg.get("lambda_contrast", 0.0))
    use_contrastive_static = static_lambda_contrast > 0.0

    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_mlm = batch["labels_mlm"].to(device)

            # Move level targets to device and optionally trim by curriculum.
            level_targets = {
                level: targets.to(device) for level, targets in batch["level_targets"].items()
            }

            stage = _get_curriculum_stage(curriculum_cfg, global_step)

            # Determine per-step hyperparameters (MLM is always on).
            if stage is not None:
                max_level = stage.get("max_supervised_level")
                if max_level is not None:
                    max_level_int = int(max_level)
                    level_targets = {
                        level: tgt for level, tgt in level_targets.items() if level <= max_level_int
                    }

                lambda_hier = float(stage.get("lambda_hier", static_lambda_hier))
                lambda_path = float(stage.get("lambda_path", 0.0))
                lambda_contrast = float(stage.get("lambda_contrast", 0.0))
            else:
                lambda_hier = static_lambda_hier
                lambda_path = static_lambda_path
                lambda_contrast = static_lambda_contrast

            # Only pass paths into the model if we actually intend to use path loss.
            paths_for_model = batch.get("paths") if (lambda_path > 0.0 and "paths" in batch) else None

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_mlm=labels_mlm,
                level_targets=level_targets,
                paths=paths_for_model,
            )

            loss_components = out["loss_components"]
            mlm_loss = loss_components.get("mlm")
            hier_loss = loss_components.get("hier")
            path_loss = loss_components.get("path")

            # Aggregate loss according to either curriculum or static weights.
            if mlm_loss is None:
                raise RuntimeError("MLM loss is expected but missing from model outputs.")

            loss = mlm_loss

            if hier_loss is not None and lambda_hier != 0.0:
                loss = loss + lambda_hier * hier_loss

            if path_loss is not None and lambda_path != 0.0:
                loss = loss + lambda_path * path_loss

            # Contrastive loss: tree-aware supervised contrastive over paths.
            use_contrastive = lambda_contrast > 0.0
            if use_contrastive and "paths" in batch:
                contrast_loss = tree_contrastive_loss(
                    embeddings=out["proj"],
                    paths=batch["paths"],
                    temperature=contrast_temp,
                )
                loss = loss + lambda_contrast * contrast_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
            optimizer.step()

            if step % cfg.get("log_every", 50) == 0:
                loss_items = {k: v.item() for k, v in loss_components.items()}
                print(
                    f"[epoch {epoch+1} step {step} global_step {global_step}] "
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


