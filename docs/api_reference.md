## TOLBERT Python API (planned)

This file describes an **aspirational Python API** for a future `tolbert` package.

- **Status**: there is **no published `tolbert` package yet**; all module paths and signatures below are **planned, not stable**.
- **Goal**: give a concrete target layout so that:
  - Docs like `usage.md` can show realistic examples.
  - The eventual implementation can be checked against a single source of truth.

### Target package layout (draft)

- **Top-level package**: `tolbert`
  - **`tolbert.modeling`**
    - `class TOLBERTConfig`
    - `class TOLBERT`
  - **`tolbert.data`**
    - `class TreeOfLifeDataset`
  - **`tolbert.losses`**
    - `def tree_contrastive_loss(embeddings, paths, temperature: float = 0.07)`
  - **`tolbert.config`**
    - `def load_tolbert_config(path: str) -> dict`

Everything below assumes this layout; when the actual code diverges, update this file and `usage.md` to match.

### Core classes and functions

- **`tolbert.modeling.TOLBERTConfig`**
  - Purpose: capture model + head configuration (levels, hidden sizes, loss weights, etc.).
  - Sketch:

```python
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TOLBERTConfig:
    base_model_name: str
    level_sizes: Dict[int, int]
    proj_dim: int = 256
    lambda_hier: float = 1.0
    lambda_path: float = 0.0
    lambda_contrast: float = 0.0
```

- **`tolbert.modeling.TOLBERT`**
  - Purpose: BERT/RoBERTa-style encoder augmented with multi-level heads and (optionally) contrastive embeddings that respect the Tree-of-Life.
  - Canonical import:

```python
from tolbert.modeling import TOLBERT, TOLBERTConfig
```

  - Provisional constructor:
    - `__init__(config: TOLBERTConfig)`
    - Alternate constructor: `from_pretrained(checkpoint: str, config: Optional[TOLBERTConfig] = None) -> "TOLBERT"`
  - Key method:
    - `forward(input_ids, attention_mask=None, labels_mlm=None, level_targets=None) -> Dict[str, object]`
      - Returns a dict containing:
        - `loss`: scalar training loss (if labels are provided).
        - `loss_components`: per-loss breakdown (MLM, level losses, etc.).
        - `mlm_logits`: token-level logits for masked language modeling.
        - `level_logits`: per-level classification logits over tree nodes.
        - `proj`: normalized contrastive embedding for each input.

- **`tolbert.data.TreeOfLifeDataset`**
  - Purpose: wrap tokenized spans and their associated paths \(\pi(x)\), exposing fields needed for MLM + hierarchical + contrastive training.
  - Canonical import:

```python
from tolbert.data import TreeOfLifeDataset
```

  - Provisional signature:
    - `__init__(self, spans_file: str, tokenizer, max_length: int = 256)`
    - Exposes items with:
      - `input_ids`, `attention_mask`, `labels_mlm`, `level_targets`, `paths`.

- **`tolbert.losses.tree_contrastive_loss`**
  - Purpose: compute a tree-aware InfoNCE-style loss using paths in the Tree-of-Life to define positives/negatives.
  - Canonical import:

```python
from tolbert.losses import tree_contrastive_loss
```

- **`tolbert.config.load_tolbert_config`**
  - Purpose: load a YAML/JSON config describing model + training hyperparameters (levels, loss weights, etc.).
  - Canonical import:

```python
from tolbert.config import load_tolbert_config
```

### Minimal training loop (planned)

The following example is **non-functional until the `tolbert` package exists**, but shows the intended API surface:

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tolbert.modeling import TOLBERT, TOLBERTConfig
from tolbert.data import TreeOfLifeDataset
from tolbert.losses import tree_contrastive_loss
from tolbert.config import load_tolbert_config


def main():
    cfg = load_tolbert_config("configs/tolbert.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    dataset = TreeOfLifeDataset(
        spans_file=cfg["spans_file"],
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 256),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
    )

    model_cfg = TOLBERTConfig(
        base_model_name=cfg["base_model_name"],
        level_sizes=cfg["level_sizes"],
        proj_dim=cfg.get("proj_dim", 256),
        lambda_hier=cfg.get("lambda_hier", 1.0),
        lambda_path=cfg.get("lambda_path", 0.0),
        lambda_contrast=cfg.get("lambda_contrast", 0.0),
    )
    model = TOLBERT(model_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))

    for batch in dataloader:
        optimizer.zero_grad()

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels_mlm=batch.get("labels_mlm"),
            level_targets=batch.get("level_targets"),
        )

        loss = out["loss"]

        if cfg.get("lambda_contrast", 0.0) > 0.0 and "paths" in batch:
            contrast_loss = tree_contrastive_loss(
                embeddings=out["proj"],
                paths=batch["paths"],
                temperature=cfg.get("contrast_temperature", 0.07),
            )
            loss = loss + cfg["lambda_contrast"] * contrast_loss

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
```

### Minimal inference script (planned)

This matches the import paths used in `usage.md` and assumes you have a trained checkpoint:

```python
import torch
from transformers import AutoTokenizer

from tolbert.modeling import TOLBERT, TOLBERTConfig


def load_encoder(checkpoint: str) -> tuple[TOLBERT, AutoTokenizer]:
    # In a real implementation, config would be loaded from disk as well.
    config = TOLBERTConfig(
        base_model_name="bert-base-uncased",
        level_sizes={1: 10, 2: 100, 3: 1000},
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = TOLBERT.from_pretrained(checkpoint, config=config)
    model.eval()
    return model, tokenizer


def encode_span(model: TOLBERT, tokenizer: AutoTokenizer, text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    return out["proj"].squeeze(0)
```

When the `tolbert` package is implemented, update this file so that:

- All import paths correspond to real modules.
- All signatures match the concrete implementations.
- The training loop and inference snippet can be copy-pasted into a working script with no further changes.

