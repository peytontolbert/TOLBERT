from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class SpanRecord:
    text: str
    node_path: Optional[List[int]]
    raw: Dict[str, Any]


class TreeOfLifeDataset(Dataset):
    """
    Minimal dataset for TOLBERT training.

    Assumes a JSONL `spans_file` where each line looks like:
        {
          "span_id": "s0",
          "text": "def forward(...):",
          "node_path": [0, 1, 2, 10]   # optional but recommended
        }

    You are free to extend this to include per-level fields instead of
    `node_path` (e.g., `level_1_id`, `level_2_id`, ...). This class is
    intentionally simple and meant as a starting point.
    """

    def __init__(
        self,
        spans_file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        mask_probability: float = 0.15,
    ) -> None:
        self.spans_file = spans_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

        self._records: List[SpanRecord] = []
        with open(spans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                node_path = obj.get("node_path")
                self._records.append(SpanRecord(text=text, node_path=node_path, raw=obj))

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
        # drop batch dimension
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _make_mlm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Very small, self-contained dynamic masking implementation.

        - 15% of non-special tokens are selected for prediction.
        - Of those, 80% replaced with [MASK], 10% with random token,
          10% left unchanged (BERT-style).
        """
        labels = input_ids.clone()

        if not hasattr(self.tokenizer, "get_special_tokens_mask"):
            # Fallback: treat everything as non-special
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    input_ids.tolist(), already_has_special_tokens=True
                ),
                dtype=torch.bool,
            )

        # mask candidates: non-special tokens
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # ignore index for unmasked tokens

        # 80% of selected tokens -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is not None:
            input_ids[indices_replaced] = mask_token_id

        # 10% of selected tokens -> random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% remain unchanged
        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]
        tokens = self._tokenize(rec.text)

        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))
        labels_mlm = self._make_mlm_labels(input_ids.clone())

        # Build per-level targets from node_path if available.
        level_targets: Dict[int, int] = {}
        if rec.node_path is not None:
            for level, node_id in enumerate(rec.node_path):
                # Level 0 is typically the root; skip if you don't train on it.
                if level == 0:
                    continue
                level_targets[level] = int(node_id)

        sample: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_mlm": labels_mlm,
            "level_targets": level_targets,
        }

        # Keep the full path around for potential contrastive loss.
        if rec.node_path is not None:
            sample["paths"] = list(rec.node_path)

        return sample


def collate_tree_of_life_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate_fn to handle:
      - dict-of-tensors for `level_targets`
      - list-of-paths for contrastive loss
    """
    # Simple stack for standard fields
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels_mlm = torch.stack([b["labels_mlm"] for b in batch], dim=0)

    # Merge level_targets into dict[level] -> tensor(batch,)
    # Collect all levels that appear in any sample.
    all_levels = set()
    for b in batch:
        all_levels.update(b["level_targets"].keys())

    level_targets: Dict[int, torch.Tensor] = {}
    for level in sorted(all_levels):
        targets_for_level = []
        for b in batch:
            # Use -100 for "unknown" / missing level for this sample
            targets_for_level.append(b["level_targets"].get(level, -100))
        level_targets[level] = torch.tensor(targets_for_level, dtype=torch.long)

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels_mlm": labels_mlm,
        "level_targets": level_targets,
    }

    # Optional paths for contrastive loss
    if all("paths" in b for b in batch):
        out["paths"] = [b["paths"] for b in batch]

    return out


