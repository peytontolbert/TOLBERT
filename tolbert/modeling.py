from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class TOLBERTConfig:
    """
    Minimal configuration for the TOLBERT encoder.

    This mirrors the sketch in `docs/api_reference.md` and is intended
    to be extended with any additional hyperparameters you need.
    """

    base_model_name: str
    level_sizes: Dict[int, int]
    proj_dim: int = 256
    lambda_hier: float = 1.0
    lambda_path: float = 0.0
    lambda_contrast: float = 0.0


class TOLBERT(nn.Module):
    """
    BERT/RoBERTa-style encoder with multi-level heads and a projection head.

    Forward returns:
      - loss: optional aggregated loss (MLM + hierarchical) if labels provided.
      - loss_components: dict with individual loss terms.
      - mlm_logits: token-level logits for MLM.
      - level_logits: dict[level_str] -> logits over nodes at that level.
      - proj: normalized CLS projection for contrastive / retrieval use.
    """

    def __init__(self, config: TOLBERTConfig):
        super().__init__()
        self.config = config

        # Backbone encoder
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        hidden_dim = self.encoder.config.hidden_size

        # Simple MLM head (you can replace with the base model's own head)
        self.mlm_head = nn.Linear(hidden_dim, self.encoder.config.vocab_size)

        # Hierarchical heads per level
        self.level_heads = nn.ModuleDict()
        for level, size in config.level_sizes.items():
            self.level_heads[str(level)] = nn.Linear(hidden_dim, size)

        # Contrastive projection head
        self.proj = nn.Linear(hidden_dim, config.proj_dim)

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, config: Optional[TOLBERTConfig] = None
    ) -> "TOLBERT":
        """
        Very minimal `from_pretrained` helper.

        - `checkpoint` is expected to be a `torch.save(model.state_dict())` file.
        - You must supply a `config` that matches the saved model.
        """
        if config is None:
            raise ValueError("TOLBERT.from_pretrained requires a TOLBERTConfig.")

        model = cls(config)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_mlm: Optional[torch.Tensor] = None,
        level_targets: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels_mlm: (batch, seq_len) with -100 for non-MLM positions.
            level_targets: dict {level(int): tensor(batch,)} with class indices.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)
        cls = hidden_states[:, 0, :]  # (batch, hidden)

        # MLM logits
        mlm_logits = self.mlm_head(hidden_states)  # (batch, seq, vocab)

        # Hierarchical logits
        level_logits: Dict[str, torch.Tensor] = {}
        for level, head in self.level_heads.items():
            level_logits[level] = head(cls)  # (batch, C_level)

        # Contrastive projection
        proj = nn.functional.normalize(self.proj(cls), dim=-1)

        loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, torch.Tensor] = {}

        # MLM loss
        if labels_mlm is not None:
            mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                labels_mlm.view(-1),
            )
            loss_dict["mlm"] = mlm_loss

        # Hierarchical classification loss
        if level_targets is not None:
            ce = nn.CrossEntropyLoss()
            for level_int, targets in level_targets.items():
                logits = level_logits[str(level_int)]
                level_loss = ce(logits, targets)
                loss_dict[f"level_{level_int}"] = level_loss

        if loss_dict:
            # Weighted sum if you later want to integrate lambda_hier/lambda_path.
            loss = sum(loss_dict.values())

        return {
            "loss": loss,
            "loss_components": loss_dict,
            "mlm_logits": mlm_logits,
            "level_logits": level_logits,
            "proj": proj,
        }


