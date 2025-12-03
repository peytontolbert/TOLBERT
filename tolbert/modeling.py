from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Overall weight on hierarchical classification loss
    lambda_hier: float = 1.0
    # Weight on path-consistency regularization term
    lambda_path: float = 0.0
    # Kept for completeness; used in the training script when adding contrastive loss.
    lambda_contrast: float = 0.0


class TOLBERT(nn.Module):
    """
    BERT/RoBERTa-style encoder with multi-level heads and a projection head.
    
    Forward returns:
      - loss: aggregated loss (MLM + hierarchical + optional path) if labels provided.
      - loss_components: dict with individual loss terms.
      - mlm_logits: token-level logits for MLM.
      - level_logits: dict[level_str] -> logits over nodes at that level.
      - proj: normalized CLS projection for contrastive / retrieval use.
    """

    def __init__(self, config: TOLBERTConfig):
        super().__init__()
        self.config = config

        # Backbone encoder
        self.encoder = AutoModel.from_pretrained(
            config.base_model_name,
            cache_dir="/data/checkpoints/",  # noqa: E501
        )
        hidden_dim = self.encoder.config.hidden_size

        # Simple MLM head (you can replace with the base model's own head)
        self.mlm_head = nn.Linear(hidden_dim, self.encoder.config.vocab_size)

        # Hierarchical heads per level
        self.level_heads = nn.ModuleDict()
        for level, size in config.level_sizes.items():
            self.level_heads[str(level)] = nn.Linear(hidden_dim, size)

        # Contrastive projection head
        self.proj = nn.Linear(hidden_dim, config.proj_dim)

        # Pre-sort levels once for stable ordering in losses
        self._sorted_levels = sorted(config.level_sizes.keys())

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        config: Optional[TOLBERTConfig] = None,
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

    def _compute_path_consistency_loss(
        self,
        level_logits: Dict[str, torch.Tensor],
        paths: Optional[Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]],
    ) -> Optional[torch.Tensor]:
        """
        Batch-local implementation of the path-consistency loss described
        in the paper / training docs, extended to handle DAGs / multi-parent
        ontologies.

        For each adjacent level pair (ℓ-1, ℓ), we:
          - build a parent -> {child,...} mapping from the observed `paths`
            across the batch (allowing multiple parents per child),
          - compute softmax over level-ℓ logits,
          - roll level-ℓ probabilities up to the parent space and
            encourage agreement between:
                - the distribution implied by children, and
                - the model's own distribution at level ℓ-1.

        In a pure tree, each child has exactly one parent; in a DAG, a child
        may appear under multiple parents across the provided paths. This
        implementation treats all observed parent-child relations as valid.
        """
        if paths is None:
            return None

        if len(self._sorted_levels) < 2:
            return None

        device = next(self.parameters()).device

        # Normalize paths into per-example path sets to support DAGs.
        # Each element in `path_sets` is a list of paths (lists of node ids).
        path_sets: list[list[list[int]]] = []
        for p in paths:
            if not p:
                path_sets.append([])
                continue
            first = p[0]
            if isinstance(first, int):
                path_sets.append([[int(x) for x in p]])  # type: ignore[arg-type]
            else:
                path_set: list[list[int]] = []
                for sub in p:  # type: ignore[assignment]
                    if isinstance(sub, (list, tuple)):
                        path_set.append([int(x) for x in sub])
                path_sets.append(path_set)

        batch_size = len(path_sets)

        total_kl = 0.0
        num_terms = 0

        # We assume level index ℓ in the paper corresponds to path index ℓ,
        # with ℓ = 1..L (and index 0 being the root, which we do not have a head for).
        # For model heads we use integer "levels" taken from config.level_sizes.
        for idx in range(1, len(self._sorted_levels)):
            level_prev = self._sorted_levels[idx - 1]
            level_curr = self._sorted_levels[idx]

            if str(level_prev) not in level_logits or str(level_curr) not in level_logits:
                continue

            logits_prev = level_logits[str(level_prev)]  # (B, C_prev)
            logits_curr = level_logits[str(level_curr)]  # (B, C_curr)

            probs_prev = F.softmax(logits_prev, dim=-1)  # (B, C_prev)
            probs_curr = F.softmax(logits_curr, dim=-1)  # (B, C_curr)

            # Build parent -> children map for this level-pair from batch paths.
            parent_to_children: Dict[int, set[int]] = {}
            for b in range(batch_size):
                for path in path_sets[b]:
                    if len(path) <= max(level_prev, level_curr):
                        continue
                    parent_id = path[level_prev]
                    child_id = path[level_curr]
                    if parent_id < 0 or child_id < 0:
                        continue
                    if parent_id not in parent_to_children:
                        parent_to_children[parent_id] = set()
                    parent_to_children[parent_id].add(child_id)

            if not parent_to_children:
                continue

            num_parents = logits_prev.size(-1)
            rolled = torch.zeros_like(probs_prev)
            for parent_id, children_ids in parent_to_children.items():
                if parent_id < 0 or parent_id >= num_parents:
                    continue
                children_idx = torch.tensor(
                    list(children_ids),
                    dtype=torch.long,
                    device=device,
                )
                children_idx = children_idx[
                    (children_idx >= 0) & (children_idx < probs_curr.size(-1))
                ]
                if children_idx.numel() == 0:
                    continue
                rolled[:, parent_id] = probs_curr[:, children_idx].sum(dim=-1)

            # Renormalize rolled-up distribution to avoid degenerate zeros.
            rolled_sum = rolled.sum(dim=-1, keepdim=True)
            rolled_sum = torch.clamp(rolled_sum, min=1e-8)
            rolled = rolled / rolled_sum

            # Ensure numerical stability for KL
            probs_prev_clamped = torch.clamp(probs_prev, min=1e-8)
            rolled_clamped = torch.clamp(rolled, min=1e-8)

            kl = F.kl_div(
                probs_prev_clamped.log(),
                rolled_clamped,
                reduction="batchmean",
            )
            total_kl = total_kl + kl
            num_terms += 1

        if num_terms == 0:
            return None

        return total_kl / float(num_terms)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_mlm: Optional[torch.Tensor] = None,
        level_targets: Optional[Dict[int, torch.Tensor]] = None,
        paths: Optional[Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels_mlm: (batch, seq_len) with -100 for non-MLM positions.
            level_targets: dict {level(int): tensor(batch,)} with class indices.
            paths: optional sequence of per-example label paths, each a
                   sequence of node ids [root, c1, c2, ..., cL]. Used for
                   path-consistency loss.
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
        proj = F.normalize(self.proj(cls), dim=-1)

        loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, torch.Tensor] = {}

        # MLM loss
        mlm_loss = None
        if labels_mlm is not None:
            mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                labels_mlm.view(-1),
            )
            loss_dict["mlm"] = mlm_loss

        # Hierarchical classification loss (sum over levels, excluding ignore_index)
        hier_loss = None
        if level_targets is not None and len(level_targets) > 0:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            per_level_losses = []
            for level_int in self._sorted_levels:
                if level_int not in level_targets:
                    continue
                logits = level_logits[str(level_int)]
                targets = level_targets[level_int]
                level_loss = ce(logits, targets)
                loss_dict[f"level_{level_int}"] = level_loss
                per_level_losses.append(level_loss)

            if per_level_losses:
                hier_loss = torch.stack(per_level_losses).mean()
                loss_dict["hier"] = hier_loss

        # Path-consistency loss (uses predicted distributions only)
        path_loss = None
        if paths is not None:
            path_loss = self._compute_path_consistency_loss(level_logits, paths)
            if path_loss is not None:
                loss_dict["path"] = path_loss

        # Aggregate total loss with configuration weights.
        components = []
        if mlm_loss is not None:
            components.append(mlm_loss)
        if hier_loss is not None and self.config.lambda_hier != 0.0:
            components.append(self.config.lambda_hier * hier_loss)
        if path_loss is not None and self.config.lambda_path != 0.0:
            components.append(self.config.lambda_path * path_loss)

        if components:
            loss = sum(components)

        return {
            "loss": loss,
            "loss_components": loss_dict,
            "mlm_logits": mlm_logits,
            "level_logits": level_logits,
            "proj": proj,
        }
