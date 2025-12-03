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
        paths: Optional[Sequence[Sequence[int]]],
    ) -> Optional[torch.Tensor]:
        """
        Batch-local implementation of the path-consistency loss described
        in the paper / training docs.

        For each adjacent level pair (ℓ-1, ℓ), we:
          - build a child -> {parent,...} mapping from the observed `paths`
          - compute softmax over level-ℓ logits
          - for each example, identify which level-ℓ nodes are *invalid*
            given its level-(ℓ-1) parent
          - penalize probability mass assigned to invalid nodes:

                L_path ≈ Σ_{examples, ℓ} Σ_{i ∈ I_{ℓ}^{invalid}} p_ℓ(i)

        This matches the "penalize mass on invalid children" formulation:
        only nodes whose recorded parents do *not* match the example's
        parent at level ℓ-1 contribute to the loss. Nodes with unknown
        parent assignments in the batch are treated as neutral (neither
        explicitly valid nor invalid).
        """
        if paths is None:
            return None

        if len(self._sorted_levels) < 2:
            return None

        device = next(self.parameters()).device

        # Convert paths (python lists) into a tensor for easier indexing.
        # paths are assumed to be full node ids including root at index 0.
        max_len = max(len(p) for p in paths)
        batch_size = len(paths)
        path_tensor = torch.full(
            (batch_size, max_len),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        for i, p in enumerate(paths):
            path_tensor[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)

        total_mass = 0.0
        num_terms = 0

        # We assume level index ℓ in the paper corresponds to path index ℓ,
        # with ℓ = 1..L (and index 0 being the root, which we do not have a head for).
        # For model heads we use integer "levels" taken from config.level_sizes.
        for idx in range(1, len(self._sorted_levels)):
            level_prev = self._sorted_levels[idx - 1]
            level_curr = self._sorted_levels[idx]

            if str(level_prev) not in level_logits or str(level_curr) not in level_logits:
                continue

            logits_curr = level_logits[str(level_curr)]  # (B, C_curr)

            probs_curr = F.softmax(logits_curr, dim=-1)  # (B, C_curr)

            # Build a child -> parent map for this level-pair from batch paths.
            # We treat class indices at level ℓ as node ids at that level, and
            # assume a single canonical parent per child (tree setting). If the
            # same child appears with different parents in the batch, the last
            # observed parent wins, which is acceptable as long as the ontology
            # is a proper tree as in the main TOLBERT setup.
            child_parent: Dict[int, int] = {}
            for b in range(batch_size):
                if level_prev >= path_tensor.size(1) or level_curr >= path_tensor.size(1):
                    continue

                parent_id = path_tensor[b, level_prev].item()
                child_id = path_tensor[b, level_curr].item()
                if parent_id < 0 or child_id < 0:
                    continue
                child_parent[child_id] = parent_id

            if not child_parent:
                continue

            num_classes = logits_curr.size(-1)

            # For each example, measure probability mass on invalid children.
            for b in range(batch_size):
                if level_prev >= path_tensor.size(1) or level_curr >= path_tensor.size(1):
                    continue

                parent_id = path_tensor[b, level_prev].item()
                child_id = path_tensor[b, level_curr].item()
                if parent_id < 0 or child_id < 0:
                    continue

                # Construct invalid mask for this example:
                #  - valid if this class has the current parent_id as its parent
                #  - invalid if it has a *different* parent id
                #  - neutral (ignored) if we never observed a parent for this class id
                invalid_indices: list[int] = []
                for cls_idx in range(num_classes):
                    p = child_parent.get(cls_idx)
                    if p is None:
                        # No parent information for this class in the batch;
                        # do not treat it as explicitly invalid.
                        continue
                    if parent_id != p:
                        invalid_indices.append(cls_idx)

                if not invalid_indices:
                    continue

                cls_tensor = torch.tensor(
                    invalid_indices,
                    dtype=torch.long,
                    device=device,
                )
                invalid_mass = probs_curr[b, cls_tensor].sum()
                total_mass = total_mass + invalid_mass
                num_terms += 1

        if num_terms == 0:
            return None

        return total_mass / float(num_terms)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_mlm: Optional[torch.Tensor] = None,
        level_targets: Optional[Dict[int, torch.Tensor]] = None,
        paths: Optional[Sequence[Sequence[int]]] = None,
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
