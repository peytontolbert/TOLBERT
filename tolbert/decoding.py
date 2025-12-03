from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def greedy_hierarchical_decode(
    level_logits: Dict[str, torch.Tensor],
    parent_to_children: Dict[int, Dict[int, List[int]]],
    levels: Optional[Sequence[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Greedy top-down decoding of a hierarchical path with child masking.

    This helper mirrors the inference-time procedure described in the TOLBERT
    paper: for each level, predictions are restricted to children of the
    previously chosen parent whenever a parent->children mapping is provided.

    Args:
        level_logits:
            Mapping from level identifier (as used in TOLBERT.level_heads,
            typically the string form of an integer like \"1\", \"2\", ...) to
            a tensor of shape (batch, C_level) with logits for that level.
        parent_to_children:
            Mapping:
                level_int -> { parent_idx -> [child_idx, ...] }
            where indices are in the same class-index space as the logits for
            that level. For example, for level 2, parent indices live at
            level 1 and child indices live at level 2.
        levels:
            Optional explicit decoding order, e.g. [1, 2, 3]. If omitted, the
            function sorts the integer level keys found in `level_logits`.

    Returns:
        dict[level_int] -> tensor(batch,) of predicted class indices per level.

    Notes:
        - If no children are registered for a given (level, parent_idx), the
          decoder falls back to an unconstrained argmax over that level's
          logits for that example.
        - This helper does not assume anything about how node ids are
          assigned; it only assumes that the indices in `parent_to_children`
          are valid indices into the corresponding logits.
    """
    if not level_logits:
        return {}

    # Determine decoding order
    if levels is not None and len(levels) > 0:
        order = list(levels)
    else:
        order = sorted(int(k) for k in level_logits.keys())

    # Sanity: ensure we have logits for all requested levels
    order = [lvl for lvl in order if str(lvl) in level_logits]
    if not order:
        return {}

    batch_size = next(iter(level_logits.values())).size(0)
    device = next(iter(level_logits.values())).device

    preds: Dict[int, torch.Tensor] = {}

    # First level: plain argmax
    first_level = order[0]
    first_logits = level_logits[str(first_level)]
    preds[first_level] = first_logits.argmax(dim=-1)

    # Subsequent levels: mask by children whenever possible
    for idx in range(1, len(order)):
        level = order[idx]
        prev_level = order[idx - 1]

        logits = level_logits[str(level)]  # (B, C_level)
        parent_map = parent_to_children.get(level, {})

        # Default: unconstrained argmax for all examples
        level_preds = logits.argmax(dim=-1)

        if parent_map:
            # Refine predictions where we know the parent->children mapping.
            refined: List[int] = []
            parent_preds = preds[prev_level]
            for b in range(batch_size):
                parent_idx = int(parent_preds[b].item())
                children = parent_map.get(parent_idx)
                if not children:
                    # No constraints for this parent; keep unconstrained argmax.
                    refined.append(int(level_preds[b].item()))
                    continue

                # Restrict logits to the valid children for this parent.
                child_indices = torch.tensor(children, dtype=torch.long, device=device)
                # Guard against out-of-range indices.
                child_indices = child_indices[
                    (child_indices >= 0) & (child_indices < logits.size(-1))
                ]
                if child_indices.numel() == 0:
                    refined.append(int(level_preds[b].item()))
                    continue

                child_logits = logits[b, child_indices]
                rel_idx = int(torch.argmax(child_logits).item())
                refined.append(int(child_indices[rel_idx].item()))

            level_preds = torch.tensor(refined, device=device, dtype=torch.long)

        preds[level] = level_preds

    return preds



