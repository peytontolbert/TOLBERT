from typing import List, Sequence

import torch


def _compute_shared_depth(pi: Sequence[int], pj: Sequence[int]) -> int:
    """
    Depth of deepest shared label beyond the root.

    We treat index 0 in the path as the root. The shared depth is the
    number of consecutive matching labels starting from index 1.
    """
    max_len = min(len(pi), len(pj))
    depth = 0
    # Start from index 1 to skip the root.
    for idx in range(1, max_len):
        if pi[idx] != pj[idx]:
            break
        depth += 1
    return depth


def tree_contrastive_loss(
    embeddings: torch.Tensor,
    paths: Sequence[Sequence[int]],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Depth-weighted tree-aware InfoNCE-style contrastive loss.

    This implements the hierarchical supervised contrastive objective
    described in the paper:

      - Each example i has a path (y_1^{(i)}, ..., y_L^{(i)}).
      - For a pair (i, j), define depth(i, j) as the deepest level at
        which they share the same label beyond the root.
      - Positives P(i) are all j != i with depth(i, j) >= 1.
      - Each positive pair is weighted by w_ij ∝ depth(i, j).
      - Negatives are examples with depth(i, j) == 0 (only root shared).

    Args:
        embeddings: (batch, dim) L2-normalized embeddings (e.g., proj outputs).
        paths: sequence of per-example paths, each a sequence of node ids.
        temperature: softmax temperature for contrastive logits.
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    # Compute pairwise shared depths
    depth_mat = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            d = _compute_shared_depth(paths[i], paths[j])
            depth_mat[i, j] = float(d)

    # Positives: depth >= 1
    pos_mask = depth_mat >= 1.0  # (B, B)

    # If no pairs share any non-root ancestor, the contrastive signal is empty.
    if not pos_mask.any():
        return torch.tensor(0.0, device=device)

    # Similarity matrix
    sim = embeddings @ embeddings.T  # (B, B)
    sim = sim / temperature

    # Exclude self from denominators by setting diag to very small value
    # (will be masked out later via logsumexp).
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(diag_mask, float("-inf"))

    # Log-softmax-like normalization per row
    logsumexp = torch.logsumexp(sim, dim=1, keepdim=True)  # (B, 1)
    log_probs = sim - logsumexp  # (B, B)

    # Depth-based weights w_ij = depth(i,j) / L_i for positives, where
    # L_i is the maximum usable depth for anchor i (excluding root).
    # This aligns with the paper's suggestion w_ij ∝ depth / L.
    # Compute L_i per-anchor from its own path length.
    anchor_depths = []
    for p in paths:
        # depth is len(p) - 1 at most (index 0 is root)
        anchor_depths.append(max(1, len(p) - 1))
    L = torch.tensor(anchor_depths, dtype=torch.float32, device=device).unsqueeze(1)  # (B, 1)

    # Avoid divide-by-zero; we clamped to at least 1 above.
    norm_depths = depth_mat / L

    weights = torch.zeros_like(depth_mat)
    weights[pos_mask] = norm_depths[pos_mask]

    # For each anchor i, compute a weighted average of log_probs over positives.
    eps = 1e-8
    row_pos_counts = pos_mask.sum(dim=1)  # (B,)
    row_weight_sums = weights.sum(dim=1)  # (B,)

    # Avoid divide-by-zero
    valid_rows = row_pos_counts > 0
    if not valid_rows.any():
        return torch.tensor(0.0, device=device)

    # Normalized weights within each row (only over positives)
    norm_weights = torch.zeros_like(weights)
    norm_weights[pos_mask] = (
        weights[pos_mask]
        / (row_weight_sums.unsqueeze(1).expand_as(weights)[pos_mask] + eps)
    )

    # For each row i, loss_i = - (1/|P(i)|) Σ_j w_ij_norm * log_probs_ij
    weighted_log_probs = (norm_weights * log_probs)  # (B, B)
    row_losses = torch.zeros(batch_size, device=device)

    # Sum over j for each i
    row_losses = -weighted_log_probs.sum(dim=1)
    # Divide by |P(i)| for anchors that have positives
    row_losses = torch.where(
        valid_rows,
        row_losses / (row_pos_counts.to(row_losses.dtype) + eps),
        torch.zeros_like(row_losses),
    )

    # Average over anchors with at least one positive
    loss = row_losses[valid_rows].mean()
    return loss
