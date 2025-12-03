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


def _normalize_paths_arg(
    paths: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]],
) -> List[List[List[int]]]:
    """
    Normalize `paths` into a list of per-example path sets.

    Accepted input formats:
      - Tree-style (single path per example):
            paths = [[v0, v1, ...], [v0, v1', ...], ...]
      - DAG-style (multiple valid paths per example):
            paths = [
              [[v0, v1, ...], [v0, v1', ...]],
              [[v0, u1, ...]],
              ...
            ]

    Output format:
        List[ List[List[int]] ], where outer index is batch example and each
        inner list is one valid path for that example.
    """
    norm: List[List[List[int]]] = []
    for p in paths:
        if not p:
            norm.append([])
            continue
        first = p[0]
        # If first element is an int, we have a single path.
        if isinstance(first, int):
            norm.append([[int(x) for x in p]])  # type: ignore[arg-type]
        else:
            # Assume list/tuple of paths.
            path_set: List[List[int]] = []
            for sub in p:  # type: ignore[assignment]
                if isinstance(sub, (list, tuple)):
                    path_set.append([int(x) for x in sub])
            norm.append(path_set)
    return norm


def _compute_shared_depth_sets(
    paths_i: List[List[int]],
    paths_j: List[List[int]],
) -> int:
    """
    Deepest shared depth between two *sets* of paths (for DAGs).

    We take the maximum shared depth over all path pairs (pi in paths_i,
    pj in paths_j).
    """
    max_depth = 0
    for pi in paths_i:
        for pj in paths_j:
            d = _compute_shared_depth(pi, pj)
            if d > max_depth:
                max_depth = d
    return max_depth


def tree_contrastive_loss(
    embeddings: torch.Tensor,
    paths: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Depth-weighted tree-aware InfoNCE-style contrastive loss.

    This implements the hierarchical supervised contrastive objective
    described in the paper, extended to handle DAGs / multiple paths per
    example:

      - Each example i has one or more valid paths
            (y_1^{(i,m)}, ..., y_L^{(i,m)})
        through the ontology graph.
      - For a pair (i, j), define depth(i, j) as the deepest level at
        which they share the same label beyond the root **along any**
        of their respective paths.
      - Positives P(i) are all j != i with depth(i, j) >= 1.
      - Each positive pair is weighted by w_ij ∝ depth(i, j).
      - Negatives are examples with depth(i, j) == 0 (only root shared).

    Args:
        embeddings: (batch, dim) L2-normalized embeddings (e.g., proj outputs).
        paths: sequence of per-example paths, which may be:
            - a single path per example: [[v0, v1, ...], ...], or
            - multiple paths per example: [[[v0, v1, ...], [v0, v1', ...]], ...]
        temperature: softmax temperature for contrastive logits.
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    # Normalize paths into per-example path sets to support DAGs.
    path_sets = _normalize_paths_arg(paths)

    # Compute pairwise shared depths over path sets.
    depth_mat = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            d = _compute_shared_depth_sets(path_sets[i], path_sets[j])
            depth_mat[i, j] = float(d)

    # Positives: depth >= 1
    pos_mask = depth_mat >= 1.0  # (B, B)

    # If no pairs share any non-root ancestor, the contrastive signal is empty.
    if not pos_mask.any():
        return torch.tensor(0.0, device=device)

    # Similarity matrix
    sim = embeddings @ embeddings.T  # (B, B)
    sim = sim / temperature

    # Exclude self from denominators by setting diag to a very large
    # negative value (rather than -inf) to avoid NaNs when multiplied
    # by zero weights later on.
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # Log-softmax-like normalization per row
    logsumexp = torch.logsumexp(sim, dim=1, keepdim=True)  # (B, 1)
    log_probs = sim - logsumexp  # (B, B)

    # Depth-based weights w_ij = depth(i,j) / L_i for positives, where
    # L_i is the maximum usable depth for anchor i (excluding root).
    # For DAGs, we use the maximum depth over that example's path set.
    anchor_depths: List[int] = []
    for path_set in path_sets:
        max_len = 1
        for p in path_set:
            if len(p) - 1 > max_len:
                max_len = len(p) - 1
        anchor_depths.append(max(1, max_len))
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
