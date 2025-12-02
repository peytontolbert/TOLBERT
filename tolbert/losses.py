from typing import List, Sequence

import torch


def tree_contrastive_loss(
    embeddings: torch.Tensor,
    paths: Sequence[Sequence[int]],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Tree-aware InfoNCE-style contrastive loss.

    Args:
        embeddings: (batch, dim) L2-normalized embeddings (e.g., proj outputs).
        paths: sequence of per-example paths, each a sequence of node ids.
               For now we use a simple rule: positives share the same node at
               level 3 (index 3 in the path) if that level exists.
        temperature: softmax temperature for contrastive logits.
    """
    batch_size = embeddings.size(0)

    # Similarity matrix
    sim = embeddings @ embeddings.T  # (batch, batch)

    # Build positive mask
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            pi = paths[i]
            pj = paths[j]
            # Require at least 4 levels and same node at level 3.
            if len(pi) > 3 and len(pj) > 3 and pi[3] == pj[3]:
                pos_mask[i, j] = True

    logits = sim / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # numerical stability

    exp_logits = torch.exp(logits)
    # Denominator: all except self
    denom = exp_logits.sum(dim=1) - torch.exp(logits.diag())

    # Numerator: sum over positives
    num = (exp_logits * pos_mask).sum(dim=1) + 1e-8

    loss = -torch.log(num / (denom + 1e-8)).mean()
    return loss


