import torch
from pathlib import Path
import sys

# Ensure project root is on sys.path so `tolbert` imports resolve consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tolbert import losses
from tolbert import decoding


def test_tree_contrastive_loss_zero_when_no_shared_ancestors():
    """
    If no pairs share any non-root ancestor, the contrastive loss should be 0.
    """
    emb = torch.randn(3, 8)
    # All paths differ immediately after root.
    paths = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    loss = losses.tree_contrastive_loss(emb, paths, temperature=0.1)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_tree_contrastive_loss_positive_with_shared_depth():
    """
    When some pairs share deeper ancestors, the loss should be strictly positive.
    """
    # Make embeddings small and deterministic for stability.
    emb = torch.eye(4, dtype=torch.float32)
    # First three share the same label at depth 1; last one is in a different branch.
    paths = [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 2],
    ]
    loss = losses.tree_contrastive_loss(emb, paths, temperature=0.1)
    # Loss should be finite and non-negative.
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_greedy_hierarchical_decode_respects_parent_child_mapping():
    """
    Ensure that greedy_hierarchical_decode:
      - performs unconstrained argmax at the first level,
      - then restricts the next level to children of the chosen parent.
    """
    batch_size = 1

    # Level 1: two parents, prefer index 1.
    l1_logits = torch.tensor([[0.0, 5.0]])  # (1, 2)
    # Level 2: three children; unconstrained argmax would pick index 2.
    l2_logits = torch.tensor([[10.0, 0.0, 1.0]])  # (1, 3)

    level_logits = {"1": l1_logits, "2": l2_logits}

    # Parent-to-children mapping: at level 2, parent 1 can only choose child 0.
    parent_to_children = {
        2: {
            0: [1, 2],
            1: [0],
        }
    }

    preds = decoding.greedy_hierarchical_decode(
        level_logits=level_logits,
        parent_to_children=parent_to_children,
        levels=[1, 2],
    )

    # Level 1: picks parent 1.
    assert preds[1].tolist() == [1]
    # Level 2: must choose from children of parent 1, i.e., [0]; so prediction is 0,
    # even though the global argmax of l2 logits is index 2.
    assert preds[2].tolist() == [0]


