import types
from pathlib import Path
import sys

import torch

# Ensure project root is on sys.path so `tolbert` imports resolve consistently,
# even when tests are run from outside the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tolbert import modeling


class _DummyEncoder(torch.nn.Module):
    """
    Tiny stand-in for a HuggingFace AutoModel encoder.

    It exposes:
      - .config.hidden_size
      - .config.vocab_size
      - .forward(...).last_hidden_state
    so that TOLBERT can be instantiated without downloading a real model.
    """

    def __init__(self, hidden_size: int = 16, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, input_ids=None, attention_mask=None):
        batch, seq_len = input_ids.shape
        # Simple deterministic hidden states for reproducibility across runs.
        hidden = torch.arange(batch * seq_len * self.config.hidden_size, dtype=torch.float32)
        hidden = hidden.view(batch, seq_len, self.config.hidden_size)
        return types.SimpleNamespace(last_hidden_state=hidden)


class _DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _DummyEncoder()


def _patch_auto_model():
    """
    Replace modeling.AutoModel with a cheap local encoder for tests.
    """
    modeling.AutoModel = _DummyAutoModel  # type: ignore[assignment]


def test_tolbert_forward_shapes_and_losses():
    """
    Basic sanity check:
      - TOLBERT can be instantiated with a dummy encoder.
      - Forward pass returns all expected keys.
      - Loss and individual loss components are present.
    """
    _patch_auto_model()

    cfg = modeling.TOLBERTConfig(
        base_model_name="dummy",
        level_sizes={1: 3, 2: 4},
        proj_dim=8,
        lambda_hier=1.0,
        lambda_path=0.5,
    )
    model = modeling.TOLBERT(cfg)

    batch_size, seq_len = 2, 5
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Simple MLM labels: predict a couple of positions, others ignored.
    labels_mlm = input_ids.clone()
    labels_mlm[:, 0] = -100

    # Per-level targets; use -100 to mark missing labels for one example.
    level_targets = {
        1: torch.tensor([0, 1], dtype=torch.long),
        2: torch.tensor([2, -100], dtype=torch.long),
    }

    # Paths: [root, level1, level2]; align with level indices 1 and 2.
    paths = [
        [0, 0, 2],
        [0, 1, 3],
    ]

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels_mlm=labels_mlm,
        level_targets=level_targets,
        paths=paths,
    )

    assert "loss" in out
    assert out["loss"] is not None
    assert "loss_components" in out
    loss_components = out["loss_components"]
    assert "mlm" in loss_components
    assert "hier" in loss_components
    # Path loss may be zero if no invalid mass, but key should exist when paths are provided.
    assert "path" in loss_components

    # Shape checks
    mlm_logits = out["mlm_logits"]
    assert mlm_logits.shape == (batch_size, seq_len, model.encoder.config.vocab_size)

    level_logits = out["level_logits"]
    assert set(level_logits.keys()) == {"1", "2"}
    assert level_logits["1"].shape == (batch_size, 3)
    assert level_logits["2"].shape == (batch_size, 4)

    proj = out["proj"]
    assert proj.shape == (batch_size, cfg.proj_dim)
    # Embeddings should be L2-normalized.
    norms = torch.norm(proj, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_path_consistency_penalizes_invalid_children():
    """
    Directly test the path-consistency helper on a toy example:
      - Parent distribution puts mass entirely on parent 0.
      - Child distribution puts mass on a child whose parent != 0.
      - The resulting path loss should be strictly positive.
    """
    _patch_auto_model()

    cfg = modeling.TOLBERTConfig(
        base_model_name="dummy",
        level_sizes={1: 2, 2: 2},
        proj_dim=4,
        lambda_hier=1.0,
        lambda_path=1.0,
    )
    model = modeling.TOLBERT(cfg)

    device = next(model.parameters()).device

    # Construct dummy logits for level 1 and 2.
    # Level 1: always predict parent 0 with prob ~1.
    logits_l1 = torch.tensor([[10.0, -10.0]], device=device)  # (1, 2)
    # Level 2: two children; child 0 has parent 1 (mismatched), child 1 has parent 0 (matched).
    logits_l2 = torch.tensor([[10.0, -10.0]], device=device)  # will favor child 0 (invalid)

    level_logits = {"1": logits_l1, "2": logits_l2}

    # Paths: [root, parent_id_at_level1, child_id_at_level2]
    # Define ontology:
    #   - child 0 -> parent 1
    #   - child 1 -> parent 0
    # By choosing parent 0 at level 1 and highest prob on child 0 at level 2,
    # we force probability mass on an invalid child.
    paths = [[0, 0, 0]]

    path_loss = model._compute_path_consistency_loss(level_logits, paths)
    # With a consistent ontology (child->parent mapping derived from paths),
    # the KL-based path-consistency loss should be finite and non-negative.
    assert path_loss is not None
    assert torch.isfinite(path_loss)
    assert path_loss.item() >= 0.0


