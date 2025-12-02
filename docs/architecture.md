## TOLBERT Architecture

TOLBERT is a BERT-like encoder equipped with **multi-level heads** that align representations with a Tree-of-Life taxonomy. It retains standard masked language modeling while adding hierarchy-aware objectives.

### 1. Base Encoder

- **Backbone**: Any HuggingFace-style BERT/RoBERTa-like model.
- **Input**:
  - Tokenized text (code, comments, paper sections).
  - Standard `input_ids`, `attention_mask`, optional token type IDs.
- **Output**:
  - Contextual hidden states \(H \in \mathbb{R}^{T \times d}\) for sequence length \(T\), hidden size \(d\).
  - A sequence representation \(h\) from the `[CLS]` token:
    - \(h = H_{\text{[CLS]}} \in \mathbb{R}^{d}\).

This encoder is trained with MLM plus hierarchical objectives, and can be fine-tuned for downstream tasks.

### 2. Multi-Level Classification Heads

For explicit tree depths \(k \in \{1, ..., K\}\) that you choose to supervise:

- Let \(C_k\) be the number of nodes at level \(k\).
- Define a classification head per level:

- \(z_k = W_k h + b_k \in \mathbb{R}^{C_k}\),
  - where \(W_k \in \mathbb{R}^{C_k \times d}\), \(b_k \in \mathbb{R}^{C_k}\).
- These logits parameterize a distribution over level-\(k\) nodes:
  - \(p_k = \text{softmax}(z_k)\).

Typical semantic assignment:

- **Level 1 head**: domain (e.g. AI-Code, AI-Papers, Systems).
- **Level 2 head**: subdomain/stack (e.g. Transformers, RL, Diffusion).
- **Level 3 head**: specific repo or paper.
- **Level 4+ head (optional)**: file/function/section classes.

Level-specific supervision comes from the path \(\pi(x)\) assigned in the Tree-of-Life.

### 3. Optional Hierarchical Embedding Layer

Instead of (or in addition to) pure classification, you can maintain **node embedding matrices**:

- For each level \(k\):
  - \(E_k \in \mathbb{R}^{C_k \times d_h}\) (node embeddings).
- A shared projection of sequence embedding into this space:
  - \(e(x) = P h \in \mathbb{R}^{d_h}\),
  - where \(P \in \mathbb{R}^{d_h \times d}\).

You can compute similarities between \(e(x)\) and each node embedding at that level:

- \(\text{sim}_k(i) = \text{cosine}(e(x), E_k[i])\),
  - and turn this into a distribution or a contrastive objective.

This enables:

- Hyperbolic or tree-aware embeddings.
- Distance-based training where closeness reflects tree distance.

### 4. Masked Language Modeling Head

- Standard token-level MLM head on top of the encoder:
  - `hidden_states -> mlm_head -> vocab_logits`.
- Typically implemented as:
  - `Linear(hidden_dim, vocab_size)` (or reuse HF’s MLM head).
- Loss:
  - Cross-entropy over masked tokens with `ignore_index = -100`.

MLM provides the semantic backbone; hierarchy-aware losses act as regularization/prior.

### 5. Contrastive Projection Head

For tree-aware contrastive learning:

- Add a projection layer:
  - \(g(x) = \text{norm}(Q h)\),
  - where \(Q \in \mathbb{R}^{d_c \times d}\) and `norm` is L2 normalization.
- Use this as the embedding for InfoNCE-style contrastive loss, where:
  - Positives: spans on **similar branches** (e.g., same repo, same paper, or same subdomain).
  - Negatives: spans on **distant branches** in the tree.

### 6. Reference PyTorch / HF Skeleton

The following class is a minimal working skeleton for TOLBERT:

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class TOLBERT(nn.Module):
    def __init__(self, base_model_name, level_sizes, hidden_dim=None, proj_dim=256):
        """
        base_model_name: HuggingFace model id, e.g. 'bert-base-uncased' or 'roberta-base'
        level_sizes: dict like {1: C1, 2: C2, 3: C3, ...}
        hidden_dim: encoder hidden size (if None, inferred from base model)
        proj_dim: dimension of contrastive embedding
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_dim = hidden_dim or self.encoder.config.hidden_size

        # MLM head (can also reuse HF's built-in MLM head if your base model has one)
        self.mlm_head = nn.Linear(hidden_dim, self.encoder.config.vocab_size)

        # Hierarchical heads
        self.level_heads = nn.ModuleDict()
        for level, size in level_sizes.items():
            self.level_heads[str(level)] = nn.Linear(hidden_dim, size)

        # Contrastive projection
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels_mlm=None,
        level_targets=None,
    ):
        """
        level_targets: dict {level(int): tensor(batch,)} of node indices for each level.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state      # (batch, seq, hidden)
        cls = hidden_states[:, 0, :]                  # [CLS] representation

        # MLM logits (token-level)
        mlm_logits = self.mlm_head(hidden_states)     # (batch, seq, vocab)

        # Hierarchical logits
        level_logits = {}
        for level, head in self.level_heads.items():
            level_logits[level] = head(cls)           # (batch, C_level)

        # Contrastive embedding
        proj = nn.functional.normalize(self.proj(cls), dim=-1)

        loss = None
        loss_dict = {}

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
            for level, targets in level_targets.items():
                logits = level_logits[str(level)]
                level_loss = ce(logits, targets)
                loss_dict[f"level_{level}"] = level_loss

        # Combine losses
        if loss_dict:
            loss = sum(loss_dict.values())

        return {
            "loss": loss,
            "loss_components": loss_dict,
            "mlm_logits": mlm_logits,
            "level_logits": level_logits,
            "proj": proj,
        }
```

For loss design, training pipeline, and usage patterns, see `training.md` and `usage.md`.

### 7. Dynamic Tree / New Nodes

In practice, your Tree-of-Life will **evolve** over time as you add new repos, papers, functions, or concepts. Since each level head uses a fixed number of classes \(C_k\), you need a strategy to grow these heads without retraining from scratch.

- **Extending classification heads**:
  - When adding new nodes at level \(k\), extend the corresponding weight matrix and bias:
    - \(W_k \in \mathbb{R}^{C_k \times d} \rightarrow W_k' \in \mathbb{R}^{(C_k + \Delta C_k) \times d}\).
    - Initialize new rows of \(W_k'\) and entries of \(b_k'\) using:
      - Parent’s weight vector.
      - Or the average of sibling/neighbor weights.
  - This preserves existing logits while giving reasonable priors for new nodes.

- **Extending node embeddings** (if using \(E_k\)):
  - Append new rows to \(E_k\) for each new node.
  - Initialize from:
    - Parent node embedding.
    - Or mean of embeddings for similar nodes (e.g., same repo family or topic cluster).

- **Fine-tuning strategy**:
  - Option A: **Local fine-tune**:
    - Freeze the backbone and all existing heads.
    - Only train parameters associated with new nodes (new rows in \(W_k\), \(E_k\)).
  - Option B: **Branch-aware fine-tune**:
    - Unfreeze the relevant level heads (and optionally a few top encoder layers).
    - Train on data from both old and new branches with a smaller learning rate.

- **When to retrain more aggressively**:
  - If the taxonomy changes significantly (e.g., large new domains, major re-clustering), consider:
    - Recomputing the Tree-of-Life.
    - Reinitializing affected level heads.
    - Running a brief global fine-tuning stage to realign geometry.

#### Tree versioning and migration

As you change the taxonomy, treat the Tree-of-Life itself as a **versioned artifact**:

- **Store a `tree_version` alongside all artifacts**:
  - Each training run, span label, and stored embedding should record which `tree_version` (e.g., `tree_v1`, `tree_v2`) it was produced under.
  - When you change the tree (e.g., split a domain, move repos between branches), bump the version and persist the mapping between **old node IDs** and **new node IDs**.
- **Maintain a compatibility / mapping layer**:
  - Keep a small mapping table from `old_node_id -> new_node_id(s)` (plus metadata like “split into”, “merged into”).
  - For **read-time migration**:
    - When loading old span labels or embeddings, map their node IDs through this table to a current representation.
    - For “split” cases, you can:
      - Route old embeddings to a **coarser ancestor** that is stable across versions, or
      - Treat them as ambiguous and down-weight them in training/eval.
- **Handling mixed-version embeddings in a vector DB**:
  - Tag each stored vector with its `tree_version` and, if relevant, its path under that version.
  - When serving retrieval:
    - Option 1 (strict): restrict queries to embeddings from a **single active tree version** (and gradually reindex old embeddings under the new tree).
    - Option 2 (compat): for older versions, map their paths to the current tree using the compatibility layer and store or cache a **logical path** in the current version for routing and evaluation.
  - Over time, you can schedule **offline re-encoding / reindexing** passes to move the bulk of your store to the latest `tree_version`, treating the compatibility layer as a bridge rather than a permanent crutch.

This setup lets you treat TOLBERT as a **living ontology encoder**, where the tree and heads can grow incrementally while preserving most of the learned structure and keeping old embeddings and labels interpretable across versions.


