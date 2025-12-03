## Training TOLBERT

This document describes the training objectives and pipeline for TOLBERT.

### 1. Objective Overview

TOLBERT optimizes a joint loss:

- \(L = L_{\text{MLM}} + \lambda_{\text{hier}} L_{\text{hier}} + \lambda_{\text{path}} L_{\text{path}} + \lambda_{\text{contrast}} L_{\text{contrast}}\)

Where:

- **\(L_{\text{MLM}}\)**: standard masked language modeling loss.
- **\(L_{\text{hier}}\)**: multi-level cross-entropy against tree nodes.
- **\(L_{\text{path}}\)**: path-consistency loss enforcing ancestor-descendant coherence.
- **\(L_{\text{contrast}}\)**: tree-aware contrastive loss (optional but powerful).

Hyperparameters \(\lambda_{\cdot}\) control relative weighting.

### 2. Masked Language Modeling (MLM)

- **Input**: tokenized spans with 15% of tokens masked or corrupted, as in standard BERT.
- **Head**: MLM head mapping token hidden states to vocabulary logits.
- **Loss**:
  - Cross-entropy over masked positions only (`ignore_index = -100` for unmasked tokens).

Role:

- Provides the base semantic signal over raw code and natural language.

### 3. Hierarchical Cross-Entropy

Each training example \(x\) has a path:

- \(\pi(x) = (v_1, ..., v_K)\), where \(v_k\) is the node at level \(k\).

For each supervised level \(k\):

- TOLBERT produces logits \(z_k\) over all nodes at level \(k\).
- Let \(y_k\) be the index of the correct node \(v_k\) among nodes at level \(k\).
- Define:

- \(L_{\text{hier}} = \sum_{k=1}^{K} \text{CE}(\text{softmax}(z_k), y_k)\).

This trains the model to identify the correct node at each depth of the tree.

### 4. Path Consistency Loss

We want predictions at deeper levels to be **consistent** with ancestors:

- Mass at level \(k+1\) should be concentrated on nodes whose ancestor at level \(k\) matches \(v_k\).

Implementation sketch:

- For each level pair \(k, k+1\):
  - Identify indices \(I_{k+1}^{\text{invalid}}\): level \(k+1\) nodes whose ancestor at level \(k\) is **not** \(v_k\).
  - Let \(p_{k+1}\) be softmax over level \(k+1\) logits.
  - Penalize probability mass assigned to invalid nodes:

  - \(L_{\text{path}} = \sum_{k=1}^{K-1} \sum_{i \in I_{k+1}^{\text{invalid}}} p_{k+1}(i)\).

This encourages local consistency between adjacent depths in the tree.

#### Partially-labeled paths & DAG-like graphs

In practice, paths are often **incomplete** and the underlying graph is closer to a **DAG** than a strict tree:

- **Missing / unknown levels**:
  - Many spans are labeled only up to a shallow depth (e.g., domain/subdomain but not repo/file).
  - Implementation-wise, treat those deeper levels as **unsupervised**:
    - For \(L_{\text{hier}}\): either skip those levels from the sum for that example, or set the per-level target index to an `ignore_index` (e.g. `-100`) so the cross-entropy does not contribute.
    - For \(L_{\text{path}}\): skip any level pair \((k, k+1)\) where one or both levels are unknown for that span.
  
- **DAG reality (multi-parent nodes)**:
  - Your `RepoGraph` / `PaperGraph` may give nodes **multiple plausible parents** (e.g., a repo that sits at the intersection of two domains).
  - The implementation supports this via a richer span format:
    - Use `node_paths` to provide **multiple full paths per span**:
      - `node_paths = [[v_0, ..., v_K], [v_0, ..., v_K'], ...]`.
    - A single **canonical path** (the first) is used to derive per-level targets for \(L_{\text{hier}}\).
    - \(L_{\text{path}}\) and \(L_{\text{contrast}}\) consume the full `node_paths` set:
      - For \(L_{\text{path}}\): parent→children mappings are built from all provided paths, and the KL-based consistency loss is computed over the resulting rolled-up distributions.
      - For \(L_{\text{contrast}}\): the shared depth between two examples is the **maximum** depth over any pair of their paths, so positives/negatives respect multi-parent ancestry.

### 5. Contrastive Tree Loss

To shape geometry in embedding space, define a contrastive loss over projected embeddings:

- Let \(e(x)\) be normalized projection (`proj` output).
- For a batch \(B\) of examples:
  - Define positives based on tree proximity:
    - Same repo (same level-3 node).
    - Or, more generally, tree distance below a threshold.
  - Negatives are other batch elements.

An InfoNCE-style loss:

- \(L_{\text{contrast}} = - \mathbb{E}_{x} \log \frac{\exp(\text{sim}(e(x), e(x^+))/\tau)}{\sum_{x' \in B} \exp(\text{sim}(e(x), e(x'))/\tau)}\)

Where:

- \(\text{sim}\) is cosine similarity, \(\tau\) is a temperature parameter.
- You can generalize to multiple positives, weighting by tree distance.

#### Example Implementation

```python
import torch


def tree_contrastive_loss(embeddings, paths, temperature=0.07):
    """
    embeddings: (batch, dim) L2-normalized.
    paths: list of tuples/lists representing node ids along the path, one per example.
           Example: [(v0, v1, v2, v3), ...]
    """
    batch_size = embeddings.size(0)

    # Similarity matrix
    sim = embeddings @ embeddings.T  # (batch, batch)

    # Positive mask based on tree rule (example: same repo at level 3)
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            # Example rule: positive if they share the same repo node at level 3
            if paths[i][3] == paths[j][3]:
                pos_mask[i, j] = True

    logits = sim / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # numerical stability

    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1) - torch.exp(logits.diag())     # exclude self

    # numerator: sum over positives
    num = (exp_logits * pos_mask).sum(dim=1) + 1e-8

    loss = -torch.log(num / (denom + 1e-8)).mean()
    return loss
```

You can plug this into your training loop:

```python
out = model(...)
loss = out["loss"]

if use_contrastive:
    contrast_loss = tree_contrastive_loss(out["proj"], batch_paths)
    loss = loss + lambda_contrast * contrast_loss
```

### 6. End-to-End Training Pipeline

Conceptually, the pipeline looks like:

```mermaid
flowchart TD

    A[Raw Repos + Papers] --> B[Graph Construction<br/>RepoGraph + PaperGraph]
    B --> C[Clustering & Taxonomy<br/>Build Tree of Life T]
    C --> D[Path Assignment<br/>(x -> π(x))]

    A --> E[Text Extraction<br/>code, comments, paper sections]
    E --> F[Tokenization<br/>BERT Tokenizer]

    F --> G[TOLBERT Encoder<br/>BERT Backbone]
    G --> H[MLM Head]
    G --> I[Level 1 Head<br/>Domain]
    G --> J[Level 2 Head<br/>Subdomain]
    G --> K[Level 3 Head<br/>Repo/Paper]
    G --> L[Contrastive Projection]

    H --> M[MLM Loss]
    I --> N[Hierarchical Loss L1]
    J --> O[Hierarchical Loss L2]
    K --> P[Hierarchical Loss L3]
    L --> Q[Contrastive Tree Loss]

    M --> R[Joint Objective]
    N --> R
    O --> R
    P --> R
    Q --> R

    R --> S[Backprop & Update]
```

### 7. Phases

#### Phase 0: Data + Taxonomy

- Build/load `RepoGraph` and `PaperGraph`.
- Run clustering and semantics to derive Tree-of-Life levels.
- Assign each text span \(x\) a path \(\pi(x)\).

#### Phase 1: Pretraining

- Initialize from `bert-base`, `roberta-base`, or a code/paper-specialized variant.
- Train on mixed batches (code snippets, docstrings, paper paragraphs) with:
  - MLM.
  - Hierarchical cross-entropy.
  - Optional path-consistency and contrastive losses.

#### Phase 2: Fine-tuning

- For downstream tasks (e.g. bug-fixing, retrieval, concept QA):
  - Attach task-specific heads.
  - Fine-tune end-to-end or freeze most of the backbone.
  - Optionally keep a small weight on hierarchical losses to preserve tree geometry.

For task-specific patterns and how to integrate with your agent stack, see `usage.md`.

### 8. Training Stability & Curriculum

This section gives practical guidance on **when to turn on each loss** and how to set \(\lambda\)s.

- **Recommended curriculum**:
  - **Stage A (warm-up)**:
    - Train with **MLM + shallow \(L_{\text{hier}}\)** only (e.g., levels 1–2).
    - Set \(\lambda_{\text{hier}} = 1.0\), \(\lambda_{\text{contrast}} = 0.0\), \(\lambda_{\text{path}} = 0.0\).
    - Wait until validation MLM loss and shallow-level accuracy stabilize.
  - **Stage B (deeper hierarchy)**:
    - Gradually add deeper levels (e.g., repo/paper level) into \(L_{\text{hier}}\).
    - Keep \(\lambda_{\text{hier}} \approx 1.0\); you can down-weight very noisy levels.
  - **Stage C (contrastive geometry)**:
    - Turn on \(L_{\text{contrast}}\) with a **small weight**: \(\lambda_{\text{contrast}} \in [0.05, 0.1]\) and \(\tau \in [0.03, 0.2]\).
    - Optionally start by defining positives only at a single level (e.g., same repo) before generalizing to tree distance.
  - **Stage D (path consistency)**:
    - Once hierarchical predictions are reasonable, enable \(L_{\text{path}}\) with a small weight: \(\lambda_{\text{path}} \in [0.05, 0.1]\).
    - You can **ramp \(\lambda_{\text{path}}\)** over time to avoid destabilizing early training.

- **Typical \(\lambda\) scales (starting points)**:
  - \(\lambda_{\text{hier}} \approx 1.0\).
  - \(\lambda_{\text{contrast}} \approx 0.05{-}0.1\) (only after the model has learned useful features).
  - \(\lambda_{\text{path}} \approx 0.05{-}0.1\).
  - Adjust based on validation metrics and whether you prioritize classification vs. geometry.

- **Common pitfalls**:
  - **Overweighting contrastive loss**:
    - Symptoms: embeddings over-collapse across branches, level-wise accuracy degrades, or the model ignores fine-grained distinctions.
    - Mitigations: reduce \(\lambda_{\text{contrast}}\), increase \(\tau\), or tighten the positive definition (e.g., same repo only).
  - **Highly imbalanced level distributions**:
    - Symptoms: head performance dominated by a few large branches; rare subtrees are poorly learned.
    - Mitigations:
      - Use **sampling strategies** (e.g., over-sample spans from small branches or under-sample from huge ones).
      - Or apply **per-class / per-branch re-weighting** inside \(L_{\text{hier}}\) so rare branches have non-negligible gradient.

### 9. Performance & scaling considerations

This section gives rough, **practical rules of thumb** for model size, memory, and training setup.

- **Backbone size to start with**:
  - Start with a **BERT-base–scale encoder** (≈110–150M parameters, 12 layers, hidden size 768) such as `bert-base`, `roberta-base`, or a code/paper-specialized variant.
  - Only move to **BERT-large–scale** (≈330M, 24 layers, hidden size 1024) if you:
    - Have multi-GPU or >24–32 GB per device.
    - Enable **mixed precision** (FP16 or bfloat16) and often **gradient checkpointing**.
  - Hierarchical heads add only a small number of parameters relative to the backbone; they should not materially change memory requirements.
- **Batch size vs GPU memory (very rough)**:
  - For **BERT-base, seq length 256**:
    - On a **16 GB** GPU: expect per-device batch sizes in the range **16–32** with mixed precision.
    - On a **24–32 GB** GPU: per-device batch sizes of **32–64** are typical, especially with gradient checkpointing.
  - For **BERT-large**, roughly **halve** these batch sizes for the same hardware.
  - Use **gradient accumulation** to reach an effective global batch size of **128–512** without requiring that many samples to fit on a single device.
  - The hierarchical heads and path loss bookkeeping add some overhead, but **throughput is dominated by the encoder**; if you hit OOM, lower sequence length first, then batch size.
- **Multi-GPU and distributed strategies**:
  - For most setups, **data parallel (DDP)** across GPUs is sufficient: each GPU holds a full copy of the model, you shard batches across devices, and synchronize gradients.
  - For very large models or long-context training, consider **ZeRO / FSDP-style sharding** (e.g., via DeepSpeed or PyTorch FSDP) to partition optimizer states and/or model parameters across GPUs.
  - Keep the **hierarchical heads replicated** across devices; they are small and simple to synchronize.
- **Contrastive loss scaling and approximations**:
  - Tree-aware contrastive loss computes a **similarity matrix of shape (B, B)** per batch; both time and memory are \(O(B^2)\).
  - In practice:
    - Use **in-batch negatives only** (no extra memory bank) as a baseline; with BERT-base and a modest projection dimension, batches of **64–256** examples are usually tractable on modern GPUs.
    - If you need more negatives without increasing per-GPU batch, use a **memory bank / feature queue** or a **cross-batch memory** mechanism that reuses embeddings from recent batches.
    - You can also **subsample negatives** or restrict contrastive pairs to a subset of the batch (e.g., only spans under certain branches) to reduce \(O(B^2)\) cost.
  - If contrastive loss becomes the bottleneck, you can:
    - Lower \(\lambda_{\text{contrast}}\) or run contrastive updates **less frequently** (e.g., every N steps).
    - Keep batch sizes modest and rely on a memory queue to maintain enough negative diversity.

### Evaluation

When you ask “is TOLBERT actually working?”, you want to measure:

- **Classification metrics (per tree level)**:
  - Accuracy, precision/recall, and F1 **per level** (domain, subdomain, repo, file, function, paper, section).
  - Report both **micro** scores (dominated by large branches) and **macro** scores (averaging over branches) to surface performance on small or rare subtrees.
  - Optional: per-branch breakdowns (e.g., accuracy per domain or per repo family) to identify where the hierarchy or labels are weakest.
- **Tree-aware metrics**:
  - **Average tree distance** between predicted and true nodes:
    - E.g., compute distance along the Tree-of-Life between each predicted node and its ground-truth node and average over the eval set.
    - You can compute this at multiple depths (domain-only, subdomain-only, repo-only, etc.).
  - **Depth-wise confusion**:
    - Confusion matrices per level (which domains/subdomains/repos are confused with which others).
    - Optionally, “ancestor/descendant confusion” rates (how often you predict the right ancestor but wrong child, or vice versa).
- **Retrieval metrics**:
  - Standard **Recall@k** and **nDCG@k** for retrieval over:
    - Repos, files, functions, and papers.
    - Within-branch retrieval (restrict candidates to the same branch) vs cross-branch retrieval (no restriction).
  - Evaluate both:
    - **Flat embedding retrieval** (pure similarity over `proj` vectors).
    - **Hierarchy-aware retrieval** (e.g., restrict by predicted branch, or weight scores by tree distance).

In practice, you can run:

- A **classification eval loop** over labeled spans (with known paths) to compute per-level accuracy/F1 and tree distance.
- A **retrieval eval loop** where you:
  - Treat each span as a query, retrieve candidate repos/files/functions/papers from a held-out set by embedding similarity.
  - Mark a retrieval as “correct” if it lands in the same ground-truth branch or subtree, and compute Recall@k / nDCG@k under that criterion.


