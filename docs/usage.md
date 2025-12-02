## Using TOLBERT in Your Stack

This document explains how to integrate TOLBERT into your workflows once the model and Tree-of-Life have been trained.

### 1. Typical Use Cases

- **Hierarchical retrieval**:
  - Retrieve repos, files, functions, or papers by:
    - Tree distance between their nodes.
    - Embedding similarity at specific levels.
- **Agent planning and grounding**:
  - Give agents access to a structured universe:
    - “Find a repo under Transformers related to this function.”
    - “Locate relevant papers under Diffusion that match this code change.”
- **Cross-architecture reasoning**:
  - Map concepts across architectures via shared branches:
    - E.g. Transformers → LLaMA → SSM-style variants.
- **Hierarchical classification**:
  - Predict domain/subdomain/repo for new snippets or paragraphs.

### 2. Inference API (Conceptual)

Assuming a TOLBERT model similar to the `TOLBERT` class from `architecture.md`:

```python
from transformers import AutoTokenizer
import torch

from tolbert.modeling import TOLBERT  # planned location; subject to change


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TOLBERT.from_pretrained("your-org/tolbert-base")  # or load_state_dict(...)
model.eval()


def encode_span(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    # cls embedding and contrastive projection
    cls = out["proj"]  # or outputs from encoder directly
    return cls.squeeze(0)  # (dim,)
```

You can then use `encode_span` to:

- Store embeddings in a vector database.
- Compute similarity scores for retrieval.
- Drive routing and selection in agent policies.

### 3. Hierarchical Predictions

At inference time, in addition to embeddings, you can use level logits:

```python
with torch.no_grad():
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

level_logits = out["level_logits"]  # dict: {str(level): (batch, C_level)}

level_predictions = {
    int(level): logits.argmax(dim=-1).item()
    for level, logits in level_logits.items()
}
```

- Map predicted node IDs to nodes in your Tree-of-Life:
  - Domain, subdomain, repo, file, function, etc.
- Use this to:
  - Route queries to relevant parts of your graph.
  - Debug and inspect model behavior (“why did it think this span belongs to RL?”).

### 4. Multi-Resolution Retrieval

You can perform retrieval at different resolutions:

- **Global similarity**:
  - Use embeddings directly from `proj` head.
- **Branch-restricted retrieval**:
  - Filter candidates by predicted domain/subdomain before similarity search.
- **Leaf-level retrieval**:
  - Combine:
    - Predicted repo/file/function.
    - Local similarity within that leaf or branch.

Conceptually:

1. Predict path \(\hat{\pi}(x)\) for a query span.
2. Use the predicted level-2 or level-3 nodes to **restrict** candidates.
3. Compute embedding similarity within that restricted set.

### 5. Integration with Repo/Paper Graphs

Because each span is mapped into both:

- A **Tree-of-Life path** (discrete).
- A **tree-aware embedding** (continuous).

You can:

- Jump between:
  - Graph nodes (repos, papers, functions).
  - Text spans (snippets, paragraphs).
  - Embeddings (vectors in your retrieval store).
- Example flows:
  - “Given this function, find relevant papers”:
    1. Get function node → path in Tree-of-Life.
    2. Use path to locate nearby paper branches.
    3. Retrieve by embedding similarity within those branches.
  - “Given this paragraph, find similar code”:
    1. Encode paragraph → path + embedding.
    2. Restrict to compatible branches (e.g. same domain, subdomain).
    3. Search over span embeddings from code under those branches.

### 6. Fine-Tuning for Downstream Tasks

For each downstream task, attach a head on top of the encoder (CLS or pooled representation):

- **Classification / tagging**:
  - Add a linear layer on top of CLS.
  - Optionally combine with hierarchical predictions (e.g. as features).
- **Span or span-pair scoring**:
  - Use similarity between `proj` outputs for ranking.
- **Generation-augmented tasks**:
  - Use TOLBERT only as retriever / context selector, feeding retrieved contexts into a generative model.

During fine-tuning:

- Decide whether to:
  - Freeze the backbone and only train task heads.
  - Or fine-tune end-to-end, possibly with small \(\lambda_{\text{hier}}\) to preserve tree geometry.

### 7. Monitoring & Diagnostics

- Track:
  - **Accuracy per tree level**: domain, subdomain, repo ID, etc.
  - **Tree distance errors**: how far predicted paths deviate from ground truth.
  - **Branch confusion matrices**: which branches often get mistaken for one another.
- Use these metrics to:
  - Refine Tree-of-Life construction (better clustering).
  - Adjust training weights (e.g. emphasize deeper levels if needed).

For concrete metric definitions and suggested evaluation loops, see the **Evaluation** section in `training.md`.

### 8. End-to-End Example: Function → Branch-Restricted Retrieval

This is a high-level example of how you might wire TOLBERT into a retrieval stack for code:

```python
from typing import List

import torch
from transformers import AutoTokenizer

from tolbert.modeling import TOLBERT  # planned location; subject to change
from my_vector_db import search_in_branch  # hypothetical helper
from my_tree_index import level3_to_branch  # maps level-3 node id -> branch / collection id


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TOLBERT.from_pretrained("your-org/tolbert-base").eval()


def encode_and_route(function_src: str, top_k: int = 20) -> List[str]:
    # 1) Encode the function into TOLBERT space
    inputs = tokenizer(
        function_src,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # 2) Get hierarchical predictions (e.g., domain → subdomain → repo)
    level_logits = out["level_logits"]          # dict: {str(level): (batch, C_level)}
    level_preds = {
        int(level): logits.argmax(dim=-1).item()
        for level, logits in level_logits.items()
    }
    repo_id = level_preds.get(3)                # assuming level 3 = repo / paper

    # 3) Get a contrastive embedding for retrieval
    query_emb = out["proj"].squeeze(0)          # (dim,)

    # 4) Map repo_id to a branch/collection in your vector DB and search
    branch_id = level3_to_branch(repo_id)
    results = search_in_branch(branch_id, query_emb, top_k=top_k)

    return results  # e.g. list of (doc_id, score) or similar
```

Conceptually:

1. **Encode** a raw function string with TOLBERT.
2. **Predict** its path nodes (domain, subdomain, repo) from `level_logits`.
3. **Restrict** retrieval to the predicted branch/collection.
4. **Search** within that branch using the contrastive embedding from the `proj` head.

For a conceptual overview and connection to your AGI substrate, see `overview.md` and `README.md`.

### 9. Worked Scenario: Paper Recommendations for a Pull Request

This scenario shows how an agent might use TOLBERT and an LLM to recommend relevant research papers for a code change (e.g., a PR that adds a new model or algorithm).

1. **Ingest the PR as spans**  
   - Extract diff hunks, new/changed functions, and key comments from the PR.  
   - Treat each significant chunk (e.g., a new `forward` method or a config block) as a span of text.

2. **Encode spans with TOLBERT**  
   - Tokenize each span and run it through the pretrained TOLBERT model.  
   - For each span, capture:
     - The **contrastive embedding** (`proj`) for retrieval.  
     - The **hierarchical predictions** (`level_logits` → predicted domain, subdomain, repo, etc.).

3. **Locate the PR’s position in the Tree-of-Life**  
   - Aggregate span-level predictions to infer:
     - Dominant **domain** (e.g., AI-Code).  
     - Dominant **subdomain** (e.g., Transformers, Diffusion).  
   - Optionally, map changed files/functions to existing tree nodes to see which repo subtree is affected.

4. **Branch-restricted retrieval over papers**  
   - Use the inferred domain/subdomain to restrict candidate papers to the **matching branches** in the Tree-of-Life (e.g., AI-Papers → Transformers).  
   - For each span embedding from the PR, retrieve top-k paper sections/paragraphs from a vector DB built over paper spans, limited to those branches.

5. **Aggregate and rank candidate papers**  
   - Combine retrieval results across all PR spans:
     - Score papers by frequency and similarity of matches across the PR.  
     - Optionally weight by tree distance (closer branches score higher).

6. **Summarize and explain with an LLM**  
   - Provide the LLM with:
     - The PR summary or key diffs.  
     - The top-N retrieved paper abstracts/sections (or short summaries).  
   - Ask it to:
     - Explain why each paper is relevant to the PR.  
     - Highlight specific methods or design decisions from the papers that relate to the code change.

7. **Present recommendations to the user**  
   - The agent surfaces:
     - A ranked list of papers with short, LLM-generated rationales.  
     - Optional follow-up prompts (e.g., “show me how this PR’s architecture differs from Paper X”).

This same pattern generalizes to other tasks (bug-fix suggestion, design review, refactor planning) by swapping “papers” for “repos/files/functions” and adjusting what the LLM is asked to do with the retrieved context.
