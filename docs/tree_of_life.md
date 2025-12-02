## Tree of Life Construction

This document describes how to build the Tree-of-Life (ToL) taxonomy that TOLBERT uses as its hierarchical prior.

### 0. Pre-TOLBERT prerequisites

Before you can build the Tree-of-Life and train TOLBERT, you need **two upstream pieces**.  
**Conceptually, these are domain-agnostic**; in this doc we phrase them in terms of code + papers because that is our reference implementation.

1. **A graph over the artifacts you care about**. In this implementation, that’s **code and papers** (our concrete graphs are `RepoGraph` / `ProgramGraph` and `PaperGraph`), but any equivalent graph over your domain (services, datasets, tickets, etc.) will do.
2. **A way to extract and chunk raw text into spans** that can be mapped back onto the tree. Here we assume spans are drawn from **code, comments, docs, and papers**, but they could equally be from other text-like sources in your universe.

#### 0.1 Upstream graphs (code + papers instantiation)

You do **not** need a perfect knowledge graph. TOLBERT only assumes a **minimal, typed graph** that can later be collapsed into a tree:

- **Code-side graph (e.g., `RepoGraph` / `ProgramGraph`)**
  - **Required node types**:
    - Repositories / projects.
    - Files / modules.
    - Functions / methods (or equivalent semantic units).
  - **Required edge types**:
    - `imports` / `module-depends-on` edges between files/modules.
    - `calls` edges between functions/methods (within and across files).
  - **Nice-to-have edges** (improve clustering / tree quality but are optional):
    - Lexical / embedding-based `similar-to`.
    - “Shares concept with” / shared symbol sets.
    - Ownership / team / service boundaries.
    - Issue / ticket / PR links.

- **Paper-side graph (e.g., `PaperGraph`)**
  - **Required node types**:
    - Papers / documents (one node per logical paper or report).
  - **Required edge types** (any *one* of these is enough to start):
    - `cites` (paper → paper citation edges), or
    - Explicit topical / collection membership (e.g., “in venue X / track Y”), or
    - “is-version-of” / “extends” relationships.
  - **Nice-to-have edges**:
    - Co-authorship or institutional links.
    - Embedding-based similarity (“nearest neighbors”).
    - Shared terminology / keyword overlap.

Internally, we materialize these as `RepoGraph` and `PaperGraph`, but TOLBERT itself only requires that you can **derive a Tree-of-Life** (levels, nodes, and parent/child relations) from *some* upstream graph.

#### 0.2 Text extraction and span chunking

TOLBERT trains on **spans**: short chunks of text (code or natural language). You need a preprocessing pipeline that:

- Extracts raw text from:
  - Code bodies, docstrings, comments.
  - README / markdown / design docs.
  - Papers (PDF/LaTeX/HTML turned into structured text).
- Chunks this text into spans and records, for each span:
  - `span_id`: unique identifier.
  - `text`: the string used for MLM and encoding.
  - `source_id`: where it came from (file path, symbol ID, paper + section).
  - A link (direct or indirect) to the **Tree-of-Life path** that contains it.

Reasonable default chunking strategies:

- **Code**
  - Primary unit: **function/method body**, including its signature and nearby docstring/comments.
  - For very large functions/files: additionally use **sliding windows** over the body (e.g., 256–512 BERT tokens with 128-token overlap).
  - Optionally add spans for:
    - Top-of-file module docstrings / comments.
    - Key class definitions with their methods summarized.

- **Documentation & READMEs**
  - Split on headings and paragraphs.
  - Target span length: ~128–256 tokens (merge short adjacent paragraphs; split very long ones).

- **Papers**
  - Preserve structure: `section → subsection → paragraphs`.
  - Use **paragraph-level spans**, optionally merging very short paragraphs.
  - Maintain stable IDs so each paragraph can be mapped back to its paper/section nodes in the tree.

You do **not** need perfect segmentation; approximate but consistent span extraction is sufficient, as long as each span can be tied back to a path in the Tree-of-Life.

### 1. Inputs

- **Upstream code graph (e.g., `RepoGraph` / `ProgramGraph`)**:
  - **Nodes (required)**: repositories/projects, files/modules, functions/methods.
  - **Edges (required)**: `imports` / module-dependency edges and `calls` edges.
  - **Edges (optional / nice-to-have)**: “similar-to”, “shares concept with”, ownership, issue links, and other semantic relations.
- **Upstream paper graph (e.g., `PaperGraph`)**:
  - **Nodes (required)**: papers / documents; optionally sections and key entities.
  - **Edges (required)**: at least one signal like `cites`, topical grouping, or “is-version-of”.
  - **Edges (optional / nice-to-have)**: co-authorship, similarity, co-occurrence, shared terminology.

You may already be computing embeddings or graph features for these graphs; TOLBERT reuses them to derive a taxonomy.

### 2. Target Levels

Define levels for code and papers (you can share or partially share the tree):

- **Level 0**: Root (All Knowledge / Unified AI + Code Universe).
- **Level 1**:
  - Domains: `AI-Code`, `AI-Papers`, `Systems`, `Infrastructure`, etc.
- **Level 2**:
  - Subdomains / stacks (per domain), e.g. for `AI-Code`:
    - `Transformers`, `Diffusion`, `RL`, `Optimization`, `Mamba-style SSMs`, etc.
- **Level 3**:
  - **Repos**: concrete repositories under each subdomain.
  - **Papers**: concrete papers under each topic.
- **Level 4+**:
  - **Code**: files → modules → functions/classes.
  - **Papers**: sections → subsections → paragraphs / equation clusters.

Exact depth and granularity are adjustable; TOLBERT only requires consistent level indices.

### 3. High-Level Procedure

1. **Top-level domains (Level 1)**:
   - Option A: Use existing labels (e.g. arXiv categories, manual tags).
   - Option B: Run clustering on global graph/node embeddings to derive coarse domains.
2. **Intermediate subdomains (Level 2)**:
   - For each Level-1 cluster, restrict to the induced subgraph and run:
     - Community detection (e.g. Louvain, Leiden).
     - Or embedding-based clustering (e.g. k-means, spectral clustering).
3. **Attach repos/papers (Level 3)**:
   - Each repo or paper node gets assigned to a Level-2 subdomain using:
     - Connectivity (edge weights).
     - Embedding similarity to cluster centroids.
4. **Below repo/paper (Levels 4+)**:
   - **Code**:
     - Use your AST / module structure to generate file → module → function nodes.
   - **Papers**:
     - Parse headings and structure section → subsection → paragraph nodes.

The result is a rooted tree \(T = (V, E, r)\) where:

- Each node \(v \in V\) has:
  - `level(v) ∈ {0, 1, ..., L}`.
  - A parent (except root) and children.
  - A semantic type (domain, subdomain, repo, file, function, section, etc.).

### 4. Assigning Paths to Text Spans

Each training span \(x\) (code snippet, docstring, paper paragraph) is mapped to a **path**:

- \(\pi(x) = (v_0, v_1, ..., v_L)\) with:
  - \(v_0 = r\) (root),
  - \(v_k \in V\) at depth \(k\),
  - \(v_L\) being the leaf (or deepest node) that contains \(x\).

Typical mapping:

- **Code span in a function**:
  - Root → AI Code → Transformers → `transformers` repo → file → function.
- **Paragraph in a paper**:
  - Root → AI Papers → Diffusion → specific paper → section → paragraph.

These paths become supervision targets for TOLBERT’s multi-level heads.

### 5. Data Structures

You will typically maintain:

- **Node table**:
  - `node_id`, `level`, `type`, `parent_id`, `metadata` (e.g. repo name, file path).
- **Edge table**:
  - `(parent_id, child_id)` for the tree edges.
- **Span-to-path mapping**:
  - For each textual span ID:
    - `span_id`, `text_source_id` (file/section), `path = [node_id_0, ..., node_id_L]`.

These structures can be materialized in a database, parquet files, or simple JSONL/CSV, depending on scale.

### 6. Tree-of-Life Diagram (Conceptual)

```mermaid
graph TD

    Root[Root: Unified Knowledge]

    subgraph Level1[Level 1: Domains]
        A1[AI Code]
        A2[AI Papers]
        A3[Systems]
    end

    Root --> A1
    Root --> A2
    Root --> A3

    subgraph Level2_AI_Code[Level 2: AI Code Subdomains]
        B1[Transformers]
        B2[Diffusion]
        B3[RL]
    end

    A1 --> B1
    A1 --> B2
    A1 --> B3

    subgraph Level3_Transformers[Level 3: Repos]
        C1[HuggingFace/transformers]
        C2[Your LLaMA Repo]
        C3[Your AGI Substrate]
    end

    B1 --> C1
    B1 --> C2
    B1 --> C3

    C1 --> D1[File: modeling_llama.py]
    C1 --> D2[File: trainer.py]

    D1 --> E1[Function: forward()]
    D1 --> E2[Function: prepare_inputs()]
```

### 7. Data & Config Details (Engineer Cheat Sheet)

This section summarizes **what you need on disk and which knobs to set** before you can train TOLBERT. It is intentionally implementation-oriented and complements `training.md`.

#### 7.1 Data files

- **Nodes file** (`nodes.parquet` or `nodes.jsonl`):
  - **Required fields**: `node_id`, `level`, `type`, `parent_id`, `name`, `attributes`.
  - **Semantics**:
    - `node_id`: stable integer/string ID.
    - `level`: depth in the tree (0 = root).
    - `type`: domain/subdomain/repo/file/function/section/etc.
    - `parent_id`: `node_id` of the parent (null or special for root).
    - `name`: human-readable label (repo name, file path, section title, ...).
    - `attributes`: free-form JSON/dict for any extra metadata.
  - **Minimal JSONL example**:

```json
{"node_id": 0, "level": 0, "type": "root", "parent_id": null, "name": "Root", "attributes": {}}
{"node_id": 1, "level": 1, "type": "domain", "parent_id": 0, "name": "AI Code", "attributes": {}}
{"node_id": 2, "level": 2, "type": "subdomain", "parent_id": 1, "name": "Transformers", "attributes": {}}
```

- **Edges file** (`edges.parquet` or `edges.jsonl`):
  - **Required fields**: `parent_id`, `child_id`.
  - Should form a rooted tree (or forest that you later root).
  - **Minimal JSONL example**:

```json
{"parent_id": 0, "child_id": 1}
{"parent_id": 1, "child_id": 2}
```

- **Spans file** (`spans.parquet` or `spans.jsonl`):
  - **Required fields** (minimal):
    - `span_id`: unique ID per training span (code snippet, paragraph, etc.).
    - `text`: raw text for MLM and encoding.
    - `source_id`: identifier of the underlying file/section/paper.
    - Either:
      - `node_path`: full list of node IDs `[v_0, ..., v_K]`, or
      - `level_k_id` columns: `level_0_id`, `level_1_id`, ..., `level_K_id`.
  - **Minimal JSONL example (with `node_path`)**:

```json
{"span_id": "s0", "text": "def forward(...):", "source_id": "repoA/file.py", "node_path": [0, 1, 2]}
{"span_id": "s1", "text": "We propose TOLBERT...", "source_id": "paperX/section1", "node_path": [0, 3, 5]}
```

You can store these as Parquet for scale, as long as the column names and semantics match the above.

#### 7.2 Core configuration knobs

- **Number of supervised levels `K`**:
  - How many depths from the root you supervise with hierarchical heads.
  - Typical: `K = 3` (e.g., domain → subdomain → repo/paper), but you can go deeper if you have reliable labels.

- **Loss weights** (see `training.md` for full formulas):
  - \(\lambda_{\text{hier}}\): weight on hierarchical cross-entropy \(L_{\text{hier}}\).
  - \(\lambda_{\text{path}}\): weight on path-consistency loss \(L_{\text{path}}\).
  - \(\lambda_{\text{contrast}}\): weight on contrastive tree loss \(L_{\text{contrast}}\).
  - **Guideline**: start with \(\lambda_{\text{hier}} = 1.0\), \(\lambda_{\text{path}} \in [0.1, 1.0]\), \(\lambda_{\text{contrast}} \in [0.0, 0.5]\) and tune per dataset.

- **Contrastive temperature \(\tau\)**:
  - Controls sharpness of the contrastive distribution (see InfoNCE definition in `training.md`).
  - Typical values: \(\tau \in [0.03, 0.2]\); lower = harder negatives, higher = smoother gradients.

- **Batch size / sampling**:
  - Larger batches help contrastive learning (more negatives), but are not required for the hierarchical losses.
  - Aim for at least **512 spans per batch** if you rely heavily on contrastive terms; **128–256** can work if memory is tight.
  - Ensure each batch has a reasonable mix of branches/repos so positives/negatives are meaningful.

For end-to-end training and integration, see `training.md` and `usage.md`.

### 8. Non-ideal Graphs: DAGs and Partial Labels

Real-world data often violates the assumption of a **perfect rooted tree**:

- **DAG-like structure (multi-parent nodes)**:
  - Your upstream `RepoGraph` / `PaperGraph` may give a node multiple parents (e.g., a repo that belongs to both “Transformers” and “RL”).
  - For TOLBERT, you can:
    - **Canonicalize to a tree** at taxonomy-construction time:
      - Choose a single parent using a deterministic heuristic (e.g., highest edge weight, most attached spans, or a fixed domain priority).
      - Optionally store the full parent set in `attributes` if you want to use it later in training-time losses.
    - Or treat **multiple parents as valid ancestors** during training:
      - Encode all plausible parents in `attributes` or separate columns.
      - In `training.md`, the \(L_{\text{path}}\) section describes how to allow multiple valid ancestors when deciding which nodes are “invalid”.

- **Partially-labeled paths**:
  - Many spans will only have labels up to a shallow depth (e.g., domain/subdomain) with deeper levels unknown.
  - You can:
    - Store only the known prefix in `node_path` and omit deeper nodes, or
    - Use a sentinel (e.g., `null` or `-1`) for unknown levels.
  - The training logic (see `training.md`) then:
    - **Ignores unknown levels** in hierarchical cross-entropy.
    - Skips path-consistency terms that involve any unknown level.

