## Tree-of-Life BERT (TOLBERT) Overview

TOLBERT (Tree-of-Life BERT) is a BERT-style encoder whose embeddings are constrained and trained to respect a hierarchical **Tree-of-Life** taxonomy built over your repositories, papers, and internal graphs.

- **Goal**: Move from a flat embedding space to a **multi-resolution semantic space** where distances reflect paths in a tree of concepts (root → domain → subdomain → repo/paper → file → function/section).
- **Inputs**: Text spans from code, comments, and papers, plus a Tree-of-Life taxonomy built from any upstream graphs over your code and papers (our internal `RepoGraph` / `ProgramGraph` and `PaperGraph` are **example implementations**, not hard requirements).
- **Outputs**:
  - Standard BERT-style contextual embeddings.
  - Hierarchy-aware predictions at multiple levels of a Tree-of-Life.
  - Optional contrastive embeddings aligned with the tree geometry.

TOLBERT is intended to be a **drop-in encoder** for downstream tasks (retrieval, QA, reasoning, bug-fixing) while injecting a strong **hierarchical prior** over your entire knowledge universe.

### Key Concepts

- **Tree-of-Life (ToL)**: A rooted tree over your universe of knowledge:
  - Level 0: root (unified knowledge).
  - Level 1: coarse domains (e.g., AI, Systems, Tooling, etc.).
  - Level 2: subdomains / stacks (e.g., Transformers, Diffusion, RL, Linux kernel).
  - Level 3: concrete artifacts (repos, individual papers).
  - Level 4+: internal structure (files, functions, sections, equation clusters).
- **Path for each span**: Every training text span \(x\) is associated with a path
  \(\pi(x) = (v_0, v_1, ..., v_L)\) from root to a leaf (or intermediate node).
- **Multi-level heads**: TOLBERT attaches classification (and optional embedding) heads at multiple tree depths to predict the node at each level.
- **Tree-aware losses**: Training includes standard MLM plus hierarchical and contrastive objectives that encourage geometric alignment with the tree.

### Glossary

- **Tree-of-Life**: The global rooted tree over your repos and papers that organizes knowledge into domains, subdomains, and concrete artifacts.
- **level**: The depth index of a node in the Tree-of-Life, with 0 at the root and larger values for more specific concepts.
- **node**: A single vertex in the Tree-of-Life (domain, subdomain, repo, file, function, paper, section, etc.).
- **path \(\pi(x)\)**: The ordered list of nodes from the root to the node associated with a span \(x\); written \(\pi(x) = (v_0, \dots, v_L)\).
- **span**: A contiguous piece of text used as a training/example unit (e.g., a sentence, paragraph, code snippet, or docstring).
- **branch**: A subtree rooted at some internal node, representing a domain or subdomain and all of its descendants.
- **leaf**: A node with no children, typically a very fine-grained artifact such as a particular function, class, or paragraph.

See `architecture.md` for the detailed model, and `tree_of_life.md` for how to construct the tree from your graphs (regardless of whether you use our `RepoGraph` / `PaperGraph` implementations or your own equivalents).

### Limitations & assumptions

TOLBERT is powerful when the taxonomy is well-structured and reasonably stable, but there are important caveats:

- **Bounded by taxonomy quality**:
  - If your Tree-of-Life is noisy, overly coarse, or misaligned with how the code/papers are actually used, TOLBERT will inherit those flaws.
  - In those cases you may see limited benefit over standard BERT beyond what the hierarchy already encodes.
- **Sensitivity to highly dynamic or adversarial codebases**:
  - If repos, files, and responsibilities change very frequently (or are deliberately obfuscated), the tree structure can drift quickly.
  - You will need more frequent **tree rebuilding, versioning, and reindexing**, and some branches may never stabilize enough for deep hierarchy to help.
- **Not always better than flat models for all tasks**:
  - For very local, token-level tasks (e.g., short-range syntax completion or micro-level code edits), a flat encoder may perform similarly or better with fewer constraints.
  - TOLBERT is most valuable when tasks **benefit from the hierarchy itself**: cross-repo retrieval, reasoning over branches, or multi-level classification, rather than purely local token prediction.

