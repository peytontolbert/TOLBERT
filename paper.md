\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{mathtools}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\title{TOLBERT: Tree-of-Life BERT for Hierarchical Semantic Representation}
\author{Anonymous Authors}
\date{}

\begin{document}

\maketitle

\begin{abstract}
Modern Transformer encoders such as BERT provide powerful bidirectional contextual representations but treat the semantic space as essentially flat: tokens and sequences are embedded into a continuous manifold without any explicit hierarchical structure reflecting how knowledge is organized in real-world domains. In large software and research ecosystems, however, artifacts are naturally arranged in trees: domains, subdomains, repositories, modules, functions; fields, subfields, papers, sections, equations. This paper introduces \emph{TOLBERT} (Tree-of-Life BERT), a BERT-style encoder whose representations are constrained to respect a Tree-of-Life (ToL) taxonomy built over a corpus. TOLBERT augments standard Masked Language Modeling (MLM) pretraining with multi-level hierarchical classification, path-consistency regularization, and tree-aware contrastive objectives. The result is a model that not only embeds text but also infers where in a hierarchical knowledge universe the text belongs, aligning geometric distances with distances in a conceptual tree. We describe how to construct the Tree of Life from domain graphs (e.g., repository graphs, citation graphs), define the TOLBERT architecture and training objective, and outline evaluation tasks where TOLBERT strictly dominates flat BERT: hierarchical classification, branch-consistent retrieval, and multi-resolution navigation over large code--paper universes.
\end{abstract}

\section{Introduction}

BERT and its descendants have become the default encoder backbone for a wide range of natural language understanding tasks. Their success is grounded in two core design decisions: (i) the Transformer encoder with multi-head self-attention, and (ii) self-supervised pretraining via Masked Language Modeling (MLM) and related objectives. However, BERT implicitly assumes that the semantic space is unstructured beyond distributional statistics: token co-occurrence and positional structure define the geometry, but there is no explicit representation of hierarchy---no notion that some concepts are ancestors of others, or that artifacts live along structured ``lineages'' of domains and subdomains.

Meanwhile, real-world engineering and scientific corpora are deeply hierarchical. A large software ecosystem decomposes into domains (AI, systems, databases), subdomains (Transformers, RL, S4, compilers), repositories, modules, and functions. Research corpora decompose into fields, subfields, canonical works, and internal structure such as sections and equations. Practitioners routinely reason in this hierarchical space: \emph{``this function lives in this module, in this repo, in this subdomain of Transformers''}. Yet standard representation learning systems typically ignore this prior and train purely flat encoders.

We propose \emph{TOLBERT}, an extension of BERT that uses a Tree-of-Life (ToL) taxonomy, derived from a corpus-level graph, as an explicit inductive bias. Each training span is not just an unlabeled sequence but an instance located along a path in a rooted tree: from the root (universal knowledge) through domains and subdomains down to concrete artifacts (repositories, papers, functions, sections). TOLBERT is trained to jointly: (1) recover masked tokens, (2) predict the correct node at multiple depths of the tree, and (3) maintain geometric consistency with the tree through contrastive and path-consistency losses. This transforms BERT from a flat semantic engine into a hierarchically grounded representation model.

At a high level, TOLBERT maintains the strengths of BERT---bidirectional context, self-supervision, compatibility with existing toolchains---while adding a multi-resolution semantic prior that aligns model geometry with domain structure. In large-scale code--paper universes, this yields a representation space aligned with how experts think: navigation by domain, subdomain, and artifact, instead of raw token similarity alone.

\section{Tree-of-Life: A Hierarchical Knowledge Prior}

We formalize the Tree of Life as a rooted tree defined over the corpus.

\subsection{Tree Definition}

Let the Tree of Life be a rooted tree
\begin{equation}
    T = (V, E, r),
\end{equation}
where $V$ is the set of nodes, $E$ the set of directed edges, and $r \in V$ the root. Each node $v \in V$ has:
\begin{itemize}[leftmargin=1.5em]
    \item A depth level $\text{level}(v) \in \{0, 1, \dots, L\}$, with $\text{level}(r) = 0$.
    \item A parent $\text{parent}(v)$ for all $v \neq r$.
    \item A semantic type, such as domain, subdomain, repository, paper, file, function, or section.
\end{itemize}

Each training instance---e.g., a code snippet, function body, or paper paragraph---is associated with a path in the tree:
\begin{equation}
    \pi(x) = (v_0, v_1, \dots, v_K),
\end{equation}
where $v_0 = r$, $v_k \in V$, and $\text{level}(v_k) = k$. This path encodes the semantic lineage of the instance $x$: from the most abstract node (universal knowledge) to a concrete leaf (specific function, section, or local concept cluster).

\subsection{Constructing the Tree from Graphs}

In practice, knowledge is not given as a perfect tree but as a graph. For example:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Repository graph:} nodes are repositories, files, and functions; edges include imports, calls, structural relations, and similarity edges.
    \item \textbf{Paper graph:} nodes are papers, sections, and extracted concept nodes; edges include citations, co-authorship, topical similarity, and shared entities.
\end{itemize}

From these graphs we derive the Tree of Life by aggregating and clustering:

\paragraph{Root and macro-domains.}
The root represents the entire knowledge universe. Large-scale clustering, metadata labels (e.g., top-level subject areas), or manual domain definitions create level-1 nodes (domains: AI, systems, theory, etc.).

\paragraph{Subdomains.}
Within each domain, we partition the induced subgraph into subdomains (e.g., Transformers, Diffusion, RL, compilers, databases) via community detection or clustering over node embeddings.

\paragraph{Artifact nodes.}
Repositories and papers form level-3 nodes, attached to the subdomain that maximizes connectivity and similarity. Deeper levels mirror intrinsic structure: modules, files, functions for code; sections, subsections, paragraphs for papers.

The result is a tree whose structure approximates how a human would hierarchically map the corpus, but constructed semi-automatically from graphs and metadata.

\begin{figure}[t]
    \centering
    \begin{tikzpicture}[
        level 1/.style={sibling distance=28mm},
        level 2/.style={sibling distance=20mm},
        level 3/.style={sibling distance=14mm},
        every node/.style={font=\small}
    ]
        \node[draw, rounded corners, inner sep=4pt]{Root: Knowledge Universe}
            child { node[draw, rounded corners]{AI Code}
                child { node[draw, rounded corners]{Transformers}
                    child { node[draw, rounded corners]{Repo A} }
                    child { node[draw, rounded corners]{Repo B} }
                }
                child { node[draw, rounded corners]{RL} }
            }
            child { node[draw, rounded corners]{AI Papers}
                child { node[draw, rounded corners]{Theory}
                    child { node[draw, rounded corners]{Paper X} }
                }
            }
            child { node[draw, rounded corners]{Systems} };
    \end{tikzpicture}
    \caption{Illustrative Tree-of-Life (ToL) over a combined code--paper universe.}
    \label{fig:tol_tree}
\end{figure}

\section{TOLBERT Architecture}

TOLBERT is a BERT-style encoder with multi-level tree heads and an optional contrastive projection head. The base encoder remains unchanged: a standard Transformer encoder stack is responsible for capturing fine-grained bidirectional context.

\subsection{Base Encoder}

Let the base encoder be any BERT-style model with $L$ layers and hidden dimension $d$. For an input sequence of length $T$ (including a \texttt{[CLS]} token), tokenized into $x = (x_1, \dots, x_T)$, the encoder outputs hidden states:
\begin{equation}
    H = \text{Encoder}(x) \in \mathbb{R}^{T \times d}.
\end{equation}
We use the \texttt{[CLS]} representation as the sequence-level embedding:
\begin{equation}
    h = H_1 \in \mathbb{R}^{d}.
\end{equation}

\subsection{Multi-Level Hierarchical Heads}

For each depth level $k \in \{1, \dots, K\}$ that we explicitly model, suppose there are $C_k$ nodes at that level. We define a linear head:
\begin{equation}
    z_k = W_k h + b_k \in \mathbb{R}^{C_k},
\end{equation}
with $W_k \in \mathbb{R}^{C_k \times d}$. The predicted distribution over nodes at level $k$ is
\begin{equation}
    p_k = \text{softmax}(z_k).
\end{equation}
Intuitively, the model is asked: given the contextualized sequence, which domain (level 1), which subdomain (level 2), which repository or paper (level 3), and possibly which file/function/section (deeper levels) does this instance belong to?

\subsection{Contrastive Projection Head}

To encourage geometric alignment with the tree, we define a projection into a lower-dimensional embedding space:
\begin{equation}
    e(x) = \text{normalize}(P h) \in \mathbb{R}^{d_h},
\end{equation}
with $P \in \mathbb{R}^{d_h \times d}$ and L2 normalization. This embedding is used for tree-aware contrastive learning, making adjacent nodes in the tree geometrically close and distant branches separated.

\subsection{MLM Head}

TOLBERT retains a standard MLM head over token-level hidden states:
\begin{equation}
    \hat{y}_t = \text{softmax}(W_{\text{MLM}} H_t),
\end{equation}
where masked tokens are predicted using the standard BERT masking scheme. This ensures TOLBERT remains a strong language model and can be initialized from any existing BERT/RoBERTa checkpoint.

\section{Training Objective}

TOLBERT’s training objective combines:
\begin{enumerate}[leftmargin=1.7em]
    \item Masked Language Modeling (MLM),
    \item Multi-level hierarchical cross-entropy,
    \item Path-consistency regularization,
    \item Tree-aware contrastive loss.
\end{enumerate}

Let a mini-batch be $\{(x_i, \pi(x_i))\}_{i=1}^B$.

\subsection{Masked Language Modeling}

We use the standard MLM objective. For a set of masked positions $M_i$ in instance $x_i$,
\begin{equation}
    \mathcal{L}_{\text{MLM}}
    =
    - \sum_{i=1}^{B} \sum_{t \in M_i}
    \log p_{\theta}(x_{i,t} \mid x_i \setminus t),
\end{equation}
where $x_i \setminus t$ denotes the sequence with the token at position $t$ masked.

\subsection{Hierarchical Cross-Entropy}

For each instance $x_i$, its path is
\begin{equation}
    \pi(x_i) = (v_{i,0}, v_{i,1}, \dots, v_{i,K}),
\end{equation}
with $v_{i,k}$ the ground-truth node at level $k$. Let $y_{i,k}$ be the index of node $v_{i,k}$ among level-$k$ nodes. The hierarchical classification loss is:
\begin{equation}
    \mathcal{L}_{\text{hier}}
    =
    - \sum_{i=1}^{B} \sum_{k=1}^{K}
    \log p_k^{(i)} (y_{i,k}),
\end{equation}
where $p_k^{(i)}$ is the predicted distribution at level $k$ for instance $i$. This encourages the \texttt{[CLS]} representation to encode enough information to infer the correct branch path through the tree at multiple resolutions.

\subsection{Path-Consistency Regularization}

Although the hierarchical cross-entropy encourages correct predictions per level, it does not enforce that predictions across levels form a valid path. Path-consistency regularization penalizes probability mass assigned to nodes whose ancestors are inconsistent.

Let $I_{i,k+1}^{\text{invalid}}$ be the indices of level-$(k+1)$ nodes whose level-$k$ ancestor does \emph{not} match $v_{i,k}$. We define
\begin{equation}
    \mathcal{L}_{\text{path}}
    =
    \sum_{i=1}^{B} \sum_{k=1}^{K-1}
    \sum_{j \in I_{i,k+1}^{\text{invalid}}}
    p_{k+1}^{(i)}(j).
\end{equation}
This pushes the model to place deeper probability mass under ancestors that are themselves probable, aligning the distributional structure with tree constraints.

\subsection{Tree-Aware Contrastive Loss}

To align geometry with the tree, we define a contrastive loss over embeddings $e(x_i)$. For each pair $(i,j)$ in the batch, we define a tree distance $d_T(i,j)$, e.g., the depth of the Lowest Common Ancestor (LCA) or path length. We choose positives as pairs whose tree distance is below a threshold (e.g., same repository or same subdomain), and negatives otherwise.

Let $P(i)$ be the set of positive indices for $i$. A simple InfoNCE-style loss is:
\begin{equation}
    \mathcal{L}_{\text{tree}}
    =
    -\frac{1}{B} \sum_{i=1}^{B}
    \log
    \frac{
        \sum_{j \in P(i)} \exp\big( \text{sim}(e(x_i), e(x_j)) / \tau \big)
    }{
        \sum_{\substack{m=1 \\ m \neq i}}^{B}
        \exp\big( \text{sim}(e(x_i), e(x_m)) / \tau \big)
    },
\end{equation}
where $\text{sim}$ is cosine similarity and $\tau$ is a temperature. This encourages instances in the same branch to be close and across-branch instances to be separated, aligning latent geometry with tree structure instead of raw token-level similarity alone.

\subsection{Joint Objective}

The final TOLBERT loss is:
\begin{equation}
    \mathcal{L}
    =
    \mathcal{L}_{\text{MLM}}
    + \lambda_{\text{hier}} \mathcal{L}_{\text{hier}}
    + \lambda_{\text{path}} \mathcal{L}_{\text{path}}
    + \lambda_{\text{tree}} \mathcal{L}_{\text{tree}},
\end{equation}
where hyperparameters $\lambda_{\text{hier}}, \lambda_{\text{path}}, \lambda_{\text{tree}}$ control the relative weight of hierarchical effects. In practice, one can initialize from a pretrained BERT, start with $\mathcal{L}_{\text{MLM}}$ and gradually introduce tree losses to avoid catastrophic forgetting.

\section{System Design and Training Pipeline}

The TOLBERT system is best understood as a pipeline: \emph{graph} $\rightarrow$ \emph{tree} $\rightarrow$ \emph{path-labeled text} $\rightarrow$ \emph{TOLBERT training}.

\begin{figure}[t]
    \centering
    \begin{tikzpicture}[
        node distance=8mm and 14mm,
        box/.style={draw, rounded corners, align=center, inner sep=4pt},
        >=Stealth
    ]
        \node[box] (A) {Raw Corpus\\Repos, Papers, Docs};
        \node[box, right=of A] (B) {Graph Construction\\RepoGraph, PaperGraph};
        \node[box, right=of B] (C) {Tree-of-Life Construction\\Clustering, Domains, Subdomains};
        \node[box, below=of B] (E) {Text Extraction\\Code, Comments, Paragraphs};
        \node[box, right=of E] (F) {Tokenization\\BERT-style};

        \node[box, below=of C, xshift=10mm] (G) {TOLBERT Encoder\\Transformer + Heads};
        \node[box, right=of G, xshift=12mm] (H) {MLM Head};
        \node[box, below=of G, xshift=-10mm] (I) {Hierarchical Heads\\Level 1..K};
        \node[box, below=of H] (J) {Contrastive Projection};

        \node[box, right=of H, xshift=12mm] (K) {MLM Loss};
        \node[box, below=of K] (L) {Hierarchical \& Path Losses};
        \node[box, below=of L] (M) {Tree Contrastive Loss};
        \node[box, right=of L, xshift=12mm] (N) {Joint Objective};

        \draw[->] (A) -- (B);
        \draw[->] (B) -- (C);
        \draw[->] (A) |- (E);
        \draw[->] (C) |- ++(0,-0.4) -| (G);
        \draw[->] (E) -- (F);
        \draw[->] (F) -- (G);

        \draw[->] (G) -- (H);
        \draw[->] (G) -- (I);
        \draw[->] (G) -- (J);

        \draw[->] (H) -- (K);
        \draw[->] (I) -| (L);
        \draw[->] (J) -- (M);

        \draw[->] (K) -- (N);
        \draw[->] (L) -- (N);
        \draw[->] (M) -- (N);
    \end{tikzpicture}
    \caption{System-level view of TOLBERT: graph and metadata are converted into a Tree of Life, which labels each span with a path $\pi(x)$; TOLBERT is then trained with MLM, hierarchical, path-consistency, and tree-aware contrastive objectives.}
    \label{fig:pipeline}
\end{figure}

This pipeline is fully compatible with existing BERT infrastructure, with the only new requirements being:
\begin{itemize}[leftmargin=1.5em]
    \item A Tree-of-Life construction module taking the graph + metadata to build levels and parent relations.
    \item A path labeling module that maps each extracted text span to its tree path.
    \item New heads and losses implemented on top of a standard encoder.
\end{itemize}

\section{Advantages Over Flat BERT}

TOLBERT retains BERT’s strengths but strictly improves its inductive bias in hierarchical domains.

\subsection{Multi-Resolution Semantics}

Flat BERT encodes each sequence as a vector without explicit resolution; any notion of a ``domain'' or ``repository'' must be inferred post hoc. TOLBERT instead jointly predicts:
\begin{itemize}[leftmargin=1.5em]
    \item Which high-level domain the instance belongs to,
    \item Which subdomain,
    \item Which specific artifact (repository, paper),
    \item Optionally which internal unit (file, section).
\end{itemize}
This makes each embedding inherently multi-resolution: from coarse (domain) to fine (function), the model encodes where the instance lives in the hierarchy.

\subsection{Tree-Aligned Geometry}

In flat BERT, proximity in latent space is a byproduct of MLM training; there is no guarantee that it matches domain-level organization. TOLBERT’s tree contrastive loss and hierarchical regularization ensure that geometry reflects the Tree of Life:
\begin{itemize}[leftmargin=1.5em]
    \item Sibling artifacts in the tree are close.
    \item Distant branches are separated.
    \item Global structure emerges as a tree-structured manifold rather than an amorphous cloud.
\end{itemize}
This makes TOLBERT particularly suitable for retrieval, navigation, and clustering in large, heterogeneous knowledge universes.

\subsection{Sample Efficiency and Transfer}

Hierarchical supervision can provide denser and more structured signals than flat labels. Predicting domains and subdomains can be easier than predicting specific fine-grained labels, and these higher-level signals can guide representation learning. TOLBERT leverages this by sharing weights across levels: the same \texttt{[CLS]} representation supports predictions at all depths, easing transfer across tasks that live at different resolutions (e.g., domain classification vs.\ repository-level retrieval vs.\ function-level reasoning).

\subsection{Compatibility with Downstream Tasks}

TOLBERT remains a drop-in encoder: downstream tasks can ignore the tree heads and use $h$ as in BERT, or can exploit them as additional signals:
\begin{itemize}[leftmargin=1.5em]
    \item Use predicted domain and subdomain as features for routing tasks.
    \item Use tree-aware embeddings for retrieval (e.g., find nearest functions within the same repository branch).
    \item Use hierarchical predictions to prune search spaces for agentic systems navigating code or literature.
\end{itemize}

\section{Example: TOLBERT in a Code--Paper Universe}

To illustrate TOLBERT, consider a universe combining code repositories and AI research papers as in Figure~\ref{fig:tol_tree}. A function body extracted from \texttt{modeling\_llama.py} is mapped to a path $\pi(x)$ that starts at the root and traverses domain, subdomain, repository, file, and function.

During training, TOLBERT must:
\begin{itemize}[leftmargin=1.5em]
    \item Predict this path at multiple levels,
    \item Align embeddings of related functions in nearby regions,
    \item Maintain strong MLM performance on the code and comments.
\end{itemize}

At inference time, TOLBERT’s embedding can be used to:
\begin{itemize}[leftmargin=1.5em]
    \item Retrieve semantically and structurally related functions,
    \item Route queries to the correct repository or section of the universe,
    \item Provide multi-resolution explanations (e.g., ``this code belongs to Transformers $\rightarrow$ LLaMA $\rightarrow$ \texttt{hf/transformers}'').
\end{itemize}

\section{Experimental Protocol (Proposed)}

To measure TOLBERT’s improvement over BERT, we define tasks that explicitly probe hierarchical awareness.

\paragraph{Hierarchical Classification.}
Predict domain, subdomain, and artifact for a held-out instance. Evaluate accuracy at each level and path-consistency (whether the most probable nodes across levels form a valid path).

\paragraph{Branch-Constrained Retrieval.}
Given a query instance, retrieve nearest neighbors in embedding space. Measure the fraction of neighbors that are in the same branch, sub-branch, or leaf as the query. Compare TOLBERT vs.\ flat BERT.

\paragraph{Multi-Resolution Navigation.}
Starting from a query node, perform stepwise retrieval down the tree (domain $\rightarrow$ subdomain $\rightarrow$ repository $\rightarrow$ function). Evaluate whether TOLBERT’s embeddings enable efficient, branch-consistent navigation.

\paragraph{Downstream Tasks.}
Fine-tune on classic tasks (e.g., code search, paper recommendation, bug localization, QA) and compare performance and data efficiency. The central hypothesis is that TOLBERT’s hierarchical prior improves performance in settings where structure matters more than local surface statistics.

\section{Limitations and Future Work}

TOLBERT introduces dependency on a Tree-of-Life construction pipeline: if the tree is noisy or mis-specified, the hierarchical bias may harm performance. In domains with ambiguous or overlapping hierarchies, a strict tree may be insufficient; extensions to DAGs or hypergraphs may be necessary. Additionally, the increased supervision requires metadata or graph structure not always available for arbitrary text.

Future directions include:
\begin{itemize}[leftmargin=1.5em]
    \item Hyperbolic embeddings for better modeling of tree geometry.
    \item Joint learning of the tree and TOLBERT, rather than using a fixed tree.
    \item Integration with agentic systems: TOLBERT as the semantic backbone for agents navigating large code+paper universes via branch-aware search.
    \item Multi-modal extensions where nodes include not just text but also diagrams, figures, and execution traces.
\end{itemize}

\section{Conclusion}

TOLBERT generalizes BERT from a flat semantic encoder to a hierarchically grounded representation model aligned with a Tree of Life over the corpus. By combining MLM with multi-level hierarchical supervision, path-consistency regularization, and tree-aware contrastive learning, TOLBERT learns embeddings that both capture local contextual meaning and respect global knowledge structure. In large, structured domains---particularly those combining code, documentation, and research papers---this hierarchical inductive bias directly matches how practitioners conceptualize their universe, enabling more interpretable, navigable, and powerful representations than flat BERT.

\vspace{1em}
\noindent\textbf{Acknowledgements.} To be added.

\bibliographystyle{plain}
% \bibliography{references} % Uncomment and provide a .bib file for a full submission.

\end{document}
