from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Basic identifiers
EntityId = str


@dataclass(frozen=True)
class Span:
    """Inclusive 1-based line span within an artifact."""

    start_line: int
    end_line: int


@dataclass
class Artifact:
    """
    Minimal file-level artifact in a program graph.

    - `uri` is typically a `program://` URI.
    - `type` is a coarse tag such as "source", "header", "binary".
    - `hash` is an implementation-defined content hash (may be empty).
    - `span` is optional; whole-file artifacts can leave it as None.
    """

    uri: str
    type: str
    hash: str
    span: Optional[Span] = None


@dataclass
class Entity:
    """
    Logical entity in a program graph.

    Common patterns:
      - Repository / project node.
      - File / module node.
      - Function / method / class node.

    `artifact_uri` and `span` are optional and only populated when the
    entity corresponds to a concrete region in a file.
    """

    id: EntityId
    kind: str
    uri: str
    artifact_uri: Optional[str] = None
    span: Optional[Span] = None
    labels: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """
    Typed relationship between entities.

    Typical edge types:
      - "owns": containment (repo→file, file→function, module→class, ...)
      - "imports": dependency between modules / files / packages
      - "calls": function/method call
      - "tests": test module or case covering a target
      - "similar_to": embedding- or heuristic-based similarity
    """

    src: EntityId
    dst: EntityId
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedAnchor:
    """
    Resolution of a URI back onto a concrete artifact region.
    """

    artifact_uri: str
    span: Span
    hash: str


class ProgramGraph:
    """
    Abstract, language-agnostic view over a program / repository.

    Implementations are expected to provide at least:
      - entities(): iterable of Entity nodes
      - edges(): iterable of Edge relationships
      - artifacts(): file-level view
      - resolve(): URI → concrete anchor

    Subclasses may add richer APIs as needed.
    """

    # The default implementation is intentionally skeletal and returns
    # empty views; concrete graphs should override these.

    def entities(self) -> Iterable[Entity]:
        return []

    def edges(self) -> Iterable[Edge]:
        return []

    def artifacts(self, kind: str) -> Iterable[Artifact]:
        return []

    def resolve(self, uri: str) -> ResolvedAnchor:
        raise NotImplementedError("resolve() must be implemented by subclasses")

    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
        return []

    def subgraph(self, seeds: List[EntityId], radius: int) -> "ProgramGraph":
        return self


