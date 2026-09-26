"""Deterministic local TF-IDF retrieval; persistence contains JSON, never pickle."""

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

from trading_agent.models.domain import DomainModel

from .contracts import (
    DOCUMENT_SOURCES,
    GroundingChunk,
    GroundingDocument,
    GroundingRequest,
    RetrievedEvidence,
    SourceProvenance,
    SourceType,
    content_hash,
    is_fresh,
)


def chunk_document(
    document: GroundingDocument, *, chunk_words: int = 180
) -> tuple[GroundingChunk, ...]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    words = document.content.split()
    chunks = []
    for ordinal, start in enumerate(range(0, len(words), chunk_words)):
        content = " ".join(words[start : start + chunk_words])
        digest = content_hash(content)
        source_digest = content_hash(document.content)
        identity = json.dumps([document.source_id, source_digest, ordinal, digest])
        chunks.append(
            GroundingChunk(
                evidence_id=content_hash(identity),
                source_id=document.source_id,
                source_type=document.source_type,
                source_hash=source_digest,
                content_hash=digest,
                content=content,
                ordinal=ordinal,
                source_timestamp=document.source_timestamp,
            )
        )
    return tuple(chunks)


class GroundingRetriever(Protocol):
    def retrieve(self, request: GroundingRequest) -> tuple[RetrievedEvidence, ...]: ...


class LocalGroundingIndex(DomainModel):
    version: Literal[1] = 1
    chunk_words: int = Field(default=180, gt=0)
    documents: tuple[GroundingDocument, ...] = ()
    chunks: tuple[GroundingChunk, ...] = ()

    @model_validator(mode="after")
    def check_integrity(self) -> "LocalGroundingIndex":
        expected = tuple(
            chunk
            for doc in self.documents
            for chunk in chunk_document(doc, chunk_words=self.chunk_words)
        )
        if self.chunks != expected:
            raise ValueError("index content hash/checksum mismatch")
        return self

    @classmethod
    def from_documents(
        cls, documents: Iterable[GroundingDocument], *, chunk_words: int = 180
    ) -> "LocalGroundingIndex":
        ordered = tuple(sorted(documents, key=lambda d: (d.source_id, content_hash(d.content))))
        if len({doc.source_id for doc in ordered}) != len(ordered):
            raise ValueError("duplicate source IDs")
        return cls(
            documents=ordered,
            chunk_words=chunk_words,
            chunks=tuple(
                chunk for doc in ordered for chunk in chunk_document(doc, chunk_words=chunk_words)
            ),
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        target = path / "index.json"
        fd, name = tempfile.mkstemp(prefix=".index-", dir=path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(self.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            Path(name).replace(target)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "LocalGroundingIndex":
        return cls.model_validate_json((path / "index.json").read_text(encoding="utf-8"))

    def inspect(self, evidence_id: str) -> GroundingChunk | None:
        return next((c for c in self.chunks if c.evidence_id == evidence_id), None)

    def retrieve(self, request: GroundingRequest) -> tuple[RetrievedEvidence, ...]:
        candidates = [
            c
            for c in self.chunks
            if (not request.source_types or c.source_type in request.source_types)
            and (not request.source_ids or c.source_id in request.source_ids)
            and is_fresh(c.source_timestamp, request.as_of, request.max_age_seconds)
        ]
        if not candidates:
            return ()
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        try:
            matrix = vectorizer.fit_transform([c.content for c in self.chunks])
        except ValueError:  # Empty vocabulary, for example punctuation-only sources.
            return ()
        scores = np.asarray((matrix @ vectorizer.transform([request.query]).T).toarray()).ravel()
        ranked = sorted(
            ((chunk, scores[i]) for i, chunk in enumerate(self.chunks) if chunk in candidates),
            key=lambda item: (-float(item[1]), item[0].source_id, item[0].ordinal),
        )
        results: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for chunk, raw_score in ranked:
            score = min(1.0, max(0.0, float(raw_score)))
            if score <= 0 or score < request.min_score or chunk.content_hash in seen:
                continue
            seen.add(chunk.content_hash)
            results.append(
                RetrievedEvidence(
                    evidence_id=chunk.evidence_id,
                    content=chunk.content,
                    provenance=SourceProvenance(
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        source_hash=chunk.source_hash,
                        content_hash=chunk.content_hash,
                        retrieved_at=request.as_of,
                        source_timestamp=chunk.source_timestamp,
                        retrieval_score=score,
                        trust_class="trusted_document",
                    ),
                )
            )
            if len(results) == request.top_k:
                break
        return tuple(results)


def build_index(
    source: Path,
    output: Path,
    *,
    source_type: SourceType = SourceType.REPOSITORY_DOC,
    explicitly_trusted: bool = False,
    chunk_words: int = 180,
) -> LocalGroundingIndex:
    """The caller explicitly selects a trusted local root. No implicit home-directory scan."""
    if source_type not in DOCUMENT_SOURCES:
        raise ValueError("only trusted document source types may be indexed")
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = ".json" if source_type == SourceType.REGISTRY_METADATA else ".md"
    paths = sorted(source.rglob("*" + suffix)) if source.is_dir() else [source]
    documents = []
    root = source.resolve() if source.is_dir() else source.parent.resolve()
    for path in paths:
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("source symlink escapes explicit trusted source boundary")
        content = path.read_text(encoding="utf-8")
        if source_type == SourceType.REGISTRY_METADATA:
            json.loads(content)  # Reject malformed metadata; never deserialize model binaries.
        if not content.strip():
            continue
        documents.append(
            GroundingDocument(
                source_id=str(path),
                content=content,
                source_type=source_type,
                explicitly_trusted=explicitly_trusted,
            )
        )
    index = LocalGroundingIndex.from_documents(documents, chunk_words=chunk_words)
    index.save(output)
    return index


def load_documents(
    source: Path,
    *,
    trusted_artifacts: tuple[Path, ...] = (),
    registry: Path | None = None,
) -> tuple[GroundingDocument, ...]:
    """Read explicit local roots; skip symlinks and require research trust/validation flags."""
    root = source.resolve() if source.is_dir() else source.parent.resolve()
    paths = sorted(source.rglob("*.md")) if source.is_dir() else [source]
    documents: list[GroundingDocument] = []
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        kind = (
            SourceType.MODEL_CARD
            if "MODEL_CARD" in path.name.upper()
            else SourceType.PHASE_REPORT
            if "REPORT" in path.name.upper()
            else SourceType.REPOSITORY_DOC
        )
        documents.append(GroundingDocument(source_id=str(path), content=content, source_type=kind))
    if registry is not None:
        for path in sorted(registry.glob("*/metadata.json")):
            if path.is_symlink() or not path.resolve().is_relative_to(registry.resolve()):
                continue
            content = path.read_text(encoding="utf-8")
            if not isinstance(json.loads(content), dict):
                raise ValueError("registry metadata must be a JSON object")
            documents.append(
                GroundingDocument(
                    source_id=str(path), content=content, source_type=SourceType.REGISTRY_METADATA
                )
            )
    for path in trusted_artifacts:
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe trusted research artifact")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(artifact, dict)
            or artifact.get("trusted") is not True
            or artifact.get("validated") is not True
        ):
            raise ValueError("research artifact must be explicitly trusted and validated")
        documents.append(
            GroundingDocument(
                source_id=str(path),
                content=artifact["content"],
                source_type=SourceType.VALIDATED_RESEARCH,
                explicitly_trusted=True,
                source_timestamp=artifact.get("source_timestamp"),
            )
        )
    return tuple(documents)
