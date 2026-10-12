"""Chroma wrapper: build the chunk index and search it with metadata filters.

Design (Phase 1.1, see CLAUDE.md):
  - One collection ("chunks"); domain / document routing uses metadata filters.
  - Cosine distance; vectors are normalized, so score = 1 - distance.
  - Embeddings are computed by our Embedder and passed in explicitly. The
    collection is created with embedding_function=None, otherwise Chroma
    silently falls back to its own default model.
  - The embedding model name is stored on the collection; opening an index
    built with a different model fails instead of returning nonsense.

Usage:
    python -m src.indexing.vector_store          # (re)build the index from chunks.jsonl

    store = VectorStore()
    store.search("Which dataset does the YOLOv11 paper use?", domain="healthcare")
"""

import json
import time
from collections import Counter

import chromadb

from src.indexing.embeddings import Embedder
from src.ingestion.chunker import CHUNKS_PATH
from src.ingestion.inventory import PROJECT_ROOT

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "chunks"
ADD_BATCH_SIZE = 256
METADATA_FIELDS = ("doc_id", "chunk_index", "domain", "title", "authors", "section",
                   "section_kind", "page_start", "page_end", "overlap_chars", "n_tokens")


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = None

    # ------------------------------------------------------------ building ---

    def build(self) -> None:
        """Drop and rebuild the collection from chunks.jsonl."""
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        if COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(COLLECTION_NAME)
        collection = self.client.create_collection(
            COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
            metadata={"embedding_model": self.embedder.model_name},
        )

        vectors = self.embedder.embed_documents([c["embedding_text"] for c in chunks],
                                                show_progress=True)
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start:start + ADD_BATCH_SIZE]
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=vectors[start:start + ADD_BATCH_SIZE],
                documents=[c["text"] for c in batch],
                metadatas=[{k: c[k] for k in METADATA_FIELDS} for c in batch],
            )
        self._collection = collection

    # ------------------------------------------------------------- reading ---

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            collection = self.client.get_collection(COLLECTION_NAME, embedding_function=None)
            built_with = (collection.metadata or {}).get("embedding_model")
            if built_with != self.embedder.model_name:
                raise RuntimeError(
                    f"Index was built with {built_with!r} but the embedder is "
                    f"{self.embedder.model_name!r}. Rebuild: python -m src.indexing.vector_store")
            self._collection = collection
        return self._collection

    @staticmethod
    def _where(domain: str | None, doc_id: str | None) -> dict | None:
        clauses = [{k: v} for k, v in (("domain", domain), ("doc_id", doc_id)) if v]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, query: str, domain: str | None = None, doc_id: str | None = None,
               top_k: int = 5) -> list[dict]:
        """Semantic search, optionally filtered by domain and/or document.

        Returns hits sorted by score (cosine similarity, higher is better).
        """
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            where=self._where(domain, doc_id),
        )
        return [
            {"chunk_id": chunk_id, "score": round(1 - distance, 4), "text": text, **meta}
            for chunk_id, distance, text, meta in zip(
                result["ids"][0], result["distances"][0],
                result["documents"][0], result["metadatas"][0])
        ]

    def get_context(self, doc_id: str, chunk_index: int, window: int = 1) -> dict:
        """A chunk plus `window` neighbours on each side, stitched into one text.

        Overlap prefixes are stripped so repeated sentences appear once, and a
        section marker is inserted whenever the section changes.
        """
        indexes = list(range(max(0, chunk_index - window), chunk_index + window + 1))
        result = self.collection.get(
            where={"$and": [{"doc_id": doc_id}, {"chunk_index": {"$in": indexes}}]})
        rows = sorted(zip(result["documents"], result["metadatas"]),
                      key=lambda row: row[1]["chunk_index"])
        if not rows:
            raise KeyError(f"No chunks found for {doc_id} around index {chunk_index}")

        parts, previous_section = [], None
        for i, (text, meta) in enumerate(rows):
            if meta["section"] != previous_section:
                parts.append(f"[Section: {meta['section']}]")
                previous_section = meta["section"]
            elif i > 0:
                text = text[meta["overlap_chars"]:]
            parts.append(text)
        first, last = rows[0][1], rows[-1][1]
        return {
            "doc_id": doc_id,
            "title": first["title"],
            "chunk_indexes": [meta["chunk_index"] for _, meta in rows],
            "page_start": first["page_start"],
            "page_end": last["page_end"],
            "text": "\n\n".join(parts),
        }

    def stats(self) -> dict:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return {"chunks": len(metas),
                "documents": len({m["doc_id"] for m in metas}),
                "per_domain": dict(Counter(m["domain"] for m in metas).most_common())}


def main() -> None:
    store = VectorStore()
    start = time.time()
    store.build()
    print(f"Index built in {time.time() - start:.0f}s -> {CHROMA_DIR.relative_to(PROJECT_ROOT)}")
    print(store.stats())


if __name__ == "__main__":
    main()
