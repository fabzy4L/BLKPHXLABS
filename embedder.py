from __future__ import annotations

import argparse
import json

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_chunks(jsonl_path: str) -> list[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_and_store(jsonl_path: str, collection_name: str, db_path: str) -> None:
    chunks = load_chunks(jsonl_path)
    if not chunks:
        print("No chunks found in input file.")
        return

    client = chromadb.PersistentClient(path=db_path)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(c["chunk_index"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_file": c["source_file"],
            "header_title": c["header_title"],
            "timestamp": c["timestamp"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    for i in range(0, len(chunks), BATCH_SIZE):
        collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
        print(f"  Stored chunks {i + 1}–{min(i + BATCH_SIZE, len(chunks))} / {len(chunks)}")

    print(f"\nDone. {len(chunks)} chunks in collection '{collection_name}' at '{db_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed JSONL chunks and store in ChromaDB.")
    parser.add_argument("input", help="Path to the .jsonl file from chunker.py")
    parser.add_argument(
        "--collection", default="dissertation", help="ChromaDB collection name (default: dissertation)"
    )
    parser.add_argument(
        "--db", default="./chroma_db", help="Directory for the ChromaDB store (default: ./chroma_db)"
    )
    args = parser.parse_args()

    embed_and_store(args.input, args.collection, args.db)


if __name__ == "__main__":
    main()
