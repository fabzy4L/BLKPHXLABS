from __future__ import annotations

import os
import argparse

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from google import genai
from google.genai import types


EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using only the "
    "provided excerpts from the dissertation. Cite the section title when relevant. "
    "If the excerpts don't contain enough information, say so."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c["metadatas"][0].get("header_title") or "Untitled section"
        text = c["documents"][0]
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n---\n\n".join(parts)


def query_chroma(collection, question: str) -> list[dict]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return [
        {"documents": [results["documents"][0][i]], "metadatas": [results["metadatas"][0][i]]}
        for i in range(len(results["documents"][0]))
    ]


def chat(collection_name: str, db_path: str) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable before running.")

    client = genai.Client(api_key=api_key)

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=collection_name, embedding_function=ef)

    print(f"Chatting with '{collection_name}' — type 'exit' to quit.\n")

    history: list[types.Content] = []

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break

        chunks = query_chroma(collection, question)
        context = build_context(chunks)
        prompt = f"Dissertation excerpts:\n\n{context}\n\nQuestion: {question}"

        history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )

        answer = response.text
        history.append(types.Content(role="model", parts=[types.Part(text=answer)]))

        print(f"\nGemini: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your dissertation via Gemini.")
    parser.add_argument("--collection", default="dissertation", help="ChromaDB collection name")
    parser.add_argument("--db", default="./chroma_db", help="Path to ChromaDB store")
    args = parser.parse_args()

    chat(args.collection, args.db)


if __name__ == "__main__":
    main()
