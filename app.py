from __future__ import annotations

import os

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from google import genai
from google.genai import types


EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 5
DB_PATH = "./chroma_db"
COLLECTION_NAME = "dissertation"

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using only the "
    "provided excerpts from the dissertation. Cite the section title when relevant. "
    "If the excerpts don't contain enough information, say so."
)


@st.cache_resource
def load_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def query_chroma(collection, question: str) -> list[dict]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return [
        {"document": results["documents"][0][i], "metadata": results["metadatas"][0][i]}
        for i in range(len(results["documents"][0]))
    ]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c["metadata"].get("header_title") or "Untitled section"
        parts.append(f"[{i}] {title}\n{c['document']}")
    return "\n\n---\n\n".join(parts)


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    if not api_key:
        st.error("GEMINI_API_KEY not set. Add it as an environment variable or in .streamlit/secrets.toml.")
        st.stop()
    return genai.Client(api_key=api_key)


st.set_page_config(page_title="Dissertation Chat", page_icon="📄", layout="centered")
st.title("📄 Dissertation Chat")

collection = load_collection()
gemini = get_gemini_client()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the dissertation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = query_chroma(collection, prompt)
    context = build_context(chunks)
    full_prompt = f"Dissertation excerpts:\n\n{context}\n\nQuestion: {prompt}"

    st.session_state.history.append(
        types.Content(role="user", parts=[types.Part(text=full_prompt)])
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=st.session_state.history,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            answer = response.text

        st.markdown(answer)

        with st.expander("Sources", expanded=False):
            for i, c in enumerate(chunks, 1):
                title = c["metadata"].get("header_title") or "Untitled section"
                st.markdown(f"**[{i}] {title}**")
                st.caption(c["document"][:300] + "...")

    st.session_state.history.append(
        types.Content(role="model", parts=[types.Part(text=answer)])
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
