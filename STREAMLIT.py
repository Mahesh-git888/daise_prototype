# app.py
import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import json
from typing import Optional

# Try to import faiss; if not available, we'll use numpy fallback
USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False

# Import the new Google GenAI SDK client
# Make sure you installed google-genai in the environment
from google import genai

st.set_page_config(layout="wide", page_title="DAISE Prototype (Gemini)")

st.title("DAISE – Early Innovation Discovery Prototype (Gemini Edition)")

# --- Gemini client init ---
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") if st.secrets is not None else None
GEMINI_KEY = GEMINI_KEY or os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    st.error("GEMINI_API_KEY not found. Set it in Streamlit Secrets or as an environment variable.")
    st.stop()

client = genai.Client(api_key=GEMINI_KEY)

# --- Load SBERT ---
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

sbert = load_sbert()

# --- UI: model selector ---
model_choice = st.selectbox("Select Gemini model (demo)", ["gemini-2.0-flash", "gemini-1.5-flash"])

# --- File upload ---
uploaded_file = st.file_uploader("Upload paper / patent / GitHub README (pdf or txt)", type=["pdf", "txt"])
persist_docs = []  # in-memory store of docs (for demo only)

def extract_text(file) -> str:
    """Extract text from PDF or plain text file-like object."""
    try:
        if file.name.lower().endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
        else:
            raw = file.read()
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

def safe_generate(model: str, prompt: str, max_output_chars: Optional[int]=4000):
    """Call Gemini and return best-available text response (robust to client shape)."""
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        # Try common response shapes
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # fallback: candidates -> content
        try:
            # resp.candidates is sometimes present
            if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                # candidate may have 'content' which can be a list of items with .text or .content
                c0 = resp.candidates[0]
                if hasattr(c0, "content"):
                    # content is often a list of dicts with 'text'
                    parts = []
                    for item in c0.content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif hasattr(item, "text"):
                            parts.append(item.text)
                    return "\n".join(parts) if parts else str(c0)
                elif hasattr(c0, "text"):
                    return c0.text
        except Exception:
            pass
        # last fallback: str(resp)
        return str(resp)
    except Exception as e:
        # bubble up useful error
        raise

# --- Embedding + search helpers ---
def embed_texts(texts):
    """Return numpy float32 embeddings for list of texts."""
    emb = sbert.encode(texts, convert_to_numpy=True)
    return np.asarray(emb, dtype="float32")

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index

def numpy_cosine_search(doc_embs, q_emb, top_k=1):
    # normalize
    d_norm = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    q_norm = q_emb / np.linalg.norm(q_emb)
    sims = (d_norm @ q_norm).reshape(-1)
    order = np.argsort(-sims)
    return order[:top_k], sims[order[:top_k]]

# --- Main flow ---
if uploaded_file:
    text = extract_text(uploaded_file)
    if not text.strip():
        st.warning("No text extracted from the file.")
    else:
        st.subheader("Document preview")
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))
        # compute embeddings
        with st.spinner("Computing embeddings..."):
            doc_emb = embed_texts([text])  # shape (1, dim)
        # store in-memory for this session
        persist_docs.append({"text": text, "emb": doc_emb})

        # Build index (FAISS or numpy)
        if USE_FAISS:
            try:
                index = build_faiss_index(np.vstack([d["emb"] for d in persist_docs]))
            except Exception as e:
                st.warning(f"FAISS index creation failed; switching to NumPy fallback. ({e})")
                USE_FAISS = False
                doc_embs = np.vstack([d["emb"] for d in persist_docs])
        else:
            doc_embs = np.vstack([d["emb"] for d in persist_docs])

        query = st.text_input("Ask about the document (example: 'What is the core innovation?')", key="query1")
        if query:
            q_emb = embed_texts([query])  # shape (1, dim)
            if USE_FAISS:
                try:
                    D, I = index.search(q_emb, k=1)
                    idx = int(I[0][0])
                    distance = float(D[0][0])
                    st.write(f"Closest doc index: {idx} (distance {distance:.4f})")
                    matched_doc = persist_docs[idx]["text"]
                except Exception as e:
                    st.warning(f"FAISS search error, using numpy fallback: {e}")
                    USE_FAISS = False
                    doc_embs = np.vstack([d["emb"] for d in persist_docs])
                    idxs, sims = numpy_cosine_search(doc_embs, q_emb[0], top_k=1)
                    idx = int(idxs[0])
                    matched_doc = persist_docs[idx]["text"]
            else:
                idxs, sims = numpy_cosine_search(doc_embs, q_emb[0], top_k=1)
                idx = int(idxs[0])
                st.write(f"Closest doc index: {idx} (cosine {float(sims[0]):.4f})")
                matched_doc = persist_docs[idx]["text"]

            # Prepare LLM prompt for answer
            prompt = (
                "You are an investment analyst AI. Read the document and answer the user's question "
                "concisely and with evidence pointers.\n\nDOCUMENT:\n"
                f"{matched_doc}\n\nUSER QUESTION: {query}\n\nProvide:\n"
                "1) A short concise answer (2-5 sentences).\n"
                "2) 3 evidence lines referencing parts of the document (quote short snippets).\n"
                "3) A recommended next action for an analyst (1 line).\n"
            )

            try:
                ai_output = safe_generate(model_choice, prompt)
                st.subheader("AI Answer")
                st.write(ai_output)
            except Exception as e:
                st.error(f"Generative call failed: {type(e).__name__}: {e}")

            # Prepare scoring prompt
            score_prompt = (
                "Read the document and assign numeric scores (1–10):\n"
                "- Technology Novelty\n- Market Potential\n- Early Traction Signal\n\n"
                "Return a valid JSON object with keys: "
                '"tech_novelty", "market_potential", "early_traction", "explanation"\n\n'
                "DOCUMENT:\n" + (matched_doc[:4000])
            )

            try:
                score_output = safe_generate(model_choice, score_prompt)
                # attempt to extract JSON from model output
                parsed = None
                try:
                    start = score_output.find("{")
                    end = score_output.rfind("}") + 1
                    if start != -1 and end != -1 and end > start:
                        maybe = score_output[start:end]
                        parsed = json.loads(maybe)
                except Exception:
                    parsed = None

                st.subheader("Scoring")
                if parsed:
                    st.json(parsed)
                else:
                    st.write(score_output)
            except Exception as e:
                st.error(f"Scoring call failed: {type(e).__name__}: {e}")

# Footer
st.markdown("---")
st.caption("DAISE prototype — minimal demo. Keep keys secret. For production, persist vectors in a vector DB and add access control.")
