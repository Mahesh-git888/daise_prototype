# STREAMLIT.py — Refactored, compact DAISE prototype
"""
Compact version of DAISE prototype:
- Detects google-genai (new) or google.generativeai (legacy).
- Lists available models when possible; allows manual override.
- Uses SentenceTransformers for embeddings; FAISS optional.
- Upload PDF/TXT, ingest, vector search, simple prompt + scoring calls.
"""
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Missing dependency: sentence-transformers. Add to requirements.txt.")
    raise

try:
    import pdfplumber
except Exception:
    pdfplumber = None

USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False

# ----------------------
# GenAI client detection
# ----------------------
GEN_CLIENT = None
GEN_KIND = None  # "new" or "legacy"

# Prefer new google-genai SDK if installed
try:
    from google import genai as genai_new  # type: ignore
    GEN_KIND = "new"
except Exception:
    try:
        import google.generativeai as genai_legacy  # type: ignore
        GEN_KIND = "legacy"
    except Exception:
        GEN_KIND = None

# ----------------------
# Streamlit setup + key
# ----------------------
st.set_page_config(page_title="DAISE (compact)", layout="wide")
st.title("DAISE — Innovation Discovery (compact)")

def get_api_key() -> Optional[str]:
    try:
        k = st.secrets.get("GEMINI_API_KEY") if st.secrets is not None else None
    except Exception:
        k = None
    if not k:
        k = os.getenv("GEMINI_API_KEY")
    return k

GEMINI_KEY = get_api_key()
if not GEMINI_KEY:
    st.error("GEMINI_API_KEY not found. Set Streamlit Secrets or environment variable.")
    st.stop()

# Initialize client
def init_client(api_key: str) -> Tuple[Optional[object], Optional[str]]:
    if GEN_KIND == "new":
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=api_key)
            return client, "new"
        except Exception as e:
            st.warning(f"Failed to init new genai client: {e}")
            return None, None
    elif GEN_KIND == "legacy":
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            return genai, "legacy"
        except Exception as e:
            st.warning(f"Failed to init legacy client: {e}")
            return None, None
    else:
        return None, None

GEN_CLIENT, GEN_KIND = init_client(GEMINI_KEY)

# ----------------------
# Model listing & chooser
# ----------------------
def list_models(client, kind: Optional[str]) -> List[str]:
    if client is None:
        return []
    try:
        if kind == "new":
            # try iterable response then fallback
            resp = client.models.list()
            models = []
            try:
                for m in resp:
                    if hasattr(m, "name"):
                        models.append(m.name)
                    elif isinstance(m, dict) and "name" in m:
                        models.append(m["name"])
                    else:
                        models.append(str(m))
            except TypeError:
                if hasattr(resp, "models"):
                    for m in resp.models:
                        models.append(getattr(m, "name", str(m)))
                else:
                    models = [str(resp)]
            return models
        elif kind == "legacy":
            if hasattr(client, "list_models"):
                lm = client.list_models()
                if isinstance(lm, (list, tuple)):
                    return [m.get("name", str(m)) if isinstance(m, dict) else str(m) for m in lm]
                return [str(lm)]
            # fallback common names
            return ["text-bison-001", "chat-bison-001"]
    except Exception:
        # Silent fallback
        if kind == "new":
            return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        elif kind == "legacy":
            return ["text-bison-001", "chat-bison-001"]
    return []

available_models = list_models(GEN_CLIENT, GEN_KIND)
st.sidebar.markdown("### GenAI Client")
st.sidebar.write(f"Client detected: **{GEN_KIND or 'none'}**")
st.sidebar.write(f"Client alive: **{bool(GEN_CLIENT)}**")
st.sidebar.write("Available models (preview):")
for m in available_models[:8]:
    st.sidebar.text(m)

# Manual override and final resolved model
manual = st.sidebar.checkbox("Use manual model name", value=False)
manual_model = st.sidebar.text_input("Manual model name", value="") if manual else ""
dropdown_options = ["auto"] + (available_models or [])
model_choice = st.sidebar.selectbox("Model (dropdown)", dropdown_options, index=0)
resolved_model = manual_model.strip() if manual and manual_model.strip() else model_choice

st.sidebar.markdown(f"**Resolved model:** `{resolved_model}`")

# ----------------------
# Embedding model (SBERT)
# ----------------------
@st.cache_resource
def load_sbert(name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

sbert = load_sbert()

def embed_texts(texts: List[str]) -> np.ndarray:
    emb = sbert.encode(texts, convert_to_numpy=True)
    return np.asarray(emb, dtype="float32")

def build_faiss_index(embs: np.ndarray):
    dim = embs.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embs)
    return idx

def numpy_cosine_search(doc_embs: np.ndarray, q_emb: np.ndarray, top_k=1):
    q = q_emb[0] if q_emb.ndim == 2 else q_emb
    d_norm = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    q_norm = q / np.linalg.norm(q)
    sims = (d_norm @ q_norm).reshape(-1)
    order = np.argsort(-sims)
    return order[:top_k], sims[order[:top_k]]

# ----------------------
# GenAI generate wrappers
# ----------------------
def generate_new(client, model: str, prompt: str) -> str:
    # Attempt the usual new SDK invocation patterns
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        # Many shapes possible: try common attributes
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            c0 = resp.candidates[0]
            # candidate may have text or content array
            if hasattr(c0, "text"):
                return c0.text
            if hasattr(c0, "content"):
                parts = []
                for item in c0.content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                if parts:
                    return "\n".join(parts)
        return str(resp)
    except Exception as e:
        raise

def generate_legacy(client, model: str, prompt: str) -> str:
    try:
        if hasattr(client, "generate_text"):
            r = client.generate_text(model=model, prompt=prompt)
            return getattr(r, "text", str(r))
        if hasattr(client, "generate"):
            r = client.generate(model=model, prompt=prompt)
            if isinstance(r, dict) and "candidates" in r:
                c = r["candidates"][0]
                return c.get("content", c.get("text", str(c)))
            return str(r)
        if hasattr(client, "chat"):
            r = client.chat.create(model=model, messages=[{"role": "user", "content": prompt}])
            if isinstance(r, dict) and "candidates" in r:
                return r["candidates"][0].get("content", str(r["candidates"][0]))
            return str(r)
        return "Legacy client: unknown response shape"
    except Exception:
        raise

def generate_text(prompt: str, model: str) -> str:
    if not model or model == "auto":
        raise RuntimeError("Model not specified (resolved to 'auto').")
    if GEN_CLIENT is None or GEN_KIND is None:
        raise RuntimeError("No GenAI client available.")
    if GEN_KIND == "new":
        return generate_new(GEN_CLIENT, model, prompt)
    else:
        return generate_legacy(GEN_CLIENT, model, prompt)

# ----------------------
# UI: upload / ingest / search
# ----------------------
if "docs" not in st.session_state:
    st.session_state.docs = []  # list of {"text": str, "emb": np.ndarray}

st.sidebar.markdown("---")
if st.sidebar.button("Clear documents"):
    st.session_state.docs = []

uploaded = st.file_uploader("Upload PDF or TXT (paper, patent, README)", type=["pdf", "txt"])
if uploaded:
    text = ""
    if uploaded.name.lower().endswith(".pdf"):
        if pdfplumber is None:
            st.error("pdfplumber not installed; cannot parse PDFs.")
        else:
            try:
                with pdfplumber.open(uploaded) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages)
            except Exception as e:
                st.error(f"PDF parse failed: {e}")
    else:
        try:
            raw = uploaded.read()
            text = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Text read failed: {e}")

    if text:
        st.write("Preview:")
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))
        try:
            emb = embed_texts([text])
            st.session_state.docs.append({"text": text, "emb": emb})
            st.success("Document ingested.")
        except Exception as e:
            st.error(f"Embedding failed: {e}")

if not st.session_state.docs:
    st.info("No documents ingested yet. Upload one to begin.")
    st.stop()

# Build index array and FAISS optional
doc_embs = np.vstack([d["emb"] for d in st.session_state.docs])  # shape (N, dim)
index = None
if USE_FAISS:
    try:
        index = build_faiss_index(doc_embs)
    except Exception:
        index = None

query = st.text_input("Ask about the ingested documents (e.g., 'Summarize core innovation')")

if query:
    q_emb = embed_texts([query])
    try:
        if index is not None:
            D, I = index.search(q_emb.astype("float32"), k=1)
            idx = int(I[0][0])
            matched_text = st.session_state.docs[idx]["text"]
            st.write(f"Matched doc: {idx} (L2 dist {float(D[0][0]):.4f})")
        else:
            idxs, sims = numpy_cosine_search(doc_embs, q_emb[0], top_k=1)
            idx = int(idxs[0])
            matched_text = st.session_state.docs[idx]["text"]
            st.write(f"Matched doc: {idx} (cosine {float(sims[0]):.4f})")
    except Exception as e:
        st.error(f"Search failed: {e}")
        matched_text = st.session_state.docs[0]["text"]

    # Construct prompt for answer and scoring
    q_prompt = (
        "You are an investment analyst AI. Read the DOCUMENT and answer the USER QUESTION concisely.\n\n"
        "DOCUMENT:\n" + matched_text + "\n\n"
        "USER QUESTION: " + query + "\n\n"
        "Please provide:\n"
        "1) Short answer (2-4 sentences).\n"
        "2) 3 short evidence snippets quoting the document.\n"
        "3) One recommended next action for an analyst.\n"
    )

    score_prompt = (
        "Read the DOCUMENT and assign numeric scores (1–10):\n"
        "- Technology Novelty\n- Market Potential\n- Early Traction Signal\n\n"
        "Return a valid JSON object with keys: "
        '"tech_novelty", "market_potential", "early_traction", "explanation"\n\n'
        "DOCUMENT:\n" + matched_text[:4000]
    )

    # Call model for answer
    try:
        out = generate_text(q_prompt, resolved_model)
        st.subheader("AI Answer")
        st.write(out)
    except Exception as e:
        st.error(f"Model call failed: {type(e).__name__}: {e}")
        msg = str(e).lower()
        if "not found" in msg or "404" in msg:
            st.error("Model not found for this client/API combo. Try manual model with a known model id.")
        st.stop()

    # Call model for scoring & try to parse JSON
    try:
        score_out = generate_text(score_prompt, resolved_model)
        parsed = None
        try:
            s = score_out
            start = s.find("{")
            end = s.rfind("}") + 1
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(s[start:end])
        except Exception:
            parsed = None
        st.subheader("Scoring")
        if parsed:
            st.json(parsed)
        else:
            st.write(score_out)
    except Exception as e:
        st.error(f"Scoring call failed: {type(e).__name__}: {e}")

st.markdown("---")
st.caption("DAISE compact — demo only. For production, persist vectors in a DB and secure keys.")
