# STREAMLIT.py
"""
DAISE Prototype - robust Streamlit entrypoint.

Improvements applied:
- Better handling of Google GenAI new vs legacy SDKs
- If model not found, surface available models and allow manual entry
- More robust generate call for new SDK (handles string/list contents)
- Clearer error messages when model is unsupported
- Keep SBERT/FAISS behavior unchanged
"""

import os
import streamlit as st
import json
import numpy as np
from typing import Tuple

# --- Optional heavy imports wrapped in try/except ---
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Missing sentence-transformers. Please add it to requirements.txt.")
    raise

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# FAISS optional
USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False

# --- GenAI client compatibility wrapper ---
GEN_CLIENT = None
GEN_CLIENT_KIND = None  # "new" or "legacy"

# Try new SDK first
try:
    from google import genai  # google-genai SDK
    GEN_CLIENT_KIND = "new"
except Exception:
    # try legacy
    try:
        import google.generativeai as legacy_genai
        GEN_CLIENT_KIND = "legacy"
    except Exception:
        GEN_CLIENT_KIND = None

# --- Streamlit page setup ---
st.set_page_config(page_title="DAISE Prototype", layout="wide")
st.title("DAISE – Early Innovation Discovery Prototype")

# --- Read API key (secrets or env) ---

def get_gemini_key() -> str:
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY") if st.secrets is not None else None
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key

GEMINI_KEY = get_gemini_key()
if not GEMINI_KEY:
    st.error(
        "GEMINI_API_KEY not found. Set it in Streamlit Secrets (recommended) or set the GEMINI_API_KEY environment variable."
    )
    st.stop()

# Initialize whichever client is present
if GEN_CLIENT_KIND == "new":
    try:
        from google import genai as genai_new
        client_new = genai_new.Client(api_key=GEMINI_KEY)
        GEN_CLIENT = client_new
        GEN_CLIENT_KIND = "new"
    except Exception as e:
        st.warning(f"Failed to initialize google-genai client: {e}")
        GEN_CLIENT = None
        GEN_CLIENT_KIND = None
elif GEN_CLIENT_KIND == "legacy":
    try:
        import google.generativeai as legacy_genai  # type: ignore
        legacy_genai.configure(api_key=GEMINI_KEY)
        GEN_CLIENT = legacy_genai
        GEN_CLIENT_KIND = "legacy"
    except Exception as e:
        st.warning(f"Failed to initialize legacy google.generativeai client: {e}")
        GEN_CLIENT = None
        GEN_CLIENT_KIND = None
else:
    st.error(
        "No Google GenAI client package detected. Install `google-genai` (preferred) or `google-generativeai` (legacy)."
    )
    st.stop()

# --- Helper: list available models for the detected client ---
def list_models_for_client() -> Tuple[list, str]:
    models = []
    label = GEN_CLIENT_KIND or "none"
    if GEN_CLIENT is None:
        return models, label

    if GEN_CLIENT_KIND == "new":
        try:
            # genai.Client.models.list() exists in the new SDK
            resp = GEN_CLIENT.models.list()
            # resp may be iterable
            try:
                for m in resp:
                    if hasattr(m, "name"):
                        models.append(m.name)
                    elif isinstance(m, dict) and "name" in m:
                        models.append(m["name"])
                    else:
                        models.append(str(m))
            except TypeError:
                # maybe resp.models
                if hasattr(resp, "models"):
                    for m in resp.models:
                        if hasattr(m, "name"):
                            models.append(m.name)
                        elif isinstance(m, dict) and "name" in m:
                            models.append(m["name"])
                        else:
                            models.append(str(m))
                else:
                    models.append(str(resp))
        except Exception:
            # fallback list of commonly expected names
            models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    elif GEN_CLIENT_KIND == "legacy":
        try:
            if hasattr(GEN_CLIENT, "list_models"):
                lm = GEN_CLIENT.list_models()
                if isinstance(lm, (list, tuple)):
                    for m in lm:
                        if isinstance(m, dict) and "name" in m:
                            models.append(m["name"])
                        else:
                            models.append(str(m))
                else:
                    models.append(str(lm))
            elif hasattr(GEN_CLIENT, "models"):
                lm = GEN_CLIENT.models()
                if isinstance(lm, (list, tuple)):
                    models.extend([str(x) for x in lm])
                else:
                    models.append(str(lm))
            else:
                models = ["text-bison-001", "chat-bison-001"]
        except Exception:
            models = ["text-bison-001", "chat-bison-001"]
    return models, label

available_models, client_label = list_models_for_client()

# Present detected client and models
st.sidebar.markdown("### GenAI Client")
st.sidebar.write(f"Detected client: **{client_label}**")
if available_models:
    st.sidebar.markdown("Available model examples (first few):")
    for m in available_models[:10]:
        st.sidebar.text(m)
else:
    st.sidebar.warning("Could not list models programmatically; a few common model names are shown instead.")

# Allow user to manually enter a model name if not present in auto list
manual_model = st.sidebar.text_input("Manual model name (optional)")

# Model selector - prefer recommended Gemini models when present
default_candidates = []
for cand in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
    if cand in available_models:
        default_candidates.append(cand)
if not default_candidates and available_models:
    default_candidates.append(available_models[0])

# If no detected models, show an 'auto' and manual entry
if not default_candidates:
    default_candidates = ["auto"]

model_choice = st.sidebar.selectbox("Model to use", options=default_candidates, index=0)

# If user provided manual name, prefer it
if manual_model:
    chosen_model = manual_model.strip()
else:
    chosen_model = model_choice

# --- Load SBERT once ---
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sbert = load_sbert_model()

# --- In-memory persistence across session for demo ---
if "docs" not in st.session_state:
    st.session_state.docs = []  # list of dicts {"text":..., "emb": np.array}

# --- Utility functions for embeddings & search ---
def embed_texts(texts):
    emb = sbert.encode(texts, convert_to_numpy=True)
    return np.asarray(emb, dtype="float32")


def build_faiss_index(embs: np.ndarray):
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index


def numpy_cosine_search(doc_embs: np.ndarray, q_emb: np.ndarray, top_k=1):
    if q_emb.ndim == 2:
        q = q_emb[0]
    else:
        q = q_emb
    d_norm = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    q_norm = q / np.linalg.norm(q)
    sims = (d_norm @ q_norm).reshape(-1)
    order = np.argsort(-sims)
    return order[:top_k], sims[order[:top_k]]

# --- Generation helpers that support both new and legacy clients ---

def generate_with_new(prompt: str, model: str):
    """Use google-genai client (genai.Client)"""
    # new SDK expects contents parameter (string or list)
    try:
        contents = prompt
        resp = GEN_CLIENT.models.generate_content(model=model, contents=contents)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # candidates/content parsing
        if hasattr(resp, "candidates") and len(resp.candidates) > 0:
            c0 = resp.candidates[0]
            # try content items
            if hasattr(c0, "content"):
                parts = []
                for item in c0.content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                if parts:
                    return "\n".join(parts)
            if hasattr(c0, "text"):
                return c0.text
        return str(resp)
    except Exception as e:
        # bubble up for caller to handle; keep message intact
        raise


def generate_with_legacy(prompt: str, model: str):
    """Use legacy google.generativeai"""
    try:
        if hasattr(GEN_CLIENT, "generate_text"):
            r = GEN_CLIENT.generate_text(model=model, prompt=prompt)
            if hasattr(r, "text"):
                return r.text
            return str(r)
        elif hasattr(GEN_CLIENT, "generate"):
            r = GEN_CLIENT.generate(model=model, prompt=prompt)
            if isinstance(r, dict) and "candidates" in r:
                c = r["candidates"][0]
                return c.get("content", c.get("text", str(c)))
            return str(r)
        else:
            if hasattr(GEN_CLIENT, "chat"):
                r = GEN_CLIENT.chat.create(model=model, messages=[{"role": "user", "content": prompt}])
                if isinstance(r, dict) and "candidates" in r:
                    return r["candidates"][0].get("content", str(r["candidates"][0]))
                return str(r)
        return "Legacy client returned unknown shape. Inspect logs."
    except Exception as e:
        raise


def generate_text(prompt: str, model: str):
    if GEN_CLIENT_KIND == "new":
        return generate_with_new(prompt, model)
    elif GEN_CLIENT_KIND == "legacy":
        return generate_with_legacy(prompt, model)
    else:
        raise RuntimeError("No GenAI client available")

# --- UI: File upload + ingestion ---
st.sidebar.markdown("---")
st.sidebar.markdown("Demo controls")
ingest_button = st.sidebar.button("Clear ingested docs")

if ingest_button:
    st.session_state.docs = []

uploaded = st.file_uploader("Upload PDF or TXT (paper, patent, README)", type=["pdf", "txt"]) 
if uploaded:
    text = ""
    if uploaded.name.lower().endswith(".pdf"):
        if pdfplumber is None:
            st.error("pdfplumber not installed. Add to requirements to parse PDFs.")
        else:
            try:
                with pdfplumber.open(uploaded) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    text = "\n".join(pages)
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
                text = ""
    else:
        try:
            raw = uploaded.read()
            text = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Text extraction failed: {e}")
            text = ""

    if text:
        st.write("Document preview:")
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

        try:
            emb = embed_texts([text])  # shape (1, dim)
            st.session_state.docs.append({"text": text, "emb": emb})
            st.success("Document ingested into session.")
        except Exception as e:
            st.error(f"Embedding failed: {e}")

# If no docs ingested, show a hint
if not st.session_state.docs:
    st.info("No documents ingested. Upload a PDF or TXT to start.")
else:
    doc_embs = np.vstack([d["emb"] for d in st.session_state.docs])
    if USE_FAISS:
        try:
            index = build_faiss_index(doc_embs)
        except Exception as e:
            st.warning(f"FAISS index creation failed: {e} — switching to NumPy fallback.")
            index = None
            USE_FAISS = False
    else:
        index = None

    query = st.text_input("Ask about the ingested documents (e.g., 'Summarize core innovation')")

    # If chosen_model is 'auto', resolve into a concrete model
    if chosen_model == "auto":
        if GEN_CLIENT_KIND == "new":
            if "gemini-2.5-flash" in available_models:
                chosen_model = "gemini-2.5-flash"
            elif "gemini-2.0-flash" in available_models:
                chosen_model = "gemini-2.0-flash"
            elif available_models:
                chosen_model = available_models[0]
            else:
                chosen_model = "gemini-2.0-flash"
        else:
            if "text-bison-001" in available_models:
                chosen_model = "text-bison-001"
            elif available_models:
                chosen_model = available_models[0]
            else:
                chosen_model = "text-bison-001"

    if query:
        q_emb = embed_texts([query])
        try:
            if index is not None and USE_FAISS:
                D, I = index.search(q_emb.astype("float32"), k=1)
                idx = int(I[0][0])
                matched_text = st.session_state.docs[idx]["text"]
                st.write(f"Matched document index: {idx}, distance: {float(D[0][0]):.4f}")
            else:
                idxs, sims = numpy_cosine_search(doc_embs, q_emb[0], top_k=1)
                idx = int(idxs[0])
                matched_text = st.session_state.docs[idx]["text"]
                st.write(f"Matched document index: {idx}, cosine: {float(sims[0]):.4f}")
        except Exception as e:
            st.error(f"Search failed: {e}")
            matched_text = st.session_state.docs[0]["text"]

        prompt = (
            "You are an investment analyst AI. Read the DOCUMENT and answer the USER QUESTION concisely.\n\n"
            "DOCUMENT:\n" + matched_text + "\n\n"
            "USER QUESTION: " + query + "\n\n"
            "Please provide:\n"
            "1) Short answer (2-4 sentences).\n"
            "2) 3 short evidence snippets quoting the document.\n"
            "3) One recommended next action for an analyst.\n"
        )

        # Call the model with a robust error handler
        try:
            out = generate_text(prompt=prompt, model=chosen_model)
            st.subheader("AI Answer")
            st.write(out)
        except Exception as e:
            st.error(f"Model call failed: {type(e).__name__}: {e}")
            msg = str(e).lower()
            if "not found" in msg or "not_found" in msg or "404" in msg:
                st.error("Model not found for this client/API combination.")
                if available_models:
                    st.info("Please pick one of the models listed in the sidebar or paste a valid model name in 'Manual model name'.")
                else:
                    st.info("No models were discovered programmatically. Use the Manual model name field to try a known model like 'gemini-2.5-flash' or 'text-bison-001'.")

        # Scoring prompt
        score_prompt = (
            "Read the DOCUMENT and assign numeric scores (1–10):\n"
            "- Technology Novelty\n- Market Potential\n- Early Traction Signal\n\n"
            "Return a valid JSON object with keys: "
            '"tech_novelty", "market_potential", "early_traction", "explanation"\n\n'
            "DOCUMENT:\n" + matched_text[:4000]
        )

        try:
            score_out = generate_text(prompt=score_prompt, model=chosen_model)
            parsed = None
            try:
                s = score_out
                start = s.find("{")
                end = s.rfind("}") + 1
                if start != -1 and end != -1 and end > start:
                    maybe = s[start:end]
                    parsed = json.loads(maybe)
            except Exception:
                parsed = None

            st.subheader("Scoring")
            if parsed:
                st.json(parsed)
            else:
                st.write(score_out)
        except Exception as e:
            st.error(f"Scoring call failed: {type(e).__name__}: {e}")

# Footer
st.markdown("---")
st.caption("DAISE prototype — demo only. For production, persist vectors in a vector DB and secure keys.")
