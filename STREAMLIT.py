# STREAMLIT.py
"""
DAISE Prototype - robust Streamlit entrypoint.

Behavior:
- Prefer new google-genai SDK (from google import genai)
- Fallback to legacy google.generativeai if new SDK not present
- List available models and let user pick a supported one
- Use SentenceTransformers for embeddings, FAISS if available (numpy fallback)
- Read GEMINI_API_KEY from Streamlit secrets or environment
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
    # create client only after key is acquired (below)
    GEN_CLIENT_KIND = "new"
except Exception:
    # try legacy
    try:
        import google.generativeai as legacy_genai
        GEN_CLIENT_KIND = "legacy"
    except Exception:
        GEN_CLIENT_KIND = None

# --- Streamlit page setup ---
st.set_page_config(page_title="DAISE Prototype (Gemini)", layout="wide")
st.title("DAISE – Early Innovation Discovery Prototype")

# --- Read API key (secrets or env) ---
def get_gemini_key() -> str:
    key = None
    # st.secrets works only when running under streamlit runtime and when .streamlit/secrets.toml exists or secrets set in cloud
    try:
        # using .get to avoid KeyError
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
    # new google-genai SDK
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
        # legacy configure style
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
    """
    Returns (list_of_model_names, label) where label describes the client kind.
    """
    models = []
    label = GEN_CLIENT_KIND or "none"
    if GEN_CLIENT is None:
        return models, label

    if GEN_CLIENT_KIND == "new":
        # new client: client.models.list() -> likely returns a sequence or object
        try:
            resp = GEN_CLIENT.models.list()
            # resp may be an iterable of model objects or a dict-like
            try:
                for m in resp:
                    # m might have .name or .model or 'name' key
                    if hasattr(m, "name"):
                        models.append(m.name)
                    elif isinstance(m, dict) and "name" in m:
                        models.append(m["name"])
                    else:
                        models.append(str(m))
            except TypeError:
                # Maybe resp is a single object with .models
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
        except Exception as e:
            # fallback: try known common model names (we will show these)
            models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    elif GEN_CLIENT_KIND == "legacy":
        try:
            # legacy client might have list_models() or list_models
            if hasattr(GEN_CLIENT, "list_models"):
                lm = GEN_CLIENT.list_models()
                # lm might be list of dicts
                if isinstance(lm, (list, tuple)):
                    for m in lm:
                        if isinstance(m, dict) and "name" in m:
                            models.append(m["name"])
                        else:
                            # the legacy lib often returns simple names
                            models.append(str(m))
                else:
                    # string or object
                    models.append(str(lm))
            elif hasattr(GEN_CLIENT, "models"):
                # some versions expose models
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

# Model selector - prefer recommended Gemini models when present
default_candidates = []
if "gemini-2.5-flash" in available_models:
    default_candidates.append("gemini-2.5-flash")
if "gemini-2.0-flash" in available_models:
    default_candidates.append("gemini-2.0-flash")
if "gemini-1.5-flash" in available_models:
    default_candidates.append("gemini-1.5-flash")
# fallback names for legacy
if not default_candidates and available_models:
    default_candidates.append(available_models[0])

model_choice = st.sidebar.selectbox("Model to use", options=default_candidates or ["auto"], index=0)

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
    # doc_embs shape (N, dim), q_emb shape (dim,) or (1, dim)
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
    try:
        resp = GEN_CLIENT.models.generate_content(model=model, contents=prompt)
        # try common fields
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and len(resp.candidates) > 0:
            c0 = resp.candidates[0]
            # try to extract content parts
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
        raise

def generate_with_legacy(prompt: str, model: str):
    """Use legacy google.generativeai"""
    try:
        # Legacy libraries have different function names by versions.
        # Try a few common patterns.
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
            # some legacy uses genai.chat.completions.create pattern
            if hasattr(GEN_CLIENT, "chat"):
                r = GEN_CLIENT.chat.create(model=model, messages=[{"role": "user", "content": prompt}])
                if isinstance(r, dict) and "candidates" in r:
                    return r["candidates"][0].get("content", str(r["candidates"][0]))
                return str(r)
        return "Legacy client returned unknown shape. Inspect logs."
    except Exception as e:
        raise

def generate_text(prompt: str, model: str):
    """Unified generation call that uses new SDK if available else legacy"""
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
    # extract text
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

        # embed and store
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
    # Build index / doc embeddings array
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

    # Query input
    query = st.text_input("Ask about the ingested documents (e.g., 'Summarize core innovation')")

    # Provide model selection info: ensure model_choice is set to a valid model name acceptable to the client
    chosen_model = model_choice
    if chosen_model == "auto":
        # pick a reasonable default
        if GEN_CLIENT_KIND == "new":
            # prefer gemini-2.0-flash if available
            if "gemini-2.0-flash" in available_models:
                chosen_model = "gemini-2.0-flash"
            elif "gemini-1.5-flash" in available_models:
                chosen_model = "gemini-1.5-flash"
            elif available_models:
                chosen_model = available_models[0]
            else:
                chosen_model = "gemini-2.0-flash"
        else:
            # legacy defaults
            if "text-bison-001" in available_models:
                chosen_model = "text-bison-001"
            elif available_models:
                chosen_model = available_models[0]
            else:
                chosen_model = "text-bison-001"

    if query:
        # search nearest doc
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

        # Build clear prompt for the model
        prompt = (
            "You are an investment analyst AI. Read the DOCUMENT and answer the USER QUESTION concisely.\n\n"
            "DOCUMENT:\n" + matched_text + "\n\n"
            "USER QUESTION: " + query + "\n\n"
            "Please provide:\n"
            "1) Short answer (2-4 sentences).\n"
            "2) 3 short evidence snippets quoting the document.\n"
            "3) One recommended next action for an analyst.\n"
        )

        # Call the model
        try:
            out = generate_text(prompt=prompt, model=chosen_model)
            st.subheader("AI Answer")
            st.write(out)
        except Exception as e:
            st.error(f"Model call failed: {type(e).__name__}: {e}")
            # If NotFound, encourage user to pick different model
            if "NotFound" in str(e) or "not found" in str(e).lower():
                st.error("Model not found for this client/API combination. Try picking another model in the sidebar or check your client type.")

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
            # Try to extract JSON from model output
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
