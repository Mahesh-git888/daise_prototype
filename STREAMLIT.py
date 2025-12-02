import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import google.generativeai as genai

# Load Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("DAISE – Early Innovation Discovery Prototype (Gemini Edition)")

uploaded_file = st.file_uploader("Upload paper/patent/GitHub README", type=["pdf", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    text = extract_text(uploaded_file)
    st.subheader("Extracted Text")
    st.write(text[:1500] + "...")

    # Embedding + Vector DB
    emb = model.encode([text])
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    query = st.text_input("Ask about the document")

    if query:
        q_emb = model.encode([query])
        D, I = index.search(q_emb, 1)

        # Call Gemini
        llm = genai.GenerativeModel("gemini-1.5-flash")

        response = llm.generate_content(
            f"Document:\n{text}\n\nQuery: {query}\nYou are an investment analyst AI. Provide a precise and structured answer."
        )

        st.subheader("AI Answer")
        st.write(response.text)

        # Scoring
        score_prompt = f"""
        Read this document and assign numeric scores (1–10):
        - Technology Novelty
        - Market Potential
        - Early Traction Signal

        Provide the result in a clean JSON format:
        {{
          "tech_novelty": ,
          "market_potential": ,
          "early_traction": ,
          "explanation": ""
        }}

        Document:\n{text[:4000]}
        """

        score_response = llm.generate_content(score_prompt)

        st.subheader("Scoring")
        st.write(score_response.text)
