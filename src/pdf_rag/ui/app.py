import streamlit as st
import yaml

from src.pdf_rag.rag.embedding import embedding_data, search_similar_documents
from src.pdf_rag.rag.generate import generate_answer
from src.pdf_rag.rag.ingest import (
    debug_save_pdf_in_text,
    extract_text_from_pdf,
    save_pdf,
)

# load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file.read())

# PDF
PDF_DOC_DIR = config["pdf"]["doc_dir"]
PDF_TEXTS_DIR = config["pdf"]["text_dir"]

# EMBEDDING
EMBEDDING_MODEL = config["embedding"]["model"]
EMBEDDING_OVERLAP = config["embedding"]["overlap"]
EMBEDDING_CHUNK_SIZE = config["embedding"]["chunk_size"]
EMBEDDING_DB_DIR = config["embedding"]["db_dir"]

# LLM
LLM_MODEL = config["llm"]["model"]
LLM_TEMPERATURE = config["llm"]["temperature"]
LLM_TOP_K = config["llm"]["k"]
LLM_PROMPT = config["llm"]["prompt"]
LLM_MAX_TOKEN = config["llm"]["max_tokens"]

st.set_page_config(layout="wide")
st.title("📄 PDF RAG Assistant")
st.caption(f"Current Engine: {config['retriever']['type']} | Model: {LLM_MODEL}")

# Sider bar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        save_pdf(uploaded_file, PDF_DOC_DIR)
        docs = extract_text_from_pdf(uploaded_file)

        if docs:
            st.success(f"Extracted text from {uploaded_file.name}")

            # DEBUG:
            debug_save_pdf_in_text(uploaded_file, docs, PDF_TEXTS_DIR)

            db = embedding_data(
                docs,
                EMBEDDING_CHUNK_SIZE,
                EMBEDDING_OVERLAP,
                EMBEDDING_MODEL,
                EMBEDDING_DB_DIR,
            )
            st.info(f"Stored {uploaded_file.name}")

        else:
            st.error("Failed to extract/vectorized text from PDF.")

st.subheader("Ask a question")
with st.form(key="search_form", clear_on_submit=False):
    user_input = st.text_input(
        "Type your question here...", placeholder="Ex: What is the revenue target?"
    )
    submit_button = st.form_submit_button(label="Search")

if submit_button and user_input:
    # display user input
    with st.chat_message("user"):
        st.markdown(user_input)

    # rag
    with st.spinner("Searching and Generating..."):
        search_results = search_similar_documents(
            user_input, LLM_TOP_K, EMBEDDING_MODEL, EMBEDDING_DB_DIR, config
        )
        ans = generate_answer(
            user_input,
            search_results,
            LLM_MODEL,
            LLM_TEMPERATURE,
            LLM_PROMPT,
            LLM_MAX_TOKEN,
        )

    # display answer
    with st.chat_message("assistant"):
        st.write(ans)

        # display reference
        st.write("---")
        st.write(f"### References ({len(search_results)} results found)")

        for i, (doc, score) in enumerate(search_results):
            source = doc.metadata.get("source", "Unknown File")
            page = doc.metadata.get("page", "-")
            expander_title = f"#{i + 1} | {source} (Page: {page})"

            c1, c2, c3 = st.columns([0.4, 3, 0.6])

            with c1:
                st.write(f"#{i + 1}")

            with c2:
                st.markdown(
                    f"**{source}** <code style='color: #666; background: #f0f2f6; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>p.{page}</code>",
                    unsafe_allow_html=True,
                )
                st.caption(doc.page_content)

            with c3:
                confidence = max(0, min(100, int(score * 100)))
                color = (
                    "#28a745"
                    if confidence >= 70
                    else "#ffc107"
                    if confidence >= 40
                    else "#dc3545"
                )
                st.markdown(
                    f"<span style='color: {color}; font-weight: bold;'>{confidence}%</span>",
                    unsafe_allow_html=True,
                )
                st.progress(confidence / 100)

            st.divider()
