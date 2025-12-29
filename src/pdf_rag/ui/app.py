import streamlit as st

from src.pdf_rag.rag.embedding import embedding_data, search_similar_documents
from src.pdf_rag.rag.generate import generate_answer
from src.pdf_rag.rag.ingest import (
    extract_text_from_pdf,
    save_pdf,
)

st.set_page_config(layout="wide")
st.title("📄 PDF RAG Assistant")

# Sider bar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        save_pdf(uploaded_file)
        extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            st.success(f"Extracted text from {uploaded_file.name}")

            # DEBUG:
            # debug_save_pdf_in_text(uploaded_file, extracted_text)

            db = embedding_data(extracted_text, uploaded_file.name)
            st.info(f"Stored {uploaded_file.name} in vectordb")

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
        search_results = search_similar_documents(user_input)
        ans = generate_answer(user_input, search_results)

    # display answer
    with st.chat_message("assistant"):
        st.write(ans)

        # display reference
        st.write("---")
        st.write(f"### References ({len(search_results)} results found)")

        for i, (doc, score) in enumerate(search_results):
            c1, c2, c3 = st.columns([0.4, 3, 0.6])

            with c1:
                st.write(f"#{i + 1}")

            with c2:
                st.markdown(
                    f"**{doc.metadata.get('source', 'Unknown')}** (p.{doc.metadata.get('page', 1)})"
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
