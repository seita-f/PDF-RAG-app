import os

import streamlit as st
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from streamlit_pdf_viewer import pdf_viewer

# from src.pdf_rag.rag.ingest import extract_text_from_pdf


def extract_text_from_pdf(pdf_file):
    try:
        laparams = LAParams(
            all_texts=True
        )  # all_texts: include text in graph, table etc

        text = extract_text(pdf_file, laparams=laparams)
        return text

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


st.set_page_config(layout="wide")
st.title("📄 PDF RAG Assistant")

# Sider bar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            st.success(f"Registered {uploaded_file.name}")

            # 保存ディレクトリの指定（例: data フォルダ）
            save_path = "data/uploads"
            os.makedirs(save_path, exist_ok=True)

            txt_filename = os.path.join(
                save_path, uploaded_file.name.replace(".pdf", ".txt")
            )

            with open(txt_filename, "w", encoding="utf-8") as file:
                file.write(extracted_text)

            st.info(f"Text saved to {txt_filename}")

        else:
            st.error("Failed to extract text from PDF.")


col_chat, col_pdf = st.columns([1.5, 1])

with col_chat:
    st.subheader("Chat")
    # chat
    if prompt := st.chat_input("Ex: What is the revenue target for 2024?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG
        ans = "Answer: It is ... (Mock now)"
        sources = [
            {"name": "manual.pdf", "score": 0.92, "text": "MOCK ANS", "page": 5},
            {"name": "info.pdf", "score": 0.85, "text": "MOCK ANS", "page": 12},
        ]

        with st.chat_message("assistant"):
            st.write(ans)

            # Reference Table Section
            num_results = len(sources)
            st.write(f"### References ({num_results} results found)")
            st.markdown(
                """
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <div style="display: flex;">
                        <div style="flex: 0.5; font-weight: bold;">Rank</div>
                        <div style="flex: 2; font-weight: bold;">Source Name / Snippet</div>
                        <div style="flex: 0.5; font-weight: bold;">Action</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # data rows
            for i, s in enumerate(sources):
                c1, c2, c3 = st.columns([0.5, 2, 0.5])

                with c1:
                    st.write(f"#{i + 1}")
                with c2:
                    st.markdown(f"**{s['name']}**")
                    st.caption(f"{s['text'][:50]}...")
                # with c3:
                #     # Score
                #     st.write(f"{int(s['score'] * 100)}%")
                with c3:
                    if st.button(
                        "Display PDF", key=f"btn_{i}", use_container_width=True
                    ):
                        st.session_state.selected_pdf = s

                st.divider()

with col_pdf:
    st.subheader("PDF Preview")
    if "selected_pdf" in st.session_state:
        s = st.session_state.selected_pdf
        st.info(f"Document Viewer: {s['name']} (p.{s['page']})")
        pdf_viewer(f"data/uploads/{s['name']}", width=400)
    else:
        st.write("Select 'View PDF' from the references to preview the document here.")
