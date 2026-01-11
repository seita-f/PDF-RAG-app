import os
import re
import unicodedata

import fitz
from langchain_core.documents import Document

from src.pdf_rag.utils.config import settings


# PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        else:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

        documents = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = _text_cleaning(text)

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_file.name, "page": page_num},
                    )
                )
        return documents

    except Exception as e:
        print(f"Error occurred in PyMuPDF: {e}")
        return None


def _text_cleaning(text):
    # "assess-\nment" -> "assessment"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = unicodedata.normalize("NFKC", text)

    return text


def debug_save_pdf_in_text(
    uploaded_file, extracted_docs, text_dir=settings.PDF_TEXTS_DIR
):
    save_path = text_dir
    os.makedirs(save_path, exist_ok=True)

    txt_filename = os.path.join(save_path, uploaded_file.name.replace(".pdf", ".txt"))

    with open(txt_filename, "w", encoding="utf-8") as file:
        if isinstance(extracted_docs, list):
            full_text = "\n\n--- Page Break ---\n\n".join(
                [doc.page_content for doc in extracted_docs]
            )
            file.write(full_text)
        else:
            file.write(extracted_docs)


def save_pdf(uploaded_file, doc_dir=settings.PDF_DOC_DIR):
    save_path = doc_dir
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, uploaded_file.name)

    # write in binary mode
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def main():
    # pass fo now
    pass


if __name__ == "__main__":
    main()
