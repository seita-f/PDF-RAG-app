import os

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def save_pdf(uploaded_file):
    pass


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


def debug_save_pdf_in_text(uploaded_file, extracted_text):
    save_path = "data/uploads"
    os.makedirs(save_path, exist_ok=True)

    txt_filename = os.path.join(save_path, uploaded_file.name.replace(".pdf", ".txt"))

    with open(txt_filename, "w", encoding="utf-8") as file:
        file.write(extracted_text)
