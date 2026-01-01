import os

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def extract_text_from_pdf(pdf_file):
    try:
        laparams = LAParams(
            all_texts=True
        )  # all_texts: include text in graph, table etc

        text = extract_text(pdf_file, laparams=laparams)

        # Normalize text data

        # [to-do] May add the funciton to deal with images in pdf

        # text = '\n'.join(text)
        return text

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def debug_save_pdf_in_text(uploaded_file, extracted_text, text_dir):
    save_path = text_dir
    os.makedirs(save_path, exist_ok=True)

    txt_filename = os.path.join(save_path, uploaded_file.name.replace(".pdf", ".txt"))

    with open(txt_filename, "w", encoding="utf-8") as file:
        file.write(extracted_text)


def save_pdf(uploaded_file, doc_dir):
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
