import os
import re
import unicodedata

import fitz

# PDFMiner
# def extract_text_from_pdf(pdf_file):
#     try:
#         laparams = LAParams(
#             all_texts=True,
#             char_margin=3.0,
#             line_margin=0.2,
#             word_margin=0.1,
#             boxes_flow=0.5,
#         )  # all_texts: include text in graph, table etc

#         text = extract_text(pdf_file, laparams=laparams)
#         text = text_cleaning(text)

#         return text

#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return None


# PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        else:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

        full_text = []

        for page in doc:
            # block mode
            blocks = page.get_text("blocks")

            # sort block（up to down, left to right)
            blocks.sort(key=lambda b: (b[1], b[0]))

            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    full_text.append(block_text)

        # add two line break between blocks
        text = "\n\n".join(full_text)

        # text cleaning
        text = text_cleaning(text)

        return text.strip()

    except Exception as e:
        print(f"Error occurred in PyMuPDF: {e}")
        return None


def text_cleaning(text):
    # "assess-\nment" -> "assessment"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = unicodedata.normalize("NFKC", text)

    return text


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
