from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def embedding_text(extracted_text, filename):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(extracted_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ChromaDB
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"source": filename}] * len(chunks),
        persist_directory="data/chroma_db",
    )

    return db
