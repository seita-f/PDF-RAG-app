from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def embedding_data(extracted_text, filename):
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


def search_similar_documents(query: str, k: int = 3):
    # Embedding user input
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(query, k=k)
    return results
