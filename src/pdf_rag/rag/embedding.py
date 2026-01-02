from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def embedding_data(
    extracted_text, filename, chunk_size, chunk_overlap, model, embedding_db
):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator='\n\n',  # split by two line breaks
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(extracted_text)

    embeddings = OpenAIEmbeddings(model=model)

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"source": filename}] * len(chunks),
        persist_directory=embedding_db,
    )

    return db


def search_similar_documents(query, k, model, embedding_db):
    # Embedding user input
    embeddings = OpenAIEmbeddings(model=model)
    db = Chroma(persist_directory=embedding_db, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(query, k=k)
    return results


def main():
    # pass fo now
    pass


if __name__ == "__main__":
    main()
