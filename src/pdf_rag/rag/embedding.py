from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_rag.utils.config import settings


def embedding_data(
    documents,
    chunk_size=settings.EMBEDDING_CHUNK_SIZE,
    chunk_overlap=settings.EMBEDDING_OVERLAP,
    model=settings.EMBEDDING_MODEL,
    embedding_db=settings.EMBEDDING_DB_DIR,
):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator='\n\n',  # split by two line breaks
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model=model)

    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=embedding_db,
    )

    #  DEBUG
    _debug_visualize_chunk(split_docs, 5)

    return db


def _debug_visualize_chunk(docs, size):
    print(f"Total chunks: {len(docs)}")
    for i in range(min(len(docs), size)):
        print(f"--- Chunk {i} ---")
        print(docs[i].page_content)
        print(f"Metadata: {docs[i].metadata}")
        print("-" * 20)


def _get_retriever(db, k=settings.RET_VECTOR_K):
    ret_type = settings.RET_TYPE

    # Vector
    vector_ret = db.as_retriever(
        search_type=settings.RET_VECTOR_SEARCH_TYPE, search_kwargs={"k": k}
    )

    if ret_type == "vector":
        return vector_ret

    all_content = db.get()
    all_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(all_content["documents"], all_content["metadatas"])
    ]

    # BM25 Retriever
    bm25_ret = BM25Retriever.from_documents(all_docs)
    bm25_ret.k = settings.RET_BM25_K

    if ret_type == "bm25":
        return bm25_ret

    if ret_type == "hybrid":
        return EnsembleRetriever(
            retrievers=[bm25_ret, vector_ret],
            weights=[
                settings.RET_BM25_W,
                settings.RET_VECTOR_W,
            ],
        )

    return vector_ret


def search_similar_documents(
    query,
    k=settings.RET_VECTOR_K,
    model=settings.EMBEDDING_MODEL,
    embedding_db=settings.EMBEDDING_DB_DIR,
):
    embeddings = OpenAIEmbeddings(model=model)
    db = Chroma(persist_directory=embedding_db, embedding_function=embeddings)

    retriever = _get_retriever(db, k)
    docs = retriever.invoke(query)

    return [(doc, 1.0) for doc in docs[:k]]


def main():
    # pass fo now
    pass


if __name__ == "__main__":
    main()
