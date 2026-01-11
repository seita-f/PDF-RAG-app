# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from src.pdf_rag.rag.embedding import _get_retriever
# from src.pdf_rag.utils.config import settings


# def create_rag_chain():

#     embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
#     db = Chroma(
#         persist_directory=settings.EMBEDDING_DB_DIR, embedding_function=embeddings
#     )
#     retriever = _get_retriever(db)

#     llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
#     prompt = ChatPromptTemplate.from_template(settings.LLM_PROMPT)

#     rag_chain = (
#         {
#             "context": retriever
#             | (lambda docs: "\n\n".join(d.page_content for d in docs)),
#             "question": RunnablePassthrough(),
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return rag_chain
