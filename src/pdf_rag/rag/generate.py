import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.pdf_rag.utils.config import settings


@mlflow.trace(name="RAG_Generation_Step")
def generate_answer(
    query: str,
    relevant_docs,
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    prompt_template=settings.LLM_PROMPT,
    max_tokens=settings.LLM_MAX_TOKEN,
):
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    context_text = "\n\n".join([doc.page_content for doc, _ in relevant_docs])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context_text, "question": query})


def main():
    # pass fo now
    pass


if __name__ == "__main__":
    main()
