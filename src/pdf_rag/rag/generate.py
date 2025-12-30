from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_answer(query: str, relevant_docs, model, temperature, prompt_template):
    llm = ChatOpenAI(model=model, temperature=temperature)

    # prompt = ChatPromptTemplate.from_template("""
    # You are a professional assistant. Answer the user's question using ONLY the following context provided.
    # If the answer is not contained within the context, respond with "I am sorry, but I could not find the information in the provided documents."

    # Context:
    # {context}

    # Question:
    # {question}

    # Answer:
    # """)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    context_text = "\n\n".join([doc.page_content for doc, score in relevant_docs])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context_text, "question": query})


def main():
    # pass fo now
    pass


if __name__ == "__main__":
    main()
