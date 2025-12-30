import pandas as pd
import yaml
from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)

from src.pdf_rag.rag.embedding import search_similar_documents
from src.pdf_rag.rag.generate import generate_answer

async_client = AsyncOpenAI()

# load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file.read())


# EMBEDDING
EMBEDDING_MODEL = config["embedding"]["model"]
EMBEDDING_OVERLAP = config["embedding"]["overlap"]
EMBEDDING_CHUNK_SIZE = config["embedding"]["chunk_size"]
EMBEDDING_DB_DIR = config["embedding"]["db_dir"]

# LLM
LLM_MODEL = config["llm"]["model"]
LLM_TEMPERATURE = config["llm"]["temperature"]
LLM_TOP_K = config["llm"]["k"]
LLM_PROMPT = config["llm"]["prompt"]

# EVAL
EVAL_LLM_MODEL = config["eval"]["llm"]["model"]
EVAL_LLM_TEMPERATURE = config["eval"]["llm"]["temperature"]
EVAL_LOGS_DIR = config["eval"]["log_dir"]

async_client = AsyncOpenAI()

evaluator_llm = llm_factory(model=EVAL_LLM_MODEL, client=async_client)
evaluator_embeddings = embedding_factory(model=EMBEDDING_MODEL, client=async_client)

metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings),
]


def run_evaluation():
    df_golden = pd.read_json(config["eval"]["golden_dataset"])

    questions = df_golden["question"].tolist()
    ground_truths = df_golden["ground_truth"].tolist()

    answers = []
    contexts = []

    print(f"Generating answers for {len(questions)} queries...")
    for q in questions:
        relevant_docs = search_similar_documents(
            q, LLM_TOP_K, EMBEDDING_MODEL, EMBEDDING_DB_DIR
        )
        ans = generate_answer(q, relevant_docs, LLM_MODEL, LLM_TEMPERATURE, LLM_PROMPT)

        answers.append(ans)
        contexts.append([doc.page_content for doc, _ in relevant_docs])

    # ragas dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    print("Evaluating metrics...")
    results = evaluate(dataset=dataset, metrics=metrics)

    print("\n--- Evaluation Results ---")
    print(results)

    return results


"""
To-do: csv -> can not split columns by ','
"""
# def save_evaluation_results(results):
#     os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

#     # filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_name = f"eval_{timestamp}.csv"
#     save_path = os.path.join(EVAL_LOGS_DIR, file_name)

#     df = results.to_pandas()

#     # params to save
#     df["run_at"] = timestamp
#     df["llm_model"] = LLM_MODEL
#     df["top_k"] = LLM_TOP_K
#     df["embed_model"] = EMBEDDING_MODEL
#     df["chunk_size"] = EMBEDDING_CHUNK_SIZE
#     df["chunk_overlap"] = EMBEDDING_OVERLAP
#     df["eval_model"] = EVAL_LLM_MODEL

#     df.to_csv(save_path, index=False)
#     print(f"\nResults are saved in {save_path}")


if __name__ == "__main__":
    results = run_evaluation()
    # save_evaluation_results(
    #     results,
    # )
