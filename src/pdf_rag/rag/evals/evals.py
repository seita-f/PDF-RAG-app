import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset
from langchain_core.tracers.context import LangChainTracer
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.pdf_rag.rag.embedding import search_similar_documents
from src.pdf_rag.rag.generate import generate_answer

# load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file.read())


# EMBEDDING
EMBEDDING_MODEL = config["embedding"]["model"]
EMBEDDING_OVERLAP = config["embedding"]["overlap"]
EMBEDDING_CHUNK_SIZE = config["embedding"]["chunk_size"]
EMBEDDING_DB_DIR = config["embedding"]["db_dir"]

# RETRIEVER
RETRIEVER_TYPE = config["retriever"]["type"]
RETRIEVER_VECTOR = config["retriever"]["vector"]["search_type"]
RETRIEVER_VECTOR_K = config["retriever"]["vector"]["k"]
RETRIEVER_BM25_K = config["retriever"]["bm25"]["k"]
RETRIEVER_BM25_W = config["retriever"]["hybrid"]["bm25_weight"]
RETRIEVER_VECTOR_W = config["retriever"]["hybrid"]["vector_weight"]

# LLM
LLM_MODEL = config["llm"]["model"]
LLM_TEMPERATURE = config["llm"]["temperature"]
LLM_MAX_TOKEN = config["llm"]["max_tokens"]
LLM_TOP_K = config["llm"]["k"]
LLM_PROMPT = config["llm"]["prompt"]

# EVAL
EVAL_LLM_MODEL = config["eval"]["llm"]["model"]
EVAL_LLM_TEMPERATURE = config["eval"]["llm"]["temperature"]
EVAL_LOGS_DIR = Path(config["eval"]["log_dir"])


# Setup LLM
async_client = AsyncOpenAI()
evaluator_llm = llm_factory(model=EVAL_LLM_MODEL, client=async_client)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model=EMBEDDING_MODEL)
)


def save_evaluation_report(results, mode):
    save_path_dir = EVAL_LOGS_DIR / mode
    save_path_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = save_path_dir / f"eval_report_{timestamp}.txt"

    summary_results_str = str(results)

    ret_type = RETRIEVER_TYPE
    retriever_details = f"- Type: {ret_type}\n"

    if ret_type == "vector":
        retriever_details += f"  - Search Type: {RETRIEVER_VECTOR}\n"
        retriever_details += f"  - K: {RETRIEVER_VECTOR_K}"
    elif ret_type == "bm25":
        retriever_details += f"  - K: {RETRIEVER_BM25_K}"
    elif ret_type == "hybrid":
        retriever_details += f"  - BM25 Weight: {RETRIEVER_BM25_W}\n"
        retriever_details += f"  - Vector Weight: {RETRIEVER_VECTOR_W}\n"
        retriever_details += f"  - Vector Search Type: {RETRIEVER_VECTOR}"

    # Report
    summary_text = f"""
==================================================
RAG EVALUATION REPORT
==================================================
Run Time:    {timestamp}
--------------------------------------------------
[METRICS SUMMARY]
{summary_results_str}

[Retriever]
{retriever_details}

[HYPERPARAMETERS - Embedding]
- Embed Model:  {EMBEDDING_MODEL}
- Overlap:      {EMBEDDING_OVERLAP}
- Chunk Size:   {EMBEDDING_CHUNK_SIZE}

[HYPERPARAMETERS - LLM]
- Model:        {LLM_MODEL}
- Temperature   {LLM_TEMPERATURE}
- Max Token     {LLM_MAX_TOKEN}
- Top K         {LLM_TOP_K}
- Prompt        {LLM_PROMPT}
==================================================
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
        # f.write("\n[DETAILED LOGS PER QUERY]\n")
        # f.write(df_details.to_string(index=False))

    print(f"✅ Report successfully saved at: {file_path}")


def run_evaluation(mode, config):
    df_golden = pd.read_json(config["eval"]["golden_dataset"])

    questions = df_golden["question"].tolist()
    ground_truths = df_golden["ground_truth"].tolist()
    expected_contexts = df_golden["context_answer"].tolist()

    answers = []
    contexts = []

    if mode == "retrieval":
        metrics = [
            ContextRecall(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
        ]
        print(f"--- Mode: {mode} (Evaluating Search Quality) ---")
    else:
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(embeddings=evaluator_embeddings, strictness=1),
        ]
        print(f"--- Mode: {mode} (Evaluating Generation Quality) ---")

    print(f"Generating answers for {len(questions)} queries...")
    for i, q in enumerate(questions):
        relevant_docs = search_similar_documents(
            q, LLM_TOP_K, EMBEDDING_MODEL, EMBEDDING_DB_DIR, config
        )
        if mode == "retrieval":
            ans = ground_truths[i]  # Not calling LLM
        else:
            ans = generate_answer(
                q, relevant_docs, LLM_MODEL, LLM_TEMPERATURE, LLM_PROMPT, LLM_MAX_TOKEN
            )

        answers.append(ans)
        contexts.append([doc.page_content for doc, _ in relevant_docs])

    # ragas dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "reference_contexts": [[expected_contexts[i]] for i in range(len(questions))],
    }
    dataset = Dataset.from_dict(data)

    tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

    print("Evaluating metrics...")
    run_config = RunConfig(max_workers=1, timeout=60)
    results = evaluate(
        dataset=dataset, metrics=metrics, run_config=run_config, callbacks=[tracer]
    )

    print(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Script")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["retrieval", "generator"],
        default="retrieval",
        help="Evaluation mode: retrieval or generator",
    )
    args = parser.parse_args()

    results = run_evaluation(args.mode, config)
    save_evaluation_report(results, args.mode)


if __name__ == "__main__":
    main()
