import argparse
import os
from datetime import datetime

import pandas as pd
from datasets import Dataset
from langchain_core.tracers.context import LangChainTracer
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

import mlflow
from src.pdf_rag.rag.embedding import search_similar_documents
from src.pdf_rag.rag.evals.mlflow_log import (
    init_mlflow,
    log_config_artifact,
    log_metrics_and_details,
    log_params_for_run,
)
from src.pdf_rag.rag.generate import generate_answer
from src.pdf_rag.utils.config import settings

# RAGAS evaluator
async_client = AsyncOpenAI()
evaluator_llm = llm_factory(model=settings.EVAL_LLM_MODEL, client=async_client)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
)


def run_evaluation(mode: str):
    df_golden = pd.read_json(settings.EVAL_GOLDEN_DATASET)

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

    for i, query in enumerate(questions):
        with mlflow.start_span(name=f"rag_query_{i + 1}") as root_span:
            root_span.set_inputs({"question": query, "mode": mode})

            relevant_docs = search_similar_documents(query)

            if mode == "retrieval":
                ans = ground_truths[i]
            else:
                ans = generate_answer(query, relevant_docs)

            answers.append(ans)
            contexts.append([doc.page_content for doc, _ in relevant_docs])

            root_span.set_outputs(
                {
                    "answer": ans,
                    "num_contexts": len(relevant_docs),
                    "contexts_preview": [
                        doc.page_content[:200] for doc, _ in relevant_docs[:5]
                    ],
                }
            )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "reference_contexts": [
                [expected_contexts[i]] for i in range(len(questions))
            ],
        }
    )

    # tracer for langsmith (learning purpose)
    tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

    mlflow.langchain.autolog(disable=True)
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=1, timeout=60),
        callbacks=[tracer],
    )
    mlflow.langchain.autolog(disable=False)

    ragas_df = results.to_pandas()
    details_df = pd.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts,
        }
    )
    details_df = pd.concat([details_df, ragas_df], axis=1)

    return results, details_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["retrieval", "generator"], default="retrieval"
    )
    args = parser.parse_args()

    init_mlflow()

    run_name = f"RAG-{args.mode}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_name):
        log_params_for_run(args.mode)
        log_config_artifact()

        results, details_df = run_evaluation(args.mode)

        log_metrics_and_details(args.mode, results, details_df)

    print("Done. Check MLflow UI: http://0.0.0.0:5000/")


if __name__ == "__main__":
    main()
