# def mlflow_logging(mode, results):
# import argparse
# import os
# from pathlib import Path
# import mlflow

# import pandas as pd
# import yaml
# from datasets import Dataset
# from langchain_core.tracers.context import LangChainTracer
# from langchain_openai import OpenAIEmbeddings
# from openai import AsyncOpenAI
# from ragas import evaluate
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.llms import llm_factory
# from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
# from ragas.run_config import RunConfig

# from src.pdf_rag.rag.embedding import search_similar_documents
# from src.pdf_rag.rag.generate import generate_answer

# def mlflow_logging(mode, results):
#     mlflow.set_tracking_uri("http://mlflow:5000")
#     mlflow.set_experiment("PDF-RAG-Eval")

#     with mlflow.start_run():
#         # Parameter
#         mlflow.log_params(
#             {
#                 "eval_mode": mode,
#                 # Embedding
#                 "emb_model": EMBEDDING_MODEL,
#                 "emb_chunk_size": EMBEDDING_CHUNK_SIZE,
#                 "emb_overlap": EMBEDDING_OVERLAP,
#                 # Retriever
#                 "ret_type": RETRIEVER_TYPE,
#                 "ret_vector_type": RETRIEVER_VECTOR,
#                 "ret_vector_k": RETRIEVER_VECTOR_K,
#                 "ret_bm25_k": RETRIEVER_BM25_K,
#                 "ret_hybrid_bm25_w": RETRIEVER_BM25_W,
#                 "ret_hybrid_vector_w": RETRIEVER_VECTOR_W,
#                 # LLM
#                 "llm_model": LLM_MODEL,
#                 "llm_temp": LLM_TEMPERATURE,
#                 "llm_max_tokens": LLM_MAX_TOKEN,
#                 "llm_top_k": LLM_TOP_K,
#                 # Eval
#                 "eval_llm_model": EVAL_LLM_MODEL,
#             }
#         )

#         # log prompt
#         # mlflow.log_param("llm_prompt", LLM_PROMPT[:250] + "...")

#         # metrics
#         scores = results.to_pandas().mean(numeric_only=True).to_dict()
#         for metric_name, score in scores.items():
#             mlflow.log_metric(metric_name, score)
#         mlflow.log_artifact(EVAL_LOGS_DIR / mode)

#     print("MLFlow logging is done. Check results at http://localhost:5000")
