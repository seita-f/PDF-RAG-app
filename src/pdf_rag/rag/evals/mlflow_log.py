from pathlib import Path

import mlflow.langchain

import mlflow
from src.pdf_rag.utils.config import settings


def init_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("PDF-RAG-Eval")
    mlflow.langchain.autolog()


def log_config_artifact():
    try:
        candidate = Path("config/config.yaml")
        mlflow.log_artifact(str(candidate))
    except Exception:
        print("Failed loading config file")


def log_params_for_run(mode: str):
    mlflow.log_params(
        {
            "eval_mode": mode,
            # Embedding
            "emb_model": settings.EMBEDDING_MODEL,
            "emb_chunk_size": settings.EMBEDDING_CHUNK_SIZE,
            "emb_overlap": settings.EMBEDDING_OVERLAP,
            # Retriever
            "ret_type": settings.RET_TYPE,
            "ret_vector_type": settings.RET_VECTOR_SEARCH_TYPE,
            "ret_vector_k": settings.RET_VECTOR_K,
            "ret_bm25_k": settings.RET_BM25_K,
            "ret_hybrid_bm25_w": settings.RET_BM25_W,
            "ret_hybrid_vector_w": settings.RET_VECTOR_W,
            # LLM
            "llm_model": settings.LLM_MODEL,
            "llm_temp": settings.LLM_TEMPERATURE,
            "llm_max_tokens": settings.LLM_MAX_TOKEN,
            "llm_top_k": settings.LLM_TOP_K,
            # Eval LLM
            "eval_llm_model": settings.EVAL_LLM_MODEL,
        }
    )
    mlflow.set_tag("release.version", "1.0.0")


def log_metrics_and_details(mode: str, results, details_df):
    # ovarall metrics score
    scores = results.to_pandas().mean(numeric_only=True).to_dict()
    mlflow.log_metrics(scores)

    # save in artifact
    out_dir = Path("mlflow_data/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"eval_details_{mode}.csv"
    details_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))
