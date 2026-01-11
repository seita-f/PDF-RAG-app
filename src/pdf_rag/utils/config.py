from dataclasses import dataclass
from pathlib import Path

import yaml

with open("config/config.yaml") as file:
    _raw_config = yaml.safe_load(file)


@dataclass(frozen=True)
class Settings:
    # PDF
    PDF_DOC_DIR = _raw_config["pdf"]["doc_dir"]
    PDF_TEXTS_DIR = _raw_config["pdf"]["text_dir"]

    # EMBEDDING
    EMBEDDING_MODEL = _raw_config["embedding"]["model"]
    EMBEDDING_OVERLAP = _raw_config["embedding"]["overlap"]
    EMBEDDING_CHUNK_SIZE = _raw_config["embedding"]["chunk_size"]
    EMBEDDING_DB_DIR = _raw_config["embedding"]["db_dir"]

    # RETRIEVER
    RET_TYPE = _raw_config["retriever"]["type"]
    RET_VECTOR_SEARCH_TYPE = _raw_config["retriever"]["vector"]["search_type"]
    RET_VECTOR_K = _raw_config["retriever"]["vector"]["k"]
    RET_BM25_K = _raw_config["retriever"]["bm25"]["k"]
    RET_BM25_W = _raw_config["retriever"]["hybrid"]["bm25_weight"]
    RET_VECTOR_W = _raw_config["retriever"]["hybrid"]["vector_weight"]

    # LLM
    LLM_MODEL = _raw_config["llm"]["model"]
    LLM_TEMPERATURE = _raw_config["llm"]["temperature"]
    LLM_MAX_TOKEN = _raw_config["llm"]["max_tokens"]
    LLM_TOP_K = _raw_config["llm"]["k"]
    LLM_PROMPT = _raw_config["llm"]["prompt"]

    # EVAL
    EVAL_LLM_MODEL = _raw_config["eval"]["llm"]["model"]
    EVAL_LLM_TEMPERATURE = _raw_config["eval"]["llm"]["temperature"]
    EVAL_LOG_DIR = Path(_raw_config["eval"]["log_dir"])
    EVAL_GOLDEN_DATASET = Path(_raw_config["eval"]["golden_dataset"])
    EVAL_RESULT_DIR = Path(_raw_config["eval"]["result_dir"])


settings = Settings()
