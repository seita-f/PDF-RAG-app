[![Python Lint](https://github.com/seita-f/PDF-RAG-app/actions/workflows/lint.yml/badge.svg)](https://github.com/seita-f/PDF-RAG-app/actions/workflows/lint.yml)
# PDF RAG App
![Image](https://github.com/user-attachments/assets/8b4a7596-a7dd-4bea-9648-5e82e5ccf5ec)
![Image](https://github.com/user-attachments/assets/1d2c5aa0-521f-4027-8a11-3d539ee04d99)

# Build with
<p style="display: inline">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- Libraries -->
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white&style=for-the-badge">
  <img src="https://img.shields.io/badge/-LangChain-1C3C3C.svg?logo=langchain&logoColor=white&style=for-the-badge">
  <!-- <img src="https://img.shields.io/badge/-Hugging%20Face-FFD21E.svg?logo=huggingface&logoColor=black&style=for-the-badge"> -->
  <!-- インフラ一覧 -->
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
  <img src="https://img.shields.io/badge/-githubactions-FFFFFF.svg?logo=github-actions&style=for-the-badge">
</p>

# How to Start
Create a `.env` file in the root directory and add your OpenAI API key and LangSmith API
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx

LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=xxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECCT=<any name>
```
Build and run the application using Docker
```
docker compose build
docker compose up
```
Evaluate RAG
```
# suppose container is up
docker compose exec app bash

# retrieval evaluation
uv run src/pdf_rag/rag/evals/evals.py --mode retrieval

# generator evaluation
uv run src/pdf_rag/rag/evals/evals.py --mode generator
```

Adjust the parameters in [`config/config.yaml`](https://github.com/seita-f/PDF-RAG-app/blob/main/config/config.yaml). <br><br>


# Evaluate
Briefly evaluated RAG system based on the **RAGAS** results. <br>
More Details can be viewed in **LangSmith** Dashboard. <br>
### 1. Configuration
[golden dataset](https://github.com/seita-f/PDF-RAG-app/blob/main/config/eval_golden_dataset.json)
| Configuration Item | Value |
| --- | --- |
| LLM Model | gpt-4 |
| Temperature | 0 |

### 2 Ex: Retrieval Evaluation
| Embedding Model | Chunk Size | Overlap | Context Precision | Context Recall |
| --- | --- | --- | --- | --- | 
| text-embedding-3-small | 1024 | 200 | 0.8000 | 0.7333 |

### 3. Ex: Generator Evaluation

| Embedding Model | Chunk Size | Overlap |
| :--- | :--- | :--- |
| text-embedding-3-small | 1024 | 200 |

| LLM Model | Temperature | Top K | Max Tokens | Faithfulness | Answer Relevancy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gpt-4o-mini | 0 | 5 | 1024 | 0.9600 | 0.7738 |

LangSmith Dashboard
<img
  src="https://github.com/user-attachments/assets/2dd3954b-8a8e-4759-9cab-623462fa6b1a"
  alt="langsmith"
  width="900"
  style="border: 2px solid black;"
/>

