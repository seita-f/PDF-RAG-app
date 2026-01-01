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
Create a `.env` file in the root directory and add your OpenAI API key
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx
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

If necessary, adjust the parameters in [`config/config.yaml`](https://github.com/seita-f/PDF-RAG-app/blob/main/config/config.yaml). <br><br>


<!-- # Evaluate
Evaluated RAG system based on the **RAGAS** results. <br>
### 1. Configuration
| Configuration Item | Value |
| --- | --- |
| LLM Model | gpt-4 |
| Temperature | 0 |
(golden dataset)[https://github.com/seita-f/PDF-RAG-app/blob/main/config/eval_golden_dataset.json]
---

### 2.1 Retrieval Evaluation
| Embedding Model | Chunk Size | Overlap | Context Precision | Context Recall |
| --- | --- | --- | --- | --- | 
| text-embedding-3-small | 1024 | 200 | 0.400 | 0.400 |


### 2.2. Retrieval Evaluation after text mining
To-do: text mining etc

### 3. Generator Evaluation
To-do: Faithfulness, Answer Relevancy -->


