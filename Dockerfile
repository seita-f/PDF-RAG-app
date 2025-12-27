# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.2-python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app 

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

# Streamlit
EXPOSE 8501

CMD ["uv", "run", "python", "-m", "streamlit", "run", "src/pdf_rag/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
