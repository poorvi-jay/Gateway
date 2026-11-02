FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY policy.yaml .
COPY app app

# Cloud Run sets PORT.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
