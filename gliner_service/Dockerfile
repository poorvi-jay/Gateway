FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# In the image, so waking from idle doesn't download 1.8 GB again.
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('nvidia/gliner-pii')"
ENV HF_HUB_OFFLINE=1

COPY app.py .
# Cloud Run sets PORT.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
