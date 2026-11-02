# GLiNER service

NVIDIA GLiNER-PII behind one endpoint, for the comparison panel on the demo page.
It runs apart from the gateway because the model needs far more memory. It keeps
nothing, and nothing it produces reaches the LLM.

    POST /detect  {"text": "...", "labels": ["person"], "threshold": 0.5}
    Authorization: Bearer <GLINER_TOKEN>

## Deploy to Google Cloud Run

From this folder, in Cloud Shell or with `gcloud` installed, with `GLINER_TOKEN` stored
in Secret Manager as `gliner-token`:

    gcloud run deploy gliner-pii --source . --region us-central1 --memory 8Gi --cpu 2 --max-instances 1 --allow-unauthenticated --update-secrets=GLINER_TOKEN=gliner-token:1

Then set `GLINER_URL` on the gateway to the service URL it prints, and give the gateway
the same `GLINER_TOKEN`. The first message after an idle spell is slow, because the
model has to load.
