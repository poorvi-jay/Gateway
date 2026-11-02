# GLiNER service

NVIDIA GLiNER-PII behind one endpoint, for the comparison panel on the demo page.
The gateway's free Render plan has too little memory for the model, so it runs
here. It keeps nothing, and nothing it produces reaches the LLM.

    POST /detect  {"text": "...", "labels": ["person"], "threshold": 0.5}

## Deploy to Google Cloud Run

From this folder, in Cloud Shell or with `gcloud` installed:

    gcloud run deploy gliner-pii --source . --region us-central1 --memory 8Gi --cpu 2 --max-instances 1 --allow-unauthenticated

Then set `GLINER_URL` on the gateway to the service URL it prints. The first
message after an idle spell is slow, because the model has to load.
