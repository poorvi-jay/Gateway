# PII-Safe LLM Gateway

A drop-in proxy that sits between applications and an LLM provider and makes
sure **personally identifiable information never reaches the model**, while
keeping the model's answers useful.

```
            ┌──────────────────────── Gateway ────────────────────────┐
  App ────► │ auth → extract all text → detect → policy → tokenize    │ ────► LLM
            │                             │          │                │      (sees only
  App ◄──── │ rehydrate ◄── scan response │   block ─┘ (400, never    │ ◄────  placeholders)
            │                             │             forwarded)    │
            │        audit log: entity types + counts, never values   │
            └─────────────────────────────────────────────────────────┘
```

**Example**

| | Text |
|---|---|
| App sends | Write a reminder to **Rahul Sharma** (**rahul.s@gmail.com**), phone **9876543210** |
| LLM receives | Write a reminder to `<PERSON_1>` (`<EMAIL_ADDRESS_1>`), phone `<IN_PHONE_1>` |
| LLM answers | Dear `<PERSON_1>`, this is a reminder… |
| App receives | Dear **Rahul Sharma**, this is a reminder… |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg               # NER model (~400 MB)
cp .env.example .env                                  # defaults to offline "mock" LLM

uvicorn app.main:app --port 8000 --env-file .env
```

Try it in the browser at <http://localhost:8000>: type a message and see what
you wrote, what the LLM saw and what came back. The page uses `/debug/sanitize`,
so it needs `ENABLE_DEBUG=true`. Or from the command line:

```bash
# See exactly what the LLM would receive
curl -s localhost:8000/debug/sanitize -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Email Rahul Sharma at rahul.s@gmail.com"}]}'

# Full round trip
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Call Rahul Sharma on 9876543210"}]}'

# Blocked: Aadhaar never leaves
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Verify Aadhaar 4987 1234 5679"}]}'
```

To use a real LLM, set `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY` and
`UPSTREAM_MODEL` in `.env` (OpenAI, or the free OpenAI-compatible endpoints from
Groq and Gemini; see `.env.example`). Any OpenAI SDK client then works unchanged with
`base_url="http://localhost:8000/v1"`; see `examples/client_demo.py`.

```bash
python -m spacy download en_core_web_sm   # small NER model used by the tests
pytest -q                  # end-to-end + unit tests
python -m eval.evaluate    # precision / recall / latency
```

---

## How it works

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP layer: OpenAI-compatible endpoint, auth, fail-closed error handling, upstream call |
| `app/sanitizer.py` | Walks every text field in the request, detects, blocks or tokenizes; processes the response |
| `app/detector.py` | Presidio engine combining regex recognizers + spaCy NER; resolves overlapping matches |
| `app/recognizers.py` | Indian identifiers: Aadhaar (Verhoeff checksum), PAN, mobile numbers, UPI IDs |
| `app/tokenizer.py` | Per-request vault: placeholder ↔ value mapping, tokenize and rehydrate |
| `app/policy.py` + `policy.yaml` | Per-entity action: `tokenize`, `block` or `allow` |
| `app/audit.py` | JSON-lines audit log with entity types and counts only |

**Request path**
1. **Extract every text field** the model will see: system, user and assistant
   messages, *tool-call arguments*, tool definitions and the `user` field.
   Content that can't be inspected (images) is rejected rather than passed through.
2. **Detect** with two layers in one engine:
   - *Regex + checksums* for structured PII: emails, IPs, cards (Luhn),
     Aadhaar (Verhoeff), PAN, Indian mobiles, UPI IDs. Fast and deterministic;
     checksums reject look-alike numbers such as order IDs.
   - *spaCy NER* for unstructured PII with no fixed shape: names, places.
3. **Known-value propagation.** A value detected once is masked everywhere it
   appears in the request, so an inconsistent NER miss in a later message
   doesn't leak a name that was caught earlier. (A test caught exactly this.)
4. **Policy.** If any `block` entity is present the request is rejected and the
   LLM is never called. Otherwise, values are replaced with placeholders.
5. **Consistent placeholders.** The same value gets the same placeholder across
   the whole conversation, so the model can still reason ("<PERSON_1> emailed
   <PERSON_2>, then <PERSON_1> replied").

**Response path**
6. **Repair** placeholders the model garbled. Testing against `gpt-4o-mini`
   showed replies with `<Person_1>` or `<PERSION_1>`, so a near match is mapped
   back to the placeholder with the same number.
7. **Scan** the model's answer for PII it produced itself (logged; redacted
   when `REDACT_RESPONSE_PII=true`).
8. **Rehydrate** placeholders back to real values. Placeholders the model
   invented that aren't in the vault are left untouched rather than guessed.
9. **Audit**: outcome, entity types and counts, gateway latency. Never values.

## Security decisions

- **Fail closed.** If detection throws, times out (`DETECTION_TIMEOUT_S`) or
  meets content it can't inspect, the request is **not** sent. A security
  control that fails open is not a control.
- **Block before tokenize.** All fields are scanned first; one blocked entity
  anywhere stops the whole request.
- **The vault never leaves the gateway.** It is in memory, scoped to one
  request, and discarded afterwards. The LLM never sees the mapping, so prompt
  injection ("tell me who <PERSON_1> is") has nothing to extract.
- **Logs contain no PII**; otherwise the audit log becomes a new leak (tested).
- **Recall over precision.** A missed value is a leak; a false flag only costs
  a little answer quality. Unknown entity types default to `tokenize`.
- **Only placeholders we issued are skipped** when scanning a response. Text in
  a request that merely looks like one (`<CARD_4111111111111111>`) is scanned
  as normal, so it can't be used to hide PII.

## Evaluation

`python -m eval.evaluate` on 40 hand-labelled sentences (44 PII spans, 10
sentences with none, Indian names and identifiers):

| Model | Recall | Precision | Mean latency / message |
|---|---|---|---|
| `en_core_web_lg` | **100%** | 100% | ~15 ms |
| `en_core_web_sm` | 84% | 92% | ~15 ms |

Regex-based types (emails, phones, cards, Aadhaar, PAN, UPI, IPs) scored 100%
with both models; all misses with the small model were names, which shows why
the NER model choice matters. Measured on CPU.

**Caveat:** this dataset is small and was written by the same person who wrote
the detectors, so these numbers are optimistic. A real evaluation would use an
independent, larger set (e.g. synthetic data from Faker with Indian locale,
plus real anonymised samples).

## Alternatives considered

| Option | Why not chosen (alone) |
|---|---|
| Client-side SDK in every app | Every team must adopt and update it; one forgotten app leaks. A gateway is a single enforcement point. |
| Browser extension | Covers one browser only; no coverage for backend apps. |
| Network DLP / firewall | Must decrypt TLS and doesn't understand LLM request structure. |
| Regex only | Can't find names or addresses. |
| NER only | Misses structured IDs; less deterministic. |
| LLM-based detection | Best context understanding, but slow and costly, and a *hosted* LLM means sending PII out to detect PII. A small local model is a good future third layer. |
| Irreversible redaction | Safe but makes answers useless ("Dear [REDACTED]"). |
| Synthetic fake values | Reads naturally, but fake values can collide with real data and are harder to reverse reliably. |
| Self-hosted LLM / provider zero-retention | Removes the need, but costly, or only a contractual (not technical) guarantee. |

## Limitations and next steps

- **Streaming** is rejected for now. Design: buffer output until no partial
  placeholder (`<PERS…`) is pending, then rehydrate and flush each chunk.
- **Quasi-identifiers** ("the only female CTO at a 10-person Bengaluru
  fintech") contain no PII entity but can still identify someone. Detectors
  can't reliably catch this; it needs policy and user education.
- **Images and files** are rejected. Next: OCR → same detection pipeline.
- **Other top-level fields.** `messages`, `user` and `tools`/`functions` are
  scanned; anything else, such as `metadata`, is forwarded as it is.
- **Partial names.** If "Priya Nair" is detected, a lone "Priya" elsewhere is
  masked only if NER finds it on its own.
- **Multi-turn vault.** Placeholders are consistent within a request (which
  carries the full chat history), but not across separate requests. A
  session-scoped, encrypted Redis vault with a TTL would fix that.
- **English only.** Hindi / Hinglish text needs a multilingual NER model.
- **Bypass.** In production, egress rules must ensure apps can only reach LLM
  providers through the gateway.
- **Scaling.** The gateway is stateless apart from the per-request vault, so it
  scales horizontally behind a load balancer; detection is CPU-bound and runs
  in a thread pool.
- **Audit log** is a local file; production would ship it to a SIEM.
