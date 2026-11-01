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
python -m eval.evaluate    # precision / recall / latency on the hand-written set
python -m eval.evaluate eval/synthetic_test.jsonl   # on the held-back synthetic set
```

---

## How it works

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP layer: OpenAI-compatible endpoint, auth, fail-closed error handling, upstream call |
| `app/sanitizer.py` | Walks every text field in the request, detects, blocks or tokenizes; processes the response |
| `app/detector.py` | Presidio engine combining regex recognizers + spaCy NER; resolves overlapping matches |
| `app/recognizers.py` | Indian identifiers (Aadhaar with Verhoeff checksum, PAN, passport, voter ID, driving licence, mobile numbers, UPI IDs, IFSC), bank accounts, dates of birth, PIN codes, API keys and passwords |
| `app/tokenizer.py` | Per-request vault: placeholder ↔ value mapping, tokenize and rehydrate |
| `app/gliner_compare.py` | Optional local comparison with NVIDIA GLiNER-PII on the demo page. Never changes what reaches the LLM. Install `requirements-gliner.txt`, set `ENABLE_GLINER=true` |
| `app/policy.py` + `policy.yaml` | Per-entity action: `tokenize`, `redact`, `block` or `allow` |
| `app/audit.py` | JSON-lines audit log with entity types and counts only |
| `eval/synthetic.py` + `eval/evaluate.py` | Seeded Faker test data; precision, recall and latency scoring |

**Request path**
1. **Extract every text field** the model will see: system, user and assistant
   messages, *tool-call arguments*, tool definitions and the `user` field.
   Content that can't be inspected (images) is rejected rather than passed through.
2. **Detect** with two layers in one engine:
   - *Regex + checksums* for structured PII: emails, IPs, cards (Luhn),
     Aadhaar (Verhoeff), PAN, Indian mobiles, UPI IDs, vendor API keys. Fast and
     deterministic; checksums reject look-alike numbers such as order IDs.
   - *Regex + context words* for shapes too common to trust alone. A passport,
     voter ID, bank account, date of birth or PIN code only counts next to a
     word like "passport" or "DOB", so order numbers and ordinary dates pass.
     Passwords have no shape, so the value after "password:" or "OTP is" is masked.
   - *spaCy NER* for unstructured PII with no fixed shape: names, places. Two
     capitalized words in a row and a list of major Indian cities back it up,
     because NER misses many Indian names and places.
3. **Known-value propagation.** A value detected once is masked everywhere it
   appears in the request, so an inconsistent NER miss in a later message
   doesn't leak a name that was caught earlier. (A test caught exactly this.)
   Words of a full name count too: after "Priya Nair", a lone "Priya" or "Nair"
   becomes the same `<PERSON_1>`, unless two people share that word.
4. **Policy.** If any `block` entity is present the request is rejected and the
   LLM is never called. Otherwise, values are replaced with placeholders.
   Any type can be switched in `policy.yaml`, for example
   `IN_AADHAAR: tokenize` (masked, restored in the answer) or `IN_AADHAAR: redact`
   (`[REDACTED_IN_AADHAAR]`, never restored); restart to apply. Blocking is the
   default for IDs because it also holds back anything else in that message
   that detection missed.
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

Three sets, scored with `python -m eval.evaluate [set]` on CPU at about 20 ms per
message. *Masked* is stricter than recall: nothing of the value may be left showing.

| Set | Model | Recall | Masked | Precision |
|---|---|---|---|---|
| **Test**: 200 synthetic sentences, 224 spans | `en_core_web_lg` | **93.8%** | **92.4%** | 90.9% |
| | `en_core_web_sm` | 91.5% | 91.1% | 96.7% |
| Dev: 200 synthetic sentences, 256 spans | `en_core_web_lg` | 94.1% | 93.8% | 98.4% |
| | `en_core_web_sm` | 92.2% | 91.8% | 96.3% |
| Hand-written: 52 sentences, 55 spans | `en_core_web_lg` | 100% | 100% | 100% |
| | `en_core_web_sm` | 92.7% | 90.9% | 92.7% |
| Test, comparison only | NVIDIA GLiNER-PII, local CPU | 98.2% | 98.2% | 81.2% |
| Dev, comparison only | NVIDIA GLiNER-PII, local CPU | 98.0% | 96.9% | 88.8% |
| Hand-written, comparison only | NVIDIA GLiNER-PII, local CPU | 98.2% | 98.2% | 87.1% |

**Trust the test row.** `eval/synthetic.py` uses Faker's `en_IN` locale with a
fixed seed. Misses on dev were looked at and fixed; test has its own seed and
its own sentence templates, and was scored once after the code was frozen. Nothing
was changed because of it. The hand-written set scored 100% only because the same
person wrote it and the detectors: the first dev run found 69% recall on names
and 67% on places, which that set had hidden.

On the test set with `en_core_web_lg`:
- **All 17 other types scored 100%**, including every type that blocks a request,
  with no request falsely blocked. Each has only 8–16 samples.
- **Names: 84%** (47/56). Misses are lone first names ("Xavier told Viraj") and
  lowercase names. Rules would mask too much here; this needs a better model.
- **Landlines: 38%** (3/8). They are only found next to a word like "landline" or
  "phone", and "Reach the clinic on 05498956489" has none.
- **One rule mistake:** "Date" is on the list of words that aren't names, so the
  surname in "Prisha Date" stayed visible.
- **Precision is mostly one error:** spaCy tags "OTP" as a name. It costs a
  masked word, not a leak.

**NVIDIA GLiNER-PII on the same data** (`--gliner`, label list fixed before scoring,
nothing tuned): it caught all 56 names and all 8 landlines, the gateway's two weak
spots, and missed 4 of 224 values (2 mobiles, 2 OTPs). Three costs keep it optional:
- **Speed and size:** about 2.8 s per message on CPU against about 20 ms, 2.2 GB of
  memory and a 1.8 GB model.
- **False flags:** it marks invoice numbers, part numbers and ordinary dates as
  personal data, which would remove details the model needs.
- **Labels can't drive blocking:** it tagged a real Aadhaar number as a card number.
  Recall above counts any label, so these scores say the value was hidden, not that
  the type was right. Block decisions stay with checksums.

**Caveats:** templated sentences are cleaner than real text, and Faker's city
list overlaps the gateway's own list of major cities, so the 100% on places is
generous. Real anonymised samples would be a better test still.

## Alternatives considered

| Option | Why not chosen (alone) |
|---|---|
| Client-side SDK in every app | Every team must adopt and update it; one forgotten app leaks. A gateway is a single enforcement point. |
| Browser extension | Covers one browser only; no coverage for backend apps. |
| Network DLP / firewall | Must decrypt TLS and doesn't understand LLM request structure. |
| Regex only | Can't find names or addresses. |
| NER only | Misses structured IDs; less deterministic. |
| LLM-based detection | Best context understanding, but slow and costly, and a *hosted* LLM means sending PII out to detect PII. A small local model is a good future third layer. |
| Irreversible redaction as the default | Safe but makes answers useless ("Dear [REDACTED]"). Available per type as `redact`. |
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
- **Names.** Lone first names and lowercase names can leak (see Evaluation). A
  lone "priya" in lowercase is masked only if NER finds it on its own: matching
  name words in any case would mask "will" for Will Smith. Two capitalized words
  are masked as a name, so a title like "Quarterly Review" can be masked too.
- **Context words.** A voter ID or landline with no word like "voter" or "phone"
  nearby is not flagged, and a phrase like "the secret is out" is masked as a
  password. Addresses are covered only as place names and PIN codes, and
  pre-2019 licence formats vary by state.
- **A better model.** The GLiNER numbers above suggest a label-conditioned model is
  the likely third layer, for names and landlines. It needs a faster option, such as
  [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
  (Apache 2.0) or a GPU, and it would not replace checksums for block decisions.
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
