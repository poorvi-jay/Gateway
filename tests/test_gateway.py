"""
End-to-end tests. The upstream LLM is replaced by a fake that records exactly
what it received, so we can assert that no raw PII ever reaches it.

Run:  pytest -q
"""
import os

os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")
os.environ["AUDIT_LOG_PATH"] = os.path.join(os.path.dirname(__file__), "test_audit.log")
os.environ["POLICY_PATH"] = os.path.join(os.path.dirname(__file__), "..", "policy.yaml")

import json  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as gw  # noqa: E402
from app.recognizers import verhoeff_valid  # noqa: E402
from app.tokenizer import Vault, rehydrate  # noqa: E402

VALID_AADHAAR = "2341 2341 2346"  # passes the Verhoeff checksum
URL = "/v1/chat/completions"


@pytest.fixture(scope="module")
def client():
    with TestClient(gw.app) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    """Captures what the LLM receives and replies with a fixed answer."""
    captured = {}

    async def fake(payload):
        captured["payload"] = payload
        user_text = payload["messages"][-1]["content"]
        captured["user_text"] = user_text
        return {"choices": [{"index": 0, "message": {
            "role": "assistant", "content": captured.get("reply", f"Echo: {user_text}")}}]}

    monkeypatch.setattr(gw, "call_upstream", fake)
    return captured


def chat(client, text, **extra):
    return client.post(URL, json={"model": "x", "messages": [{"role": "user", "content": text}], **extra})


# ---------------------------------------------------------------- core flow
def test_pii_never_reaches_llm(client, fake_llm):
    r = chat(client, "Email Rahul Sharma at rahul.s@gmail.com or call 9876543210")
    assert r.status_code == 200
    sent = json.dumps(fake_llm["payload"])
    for secret in ["Rahul Sharma", "rahul.s@gmail.com", "9876543210"]:
        assert secret not in sent
    assert "<PERSON_1>" in sent and "<EMAIL_ADDRESS_1>" in sent and "<IN_PHONE_1>" in sent


def test_answer_is_rehydrated(client, fake_llm):
    fake_llm["reply"] = "Dear <PERSON_1>, we will call you on <IN_PHONE_1>."
    r = chat(client, "Write a note to Rahul Sharma, phone 9876543210")
    answer = r.json()["choices"][0]["message"]["content"]
    assert answer == "Dear Rahul Sharma, we will call you on 9876543210."


def test_same_value_same_placeholder_across_messages(client, fake_llm):
    body = {"model": "x", "messages": [
        {"role": "system", "content": "You help Priya Nair."},
        {"role": "user", "content": "Remind Priya Nair about the meeting."}]}
    client.post(URL, json=body)
    msgs = fake_llm["payload"]["messages"]
    assert "<PERSON_1>" in msgs[0]["content"] and "<PERSON_1>" in msgs[1]["content"]


def test_tool_call_arguments_are_scanned(client, fake_llm):
    body = {"model": "x", "messages": [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "send_sms", "arguments": '{"to": "9876543210"}'}}]},
        {"role": "user", "content": "done?"}]}
    client.post(URL, json=body)
    assert "9876543210" not in json.dumps(fake_llm["payload"])


def test_clean_text_unchanged(client, fake_llm):
    chat(client, "Explain how binary search works.")
    assert fake_llm["user_text"] == "Explain how binary search works."


# ---------------------------------------------------------------- blocking
@pytest.mark.parametrize("text,entity", [
    (f"My Aadhaar is {VALID_AADHAAR}", "IN_AADHAAR"),
    ("Charge card 4111 1111 1111 1111", "CREDIT_CARD"),
    ("PAN: ABCPE1234F", "IN_PAN"),
])
def test_sensitive_ids_are_blocked(client, fake_llm, text, entity):
    r = chat(client, text)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "pii_blocked"
    assert entity in r.json()["error"]["message"]
    assert "payload" not in fake_llm  # LLM was never called


def test_random_12_digits_not_aadhaar(client, fake_llm):
    # Fails the Verhoeff checksum, so it's an order number, not an Aadhaar.
    assert not verhoeff_valid("234123412345")
    r = chat(client, "Order 2341 2341 2345 has shipped")
    assert r.status_code == 200


# ---------------------------------------------------------------- fail closed
def test_detector_crash_blocks_request(client, fake_llm, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(client.app.state.detector, "detect", boom)
    r = chat(client, "Call Rahul on 9876543210")
    assert r.status_code == 503
    assert "payload" not in fake_llm


def test_images_rejected(client, fake_llm):
    body = {"model": "x", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    r = client.post(URL, json=body)
    assert r.status_code == 400
    assert "payload" not in fake_llm


def test_streaming_rejected(client, fake_llm):
    r = chat(client, "hi", stream=True)
    assert r.status_code == 400


# ---------------------------------------------------------------- audit log
def test_audit_log_has_no_raw_pii(client, fake_llm):
    chat(client, "Email Rahul Sharma at rahul.s@gmail.com")
    log = open(os.environ["AUDIT_LOG_PATH"]).read()
    assert "rahul.s@gmail.com" not in log and "Rahul Sharma" not in log
    assert "EMAIL_ADDRESS" in log


# ---------------------------------------------------------------- unit
def test_unknown_placeholder_left_alone():
    v = Vault()
    v.placeholder_for("PERSON", "Asha")
    assert rehydrate("<PERSON_1> and <PERSON_9>", v) == "Asha and <PERSON_9>"
