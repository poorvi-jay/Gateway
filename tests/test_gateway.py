"""
End-to-end tests. The upstream LLM is replaced by a fake that records exactly
what it received, so we can assert that no raw PII ever reaches it.

Run:  pytest -q
"""
import os

os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")
os.environ["AUDIT_LOG_PATH"] = os.path.join(os.path.dirname(__file__), "test_audit.log")
os.environ["POLICY_PATH"] = os.path.join(os.path.dirname(__file__), "..", "policy.yaml")
os.environ["RATE_LIMIT_PER_MIN"] = "0"         # the suite sends far more than 30/min
os.environ["GLOBAL_RATE_LIMIT_PER_MIN"] = "0"

import json  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as gw  # noqa: E402
from app.detector import Finding, _finding, _merge_names  # noqa: E402
from app.gliner_compare import GlinerDetector, entity_type  # noqa: E402
from app.policy import Policy, load_policy  # noqa: E402
from app.recognizers import verhoeff_valid  # noqa: E402
from app.sanitizer import sanitize_request  # noqa: E402
from app.tokenizer import Vault, rehydrate, repair  # noqa: E402
from eval.synthetic import DEV_TEMPLATES, TEST_TEMPLATES, generate  # noqa: E402

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


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_demo_page_has_a_label_for_every_scanned_type(client):
    page = client.get("/").text
    missing = [e for e in client.app.state.policy.scanned_entities if f"{e}:" not in page]
    assert not missing


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
    ("Charge card 4111 1111 1111 1111", "CREDIT_CARD"),
    ("Mistyped card number 4111 1111 1111 1112", "CREDIT_CARD"),
    ("card 4111.1111.1111.1111", "CREDIT_CARD"),
    ("debit card 411111111111", "CREDIT_CARD"),  # a few digits dropped
])
def test_sensitive_ids_are_blocked(client, fake_llm, text, entity):
    r = chat(client, text)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "pii_blocked"
    assert entity in r.json()["error"]["message"]
    assert "payload" not in fake_llm  # LLM was never called


@pytest.mark.parametrize("text,value,entity", [
    (f"My Aadhaar is {VALID_AADHAAR}", VALID_AADHAAR, "IN_AADHAAR"),
    ("my aadhar is 987654321013", "987654321013", "IN_AADHAAR"),  # fails the checksum
    ("email and aadhar 129876543210.", "129876543210", "IN_AADHAAR"),  # impossible first digit
    ("aadhaar no 12987654321", "12987654321", "IN_AADHAAR"),      # a digit dropped
    ("aadhaar no 1298765432101", "1298765432101", "IN_AADHAAR"),  # a digit added
    ("aadhaar 4987.1234.5679", "4987.1234.5679", "IN_AADHAAR"),   # dots
    ("adhaar 129876543210", "129876543210", "IN_AADHAAR"),        # misspelt context word
    ("aadhar card 129876543210", "129876543210", "IN_AADHAAR"),   # "card" must not block it
    ("PAN: ABCPE1234F", "ABCPE1234F", "IN_PAN"),
    ("My PAN is ABCXX1234F", "ABCXX1234F", "IN_PAN"),             # invalid fourth letter
    ("Driving licence MH12 20110012345", "MH12 20110012345", "IN_DRIVING_LICENCE"),
])
def test_ids_are_redacted_and_never_come_back(client, fake_llm, text, value, entity):
    r = chat(client, text)
    assert r.status_code == 200
    assert value not in json.dumps(fake_llm["payload"])
    answer = r.json()["choices"][0]["message"]["content"]
    assert f"[REDACTED_{entity}]" in answer and value not in answer


@pytest.mark.parametrize("action,sent,answer", [
    ("tokenize", "<IN_AADHAAR_1>", VALID_AADHAAR),
    ("redact", "[REDACTED_IN_AADHAAR]", "[REDACTED_IN_AADHAAR]"),
])
def test_aadhaar_can_be_unblocked_by_policy(client, fake_llm, monkeypatch, action, sent, answer):
    monkeypatch.setitem(client.app.state.policy.actions, "IN_AADHAAR", action)
    r = chat(client, f"My Aadhaar is {VALID_AADHAAR}")
    assert r.status_code == 200
    assert fake_llm["user_text"] == f"My Aadhaar is {sent}"
    assert VALID_AADHAAR not in json.dumps(fake_llm["payload"])
    assert r.json()["choices"][0]["message"]["content"] == f"Echo: My Aadhaar is {answer}"


def test_redact_covers_every_mention_and_is_never_restored():
    payload = {"messages": [{"role": "user", "content": "Ask Priya Nair, then Priya"}]}
    detector = StubDetector({"Ask Priya Nair, then Priya": ["Priya Nair"]})
    out = sanitize_request(payload, detector, Policy(actions={"PERSON": "redact"}))
    assert out.payload["messages"][0]["content"] == "Ask [REDACTED_PERSON], then [REDACTED_PERSON]"
    assert len(out.vault) == 0


class FakeGliner:
    def detect(self, text):
        s = text.index("Rahul")
        return [Finding("PERSON", s, s + 5, 0.9, "Rahul")]


@pytest.mark.parametrize("label,value,expected", [
    ("password", "ABCPE1234F", "NATIONAL_ID"),
    ("credit_debit_card", "4987 1234 5679", "NATIONAL_ID"),
    ("password", "aadhar 987654321012", "NATIONAL_ID"),
    ("account_number", "123456789012", "BANK_ACCOUNT"),   # fails the Aadhaar checksum
    ("credit_debit_card", "4111 1111 1111 1111", "CREDIT_CARD"),
    ("password", "hunter2", "PASSWORD"),
])
def test_gliner_labels_pan_and_aadhaar_as_national_id(label, value, expected):
    assert entity_type(label, value) == expected


def test_gliner_comparison_is_off_by_default(client):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/debug/gliner", json=body).json() == {"available": False}


def test_gliner_comparison_masks_text_and_never_calls_llm(client, fake_llm, monkeypatch):
    monkeypatch.setattr(client.app.state, "gliner", FakeGliner())
    body = {"messages": [{"role": "user", "content": "Call Rahul"}]}
    data = client.post("/debug/gliner", json=body).json()
    assert data["masked"] == "Call <PERSON_1>" and data["entity_types"] == ["PERSON"]
    assert "payload" not in fake_llm


@pytest.mark.parametrize("extra", [
    {"metadata": {"customer": "rahul.s@gmail.com"}},
    {"prediction": {"type": "content", "content": "rahul.s@gmail.com"}},
])
def test_fields_we_cant_inspect_are_rejected(client, fake_llm, extra):
    r = chat(client, "hi", **extra)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_content"
    assert "payload" not in fake_llm


def test_unknown_message_field_is_rejected(client, fake_llm):
    message = {"role": "user", "content": "hi", "note": "rahul.s@gmail.com"}
    assert client.post(URL, json={"model": "x", "messages": [message]}).status_code == 400
    assert "payload" not in fake_llm


def test_empty_fields_echoed_by_sdks_are_fine(client, fake_llm):
    message = {"role": "assistant", "content": "hi", "refusal": None, "annotations": []}
    body = {"model": "x", "metadata": {}, "temperature": 0.2, "messages": [message]}
    assert client.post(URL, json=body).status_code == 200


def test_response_format_is_scanned(client, fake_llm):
    schema = {"name": "x", "description": "for rahul.s@gmail.com"}
    chat(client, "hi", response_format={"type": "json_schema", "json_schema": schema})
    assert "rahul.s@gmail.com" not in json.dumps(fake_llm["payload"])
    assert fake_llm["payload"]["response_format"]["json_schema"]["name"] == "x"


def test_card_in_response_format_blocks(client, fake_llm):
    schema = {"name": "x", "description": "card 4111 1111 1111 1111"}
    r = chat(client, "hi", response_format={"type": "json_schema", "json_schema": schema})
    assert r.status_code == 400 and r.json()["error"]["code"] == "pii_blocked"
    assert "payload" not in fake_llm


@pytest.mark.parametrize("extra", [{"stop": ["rahul.s@gmail.com"]},
                                   {"tool_choice": {"type": "function",
                                                    "function": {"name": "rahul.s@gmail.com"}}}])
def test_stop_and_tool_choice_with_pii_are_rejected(client, fake_llm, extra):
    assert chat(client, "hi", **extra).status_code == 400
    assert "payload" not in fake_llm


def test_stop_and_tool_choice_without_pii_pass(client, fake_llm):
    assert chat(client, "hi", stop=["END"], tool_choice="auto").status_code == 200


def test_oversized_body_is_rejected(client, fake_llm):
    assert chat(client, "hi " * 70_000).status_code == 400
    assert "payload" not in fake_llm


def test_blocked_text_never_leaves_for_remote_gliner(client, monkeypatch):
    def post(*args, **kwargs):
        raise AssertionError("blocked text left the gateway")

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(gw, "GLINER_URL", "https://gliner.test")
    monkeypatch.setattr(client.app.state, "gliner", GlinerDetector("https://gliner.test"),
                        raising=False)
    body = {"messages": [{"role": "user", "content": "Charge card 4111 1111 1111 1111"}]}
    assert client.post("/debug/gliner", json=body).json() == {"available": True, "blocked": True}


def test_remote_gliner_gets_the_shared_token(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs["headers"])
        return httpx.Response(200, json=[], request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("GLINER_TOKEN", "shared")
    GlinerDetector("https://gliner.test").detect("hi")
    assert seen["Authorization"] == "Bearer shared"


def test_gliner_service_result_masks_text(client, monkeypatch):
    request = httpx.Request("POST", "https://gliner.test/detect")
    answer = [{"label": "person", "start": 5, "end": 10, "score": 0.9}]
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: httpx.Response(200, json=answer, request=request))
    monkeypatch.setattr(client.app.state, "gliner", GlinerDetector("https://gliner.test/"),
                        raising=False)  # an earlier monkeypatch's undo removes the attribute
    body = {"messages": [{"role": "user", "content": "Call Rahul"}]}
    assert client.post("/debug/gliner", json=body).json()["masked"] == "Call <PERSON_1>"


@pytest.mark.parametrize("reply", [
    httpx.Response(200, text="<html>Service unavailable</html>"),
    httpx.Response(503),
    httpx.ConnectTimeout("no answer"),
])
def test_gliner_service_failure_is_a_clean_503(client, monkeypatch, reply):
    def post(*args, **kwargs):
        if isinstance(reply, Exception):
            raise reply
        reply.request = httpx.Request("POST", "https://gliner.test/detect")
        return reply

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(client.app.state, "gliner", GlinerDetector("https://gliner.test"),
                        raising=False)
    body = {"messages": [{"role": "user", "content": "Call Rahul"}]}
    r = client.post("/debug/gliner", json=body)
    assert r.status_code == 503 and r.json()["error"]["code"] == "gliner_unavailable"


def test_unknown_policy_action_refuses_to_start(tmp_path):
    bad = tmp_path / "policy.yaml"
    bad.write_text("entities:\n  IN_AADHAAR: mask\n")
    with pytest.raises(ValueError):
        load_policy(bad)


@pytest.mark.parametrize("text,entity", [
    ("My passport number is K1234567", "IN_PASSPORT"),
    ("K1234567 is my passport", "IN_PASSPORT"),
    ("Voter ID ABC1234567", "IN_VOTER_ID"),
    ("Use sk-proj-abc123def456ghi789jkl012", "API_KEY"),
    ("key = AKIAIOSFODNN7EXAMPLE", "API_KEY"),
    ("token rzp_live_abcdef1234567890", "API_KEY"),
    ("-----BEGIN RSA PRIVATE KEY-----", "API_KEY"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc123def456", "API_KEY"),
    ("Authorization: Bearer abcdef123456abcdef123456", "API_KEY"),
])
def test_ids_and_keys_are_blocked(client, fake_llm, text, entity):
    r = chat(client, text)
    assert r.status_code == 400 and entity in r.json()["error"]["message"]
    assert "payload" not in fake_llm


@pytest.mark.parametrize("text,secret,placeholder", [
    ("IFSC is HDFC0001234", "HDFC0001234", "<IN_IFSC_1>"),
    ("Pay to account number 123456789012", "123456789012", "<BANK_ACCOUNT_1>"),
    ("A/C no 123456789012", "123456789012", "<BANK_ACCOUNT_1>"),
    ("Your OTP is 482913.", "482913", "<PASSWORD_1>."),
    ("CVV 123", "123", "<PASSWORD_1>"),
    ("Date of birth: 12/03/1990", "12/03/1990", "<DATE_OF_BIRTH_1>"),
    ("DOB 1990-03-12", "1990-03-12", "<DATE_OF_BIRTH_1>"),
    ("She was born on 12 March 1990", "12 March 1990", "<DATE_OF_BIRTH_1>"),
    ("PIN code 560001", "560001", "<PIN_CODE_1>"),
    ("my password is hunter2", "hunter2", "<PASSWORD_1>"),
    ('{"password": "S3cret!x"}', "S3cret!x", "<PASSWORD_1>"),
    ("UPI PIN: 123456", "123456", "<PASSWORD_1>"),
    ("Invoice for Hamsini Goda today", "Hamsini Goda", "<PERSON_1>"),
    ("Invoice for Maya D'Souza today", "Maya D'Souza", "<PERSON_1>"),
    ("Moved to Warangal", "Warangal", "<LOCATION_1>"),
    ("Office landline is 04696291370", "04696291370", "<PHONE_NUMBER_1>"),
])
def test_new_types_are_tokenized(client, fake_llm, text, secret, placeholder):
    chat(client, text)
    assert secret not in fake_llm["user_text"] and placeholder in fake_llm["user_text"]


@pytest.mark.parametrize("text", [
    "The meeting moved to 12/03/2026",       # dates stay allowed without context
    "Order 123456789012 has shipped",        # digits alone are not an account
    "Shopping cart 560001 is empty",         # "pin" must not match inside "shopping"
    "Ticket K1234567 and batch ABC1234567",  # ID shapes alone must not block
    "Reference 4111 1111 1111 1112 closed",  # a failed Luhn check needs a word like "card"
    "Part ABCXX1234F is in stock",           # PAN shape needs the word "PAN"
    "Password is required for this form",
    "Send the token issue logs today",       # "is" must not match inside "issue"
    "Ask Customer Support about it",         # not every capitalized pair is a name
    "Pay Rs 500 today",
    "The task-management-system-for-teams is live",
])
def test_lookalikes_pass_untouched(client, fake_llm, text):
    assert chat(client, text).status_code == 200
    assert fake_llm["user_text"] == text


def test_password_keyword_stays_readable(client, fake_llm):
    fake_llm["reply"] = "Your password is <PASSWORD_1>."
    r = chat(client, "password: hunter2")
    assert fake_llm["user_text"] == "password: <PASSWORD_1>"
    assert r.json()["choices"][0]["message"]["content"] == "Your password is hunter2."


def test_random_12_digits_not_aadhaar(client, fake_llm):
    # Fails the Verhoeff checksum, so it's an order number, not an Aadhaar.
    assert not verhoeff_valid("234123412345")
    r = chat(client, "Order 2341 2341 2345 has shipped")
    assert r.status_code == 200


@pytest.mark.parametrize("text", [
    "my card is lost, call 9876543210",   # a phone number near "card" isn't a card
    "Reference 2024-01-15 10 has shipped",
])
def test_loose_id_patterns_leave_other_numbers_alone(client, text):
    types = [f.entity_type for f in client.app.state.detector.detect(text)]
    assert "CREDIT_CARD" not in types and "IN_AADHAAR" not in types


def test_checksum_alone_cant_make_an_aadhaar_of_first_digit_1(client):
    assert verhoeff_valid("123456789010")
    findings = client.app.state.detector.detect("Order 123456789010 has shipped")
    assert "IN_AADHAAR" not in [f.entity_type for f in findings]


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


# ---------------------------------------------------------------- bypasses
@pytest.mark.parametrize("text", [
    "Charge <CARD_4111111111111111> now",
    "Charge CARD_4111111111111111 now",
])
def test_placeholder_shape_cannot_hide_blocked_id(client, fake_llm, text):
    r = chat(client, text)
    assert r.status_code == 400
    assert "payload" not in fake_llm


def test_placeholder_shape_cannot_hide_phone(client, fake_llm):
    chat(client, "Call <X_9876543210> now")
    assert "9876543210" not in json.dumps(fake_llm["payload"])


def test_email_with_underscore_fully_masked(client, fake_llm):
    chat(client, "Write to rahul_s@gmail.com")
    assert "rahul" not in json.dumps(fake_llm["payload"])


def test_response_keeps_our_placeholder_but_rehydrates(client, fake_llm):
    fake_llm["reply"] = "Noted, <IN_PHONE_1>."
    r = chat(client, "Save 9876543210")
    assert r.json()["choices"][0]["message"]["content"] == "Noted, 9876543210."


def test_message_name_is_scanned(client, fake_llm):
    body = {"model": "x", "messages": [
        {"role": "system", "content": "You help Priya Nair."},
        {"role": "user", "name": "Priya Nair", "content": "hi"}]}
    client.post(URL, json=body)
    assert "Priya" not in json.dumps(fake_llm["payload"])
    assert fake_llm["payload"]["messages"][1]["name"] == "PERSON_1"  # OpenAI rejects "<"


def test_tool_message_name_is_kept(client, fake_llm):
    body = {"model": "x", "messages": [
        {"role": "system", "content": "You help Priya Nair."},
        {"role": "tool", "name": "Priya Nair", "tool_call_id": "1", "content": "hi"},
        {"role": "user", "content": "ok"}]}
    client.post(URL, json=body)
    assert fake_llm["payload"]["messages"][1]["name"] == "Priya Nair"


def test_legacy_function_call_is_scanned(client, fake_llm):
    body = {"model": "x", "messages": [
        {"role": "assistant", "content": None,
         "function_call": {"name": "send_sms", "arguments": '{"to": "9876543210"}'}},
        {"role": "user", "content": "done?"}]}
    client.post(URL, json=body)
    assert "9876543210" not in json.dumps(fake_llm["payload"])


@pytest.mark.parametrize("message", [
    {"role": "user", "content": {"text": "call 9876543210"}},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "f", "arguments": {"to": "9876543210"}}}]},
])
def test_uninspectable_shapes_rejected(client, fake_llm, message):
    r = client.post(URL, json={"model": "x", "messages": [message]})
    assert r.status_code == 400
    assert "payload" not in fake_llm


def tool(description, **properties):
    return {"type": "function", "function": {
        "name": "send_mail", "description": description,
        "parameters": {"type": "object", "properties": properties, "required": ["name"]}}}


def test_user_field_is_scanned(client, fake_llm):
    chat(client, "hi", user="rahul.s@gmail.com")
    assert fake_llm["payload"]["user"] == "<EMAIL_ADDRESS_1>"


def test_non_string_user_rejected(client, fake_llm):
    assert chat(client, "hi", user=9876543210).status_code == 400
    assert "payload" not in fake_llm


def test_tool_definitions_are_scanned_and_stay_valid(client, fake_llm):
    tools = [tool("Sends mail to rahul.s@gmail.com",
                  name={"type": "string", "description": "Or call 9876543210"},
                  to={"type": "string", "enum": ["rahul.s@gmail.com"]})]
    chat(client, "hi", tools=tools)
    sent = fake_llm["payload"]["tools"][0]["function"]
    assert "rahul" not in json.dumps(sent) and "9876543210" not in json.dumps(sent)
    assert sent["name"] == "send_mail" and sent["parameters"]["required"] == ["name"]
    assert sent["parameters"]["properties"]["name"]["type"] == "string"
    assert sent["parameters"]["properties"]["to"]["enum"] == ["<EMAIL_ADDRESS_1>"]


def test_clean_tools_scanned_once_but_still_masked(client, fake_llm, monkeypatch):
    detector, calls = client.app.state.detector, []
    real = detector.detect
    monkeypatch.setattr(detector, "detect",
                        lambda text, ignore=(): calls.append(text) or real(text, ignore))
    tools = [tool("Looks up one billing record, default 12 items.")]
    chat(client, "hi", tools=tools)
    chat(client, "Keep 12 items for 9876543210", tools=tools)
    assert sum("billing record" in t for t in calls) == 1

    fake_llm["payload"].clear()
    tools = [tool("Looks up one billing record, fallback 9876543210.")]
    chat(client, "Save 9876543210", tools=tools)
    chat(client, "Save 9876543210", tools=tools)
    assert "9876543210" not in json.dumps(fake_llm["payload"])


def test_legacy_functions_are_scanned(client, fake_llm):
    chat(client, "hi", functions=[{"name": "f", "description": "Call 9876543210"}])
    assert "9876543210" not in json.dumps(fake_llm["payload"])


def test_blocked_id_in_tool_definition_blocks_request(client, fake_llm):
    r = chat(client, "hi", tools=[tool("Card 4111 1111 1111 1111")])
    assert r.status_code == 400 and r.json()["error"]["code"] == "pii_blocked"
    assert "payload" not in fake_llm


def test_value_seen_in_message_is_masked_in_tools(client, fake_llm):
    chat(client, "Save 9876543210", tools=[tool("Default is 9876543210.")])
    sent = fake_llm["payload"]
    assert "<IN_PHONE_1>" in sent["messages"][0]["content"]
    assert "<IN_PHONE_1>" in sent["tools"][0]["function"]["description"]


def test_debug_shows_sanitized_tools_and_user(client):
    body = {"messages": [{"role": "user", "content": "hi"}], "user": "rahul.s@gmail.com",
            "tools": [tool("Mail rahul.s@gmail.com")]}
    out = client.post("/debug/sanitize", json=body).json()
    assert out["user"] == "<EMAIL_ADDRESS_1>"
    assert "rahul" not in json.dumps(out)
    assert out["entity_types"] == ["EMAIL_ADDRESS", "EMAIL_ADDRESS"]


@pytest.mark.parametrize("reply", ["<Person_1> called.", "<PERSION_1> called.", "<person_1> called."])
def test_garbled_placeholder_is_restored(client, fake_llm, reply):
    fake_llm["reply"] = reply
    r = chat(client, "Write a note to Rahul Sharma")
    assert r.json()["choices"][0]["message"]["content"] == "Rahul Sharma called."


def test_garbled_placeholder_needs_matching_number():
    v = Vault()
    v.placeholder_for("PERSON", "Asha")
    assert repair("<PERSION_9>", v) == "<PERSION_9>"
    assert repair("<IN_PHONE_1>", v) == "<IN_PHONE_1>"


def failing_upstream(monkeypatch, status):
    async def fail(payload):
        response = httpx.Response(status, json={"error": {"message": "from upstream"}})
        raise httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://x"),
                                    response=response)
    monkeypatch.setattr(gw, "call_upstream", fail)


@pytest.mark.parametrize("status", [400, 429])
def test_client_errors_from_upstream_are_passed_on(client, monkeypatch, status):
    failing_upstream(monkeypatch, status)
    r = chat(client, "hi")
    assert r.status_code == status and r.json()["error"]["message"] == "from upstream"


def test_bad_upstream_key_stays_a_gateway_error(client, monkeypatch):
    failing_upstream(monkeypatch, 401)
    r = chat(client, "hi")
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"


@pytest.mark.parametrize("route", [URL, "/debug/sanitize"])
@pytest.mark.parametrize("body", ["{", "[1]", '{"model": "x"}'])
def test_malformed_body_is_400(client, fake_llm, route, body):
    assert client.post(route, content=body).status_code == 400
    assert "payload" not in fake_llm


def test_debug_rejects_images(client):
    body = {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}
    assert client.post("/debug/sanitize", json=body).status_code == 400


def test_wrong_api_key_rejected(client, fake_llm, monkeypatch):
    monkeypatch.setattr(gw, "GATEWAY_API_KEY", "secret")
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post(URL, json=body, headers={"authorization": "Bearer nope"}).status_code == 401
    ok = client.post(URL, json=body, headers={"authorization": "Bearer secret"})
    assert ok.status_code == 200


# ---------------------------------------------------------------- precision
@pytest.mark.parametrize("text,sent", [
    ("Refund Priya Nair today. Tell Priya once it is done.",
     "Refund <PERSON_1> today. Tell <PERSON_1> once it is done."),
    ("Reimburse Priya Nair today", "<PERSON_1> today"),  # not a known verb: masked whole, no leak
])
def test_word_before_a_full_name_leaves_no_part_showing(client, fake_llm, text, sent):
    chat(client, text)
    assert fake_llm["user_text"] == sent


def test_overlapping_names_are_merged():
    found = _merge_names([Finding("PERSON", 0, 5, 0.85, "Priya"),
                          Finding("PERSON", 0, 10, 0.55, "Priya Nair")], "Priya Nair left")
    assert [f.text for f in found] == ["Priya Nair"]


def test_verb_is_not_masked_as_part_of_name():
    text = "Email Rahul Sharma now"

    class Result:
        entity_type, start, end, score = "PERSON", 0, len("Email Rahul Sharma"), 0.85

    finding = _finding(Result, text)
    assert finding.text == "Rahul Sharma"


def test_quarter_is_not_a_location(client):
    assert client.app.state.detector.detect("Our Q3 revenue grew") == []


class StubDetector:
    """Finds exactly the values given, per text."""
    def __init__(self, by_text):
        self.by_text = by_text

    def detect(self, text):
        return [Finding("PERSON", text.index(v), text.index(v) + len(v), 0.9, v)
                for v in self.by_text.get(text, [])]


def test_propagation_prefers_longest_and_ignores_case():
    texts = {"Priya": ["Priya"], "Priya Nair": ["Priya Nair"]}
    payload = {"messages": [{"role": "user", "content": t} for t in texts]
               + [{"role": "user", "content": "Ask Priya Nair or PRIYA NAIR"}]}
    out = sanitize_request(payload, StubDetector(texts), Policy(actions={"PERSON": "tokenize"}))
    assert "nair" not in out.payload["messages"][2]["content"].lower()


def test_propagation_matches_whole_words_only():
    texts = {"Raj Patel": ["Raj Patel"], "Raj": ["Raj"]}
    payload = {"messages": [{"role": "user", "content": t} for t in texts]
               + [{"role": "user", "content": "Back from Rajasthan"}]}
    out = sanitize_request(payload, StubDetector(texts), Policy(actions={"PERSON": "tokenize"}))
    assert out.payload["messages"][2]["content"] == "Back from Rajasthan"


def sanitize_names(texts, last):
    payload = {"messages": [{"role": "user", "content": t} for t in [*texts, last]]}
    out = sanitize_request(payload, StubDetector(texts), Policy(actions={"PERSON": "tokenize"}))
    return out.payload["messages"][-1]["content"]


def test_part_of_a_name_is_masked_as_that_person():
    assert sanitize_names({"Priya Nair": ["Priya Nair"]},
                          "Ask Priya, Ms Nair or PRIYA") == "Ask <PERSON_1>, Ms <PERSON_1> or <PERSON_1>"


def test_short_name_found_by_ner_is_the_same_person():
    texts = {"Refund Priya Nair": ["Priya Nair"], "Tell Priya": ["Priya"]}
    assert sanitize_names(texts, "Ask Priya") == "Ask <PERSON_1>"


def test_name_part_shared_by_two_people_is_not_merged():
    last = sanitize_names({"Priya Nair": ["Priya Nair"], "Priya Menon": ["Priya Menon"]}, "Ask Priya")
    assert "Priya" not in last and last not in ("Ask <PERSON_1>", "Ask <PERSON_2>")


def test_name_parts_skip_lowercase_words_and_titles():
    assert sanitize_names({"Will Smith": ["Will Smith"]}, "It will rain") == "It will rain"
    assert sanitize_names({"Dr Asha Rao": ["Dr Asha Rao"]}, "See Dr Lee") == "See Dr Lee"


# ---------------------------------------------------------------- unit
def test_synthetic_data_is_reproducible_and_labelled():
    rows = generate(DEV_TEMPLATES + TEST_TEMPLATES, seed=1)
    assert rows == generate(DEV_TEMPLATES + TEST_TEMPLATES, seed=1)
    assert all(value in row["text"] for row in rows for _, value in row["pii"])
    aadhaar = [v for row in rows for e, v in row["pii"] if e == "IN_AADHAAR"]
    assert aadhaar and all(verhoeff_valid(v.replace(" ", "").replace("-", "")) for v in aadhaar)


def test_unknown_placeholder_left_alone():
    v = Vault()
    v.placeholder_for("PERSON", "Asha")
    assert rehydrate("<PERSON_1> and <PERSON_9>", v) == "Asha and <PERSON_9>"


# ---------------------------------------------------------------- pre-submission hardening
def test_misspelled_policy_type_refuses_to_start(tmp_path):
    from app.detector import Detector
    bad = tmp_path / "policy.yaml"
    bad.write_text("entities:\n  IN_AADHAR: redact\n")   # typo for IN_AADHAAR
    with pytest.raises(ValueError, match="IN_AADHAR"):
        Detector(load_policy(bad))


def test_more_than_one_choice_rejected(client, fake_llm):
    r = chat(client, "hello", n=3)
    assert r.status_code == 400
    assert "payload" not in fake_llm


@pytest.mark.parametrize("extra,key,expected", [
    ({}, "max_tokens", 1024),
    ({"max_tokens": 50_000}, "max_tokens", 1024),
    ({"max_tokens": 200}, "max_tokens", 200),
    ({"max_completion_tokens": 50_000}, "max_completion_tokens", 1024),
])
def test_output_tokens_capped(client, fake_llm, extra, key, expected):
    assert chat(client, "hello", **extra).status_code == 200
    assert fake_llm["payload"][key] == expected


def test_bad_max_tokens_rejected(client, fake_llm):
    assert chat(client, "hello", max_tokens="lots").status_code == 400
    assert "payload" not in fake_llm


def test_debug_routes_have_a_smaller_body_limit(client):
    big = {"messages": [{"role": "user", "content": "a " * 15_000}]}
    assert client.post("/debug/sanitize", json=big).status_code == 400
    assert client.post("/debug/gliner", json=big).status_code == 400


def test_debug_sanitize_fails_closed_on_error(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(client.app.state.detector, "detect", boom)
    r = client.post("/debug/sanitize", json={"messages": [{"role": "user", "content": "hi Rahul"}]})
    assert r.status_code == 503


def test_debug_sanitize_runs_off_the_event_loop(client, monkeypatch):
    # A slow scan must hit the timeout, not hang the server.
    import time as _t
    monkeypatch.setattr(gw, "DETECTION_TIMEOUT_S", 0.2)
    monkeypatch.setattr(client.app.state.detector, "detect", lambda *a, **k: _t.sleep(1) or [])
    r = client.post("/debug/sanitize", json={"messages": [{"role": "user", "content": "slow"}]})
    assert r.status_code == 503


def test_rate_limit(client, fake_llm, monkeypatch):
    monkeypatch.setattr(gw, "RATE_LIMIT_PER_MIN", 3)
    monkeypatch.setattr(gw, "limiter", gw.RateLimiter())
    codes = [chat(client, "hello").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    # a different client still gets through
    r = client.post(URL, headers={"x-forwarded-for": "203.0.113.9"},
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_global_rate_limit_stops_forged_ips(client, fake_llm, monkeypatch):
    monkeypatch.setattr(gw, "GLOBAL_RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(gw, "limiter", gw.RateLimiter())
    codes = [client.post(URL, headers={"x-forwarded-for": f"198.51.100.{i}"},
                         json={"model": "x", "messages": [{"role": "user", "content": "hi"}]}
                         ).status_code for i in range(3)]
    assert codes == [200, 200, 429]
