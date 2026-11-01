"""
Generates labelled Indian test data, so the evaluation doesn't rest only on
sentences written by the detector's author.

  dev  - misses here are looked at and fixed
  test - different seed AND different sentence templates; only ever reported.
         Same templates with new values would just repeat what dev tuned for.

Run from the project root:   python -m eval.synthetic
Then score a set with:       python -m eval.evaluate eval/synthetic_test.jsonl
"""
import json
import random
import re
import string
from pathlib import Path

from faker import Faker

from app.recognizers import verhoeff_valid

ROOT = Path(__file__).resolve().parent.parent
SIZE = 200  # sentences per set, split evenly across the templates

DEV_TEMPLATES = [
    "Please send the invoice to {name} at {email}.",
    "Call {first} on {phone} before noon.",
    "{name} from {city} raised a complaint about the delivery.",
    "Refund the amount to {upi} for customer {name}.",
    "My Aadhaar number is {aadhaar}, please update it.",
    "PAN {pan} belongs to {name}.",
    "Charge {card} for the annual plan.",
    "Customer passport no: {passport}",
    "Voter ID {voter} was verified at the camp.",
    "DL number {licence}, valid till next year.",
    "Transfer to A/C {account}, IFSC {ifsc}.",
    "DOB: {dob}",
    "Ship to {city}, PIN {pin}.",
    "The server at {ip} belongs to {name}.",
    "Office landline is {landline}.",
    "hi this is {name_lower}, my number is {phone}",
    '{"customer": "{name}", "email": "{email}"}',
    "login password: {password}",
    "Why is {api_key} returning a 401?",
    "{first} and {first} will join the call from {city}.",
    # no PII
    "Order {order} was shipped on {date}.",
    "Batch {sku} passed quality checks.",
    "We sold {qty} units in Q3.",
    "Please reset your password from the settings page.",
    "The meeting is on {date} in the main hall.",
]

TEST_TEMPLATES = [
    "Kindly forward the statement to {email} and copy {name}.",
    "Ring {name} at {phone} regarding the policy renewal.",
    "Our branch manager {first} moved to {city} last month.",
    "Collect payment via {upi}.",
    "Aadhaar: {aadhaar}",
    "Please verify income tax PAN {pan}.",
    "Card number {card} was declined.",
    "Renew passport {passport} before travelling.",
    "Electoral roll shows EPIC {voter}.",
    "Licence no {licence} was suspended.",
    "Beneficiary account {account}, branch code {ifsc}.",
    "He was born on {dob} in {city}.",
    "Pincode {pin} is not serviceable.",
    "Blocked login attempts from {ip}.",
    "Reach the clinic on {landline}.",
    "thanks, regards {name_lower}",
    "to: {name}\nphone: {phone}\nemail: {email}",
    "my otp is {otp}",
    "token = {api_key}",
    "{first} told {first} that the report was ready.",
    # no PII
    "Invoice {order} is due on {date}.",
    "Part number {sku} is out of stock.",
    "Revenue grew by {qty} percent this year.",
    "The OTP screen takes too long to load.",
    "Standup moves to {date}.",
]


class Generator:
    def __init__(self, seed: int):
        self.fake = Faker("en_IN")
        self.fake.seed_instance(seed)
        self.rng = random.Random(seed)

    def digits(self, n: int) -> str:
        return "".join(self.rng.choices(string.digits, k=n))

    def letters(self, n: int) -> str:
        return "".join(self.rng.choices(string.ascii_uppercase, k=n))

    def phone(self) -> str:
        n = self.rng.choice("6789") + self.digits(9)
        return self.rng.choice([n, f"+91 {n[:5]} {n[5:]}", f"{n[:5]}-{n[5:]}", f"0{n}", f"+91-{n}"])

    def aadhaar(self) -> str:
        body = self.rng.choice("23456789") + self.digits(10)
        check = next(d for d in string.digits if verhoeff_valid(body + d))
        n = body + check
        return self.rng.choice([n, f"{n[:4]} {n[4:8]} {n[8:]}", f"{n[:4]}-{n[4:8]}-{n[8:]}"])

    def card(self) -> str:
        n = self.fake.credit_card_number(card_type="visa16")
        return self.rng.choice([n, " ".join(n[i:i + 4] for i in range(0, 16, 4))])

    def passport(self) -> str:
        first = self.rng.choice("ABCDEFGHJKLMNPRSTUVWY")
        return f"{first}{self.rng.randint(1, 9)}{self.digits(5)}{self.rng.randint(1, 9)}"

    def licence(self) -> str:
        state = self.rng.choice(["MH", "KA", "DL", "TN", "UP", "GJ", "WB", "KL"])
        year = self.rng.randint(1995, 2024)
        rto = self.digits(2)
        return self.rng.choice([f"{state}{rto}{year}{self.digits(7)}",
                                f"{state}-{rto}-{year}-{self.digits(7)}",
                                f"{state}{rto} {year}{self.digits(7)}"])

    def date(self) -> str:
        d = self.fake.date_of_birth(minimum_age=18, maximum_age=80)
        return d.strftime(self.rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y"]))

    def api_key(self) -> str:
        body = self.fake.password(length=32, special_chars=False)
        return self.rng.choice([f"sk-proj-{body}", f"gsk_{body}", f"AKIA{self.letters(16)}",
                                f"ghp_{self.fake.password(length=36, special_chars=False)}",
                                f"rzp_live_{body}"])

    def slots(self) -> dict:
        """slot -> (entity type, or None for filler that isn't PII; value)"""
        f, r = self.fake, self.rng
        return {
            "name": ("PERSON", f.name),
            "first": ("PERSON", f.first_name),
            "name_lower": ("PERSON", lambda: f.name().lower()),
            "city": ("LOCATION", f.city),
            "phone": ("IN_PHONE", self.phone),
            "landline": ("PHONE_NUMBER", f.phone_number),
            "email": ("EMAIL_ADDRESS", f.email),
            "upi": ("UPI_ID", lambda: f"{f.user_name()}@{r.choice(['okaxis', 'oksbi', 'ybl', 'paytm'])}"),
            "ip": ("IP_ADDRESS", f.ipv4_public),
            "aadhaar": ("IN_AADHAAR", self.aadhaar),
            "pan": ("IN_PAN", lambda: f"{self.letters(3)}P{self.letters(1)}{self.digits(4)}{self.letters(1)}"),
            "card": ("CREDIT_CARD", self.card),
            "passport": ("IN_PASSPORT", self.passport),
            "voter": ("IN_VOTER_ID", lambda: self.letters(3) + self.digits(7)),
            "licence": ("IN_DRIVING_LICENCE", self.licence),
            "ifsc": ("IN_IFSC", lambda: f"{r.choice(['HDFC', 'ICIC', 'SBIN', 'UTIB'])}0{self.digits(6)}"),
            "account": ("BANK_ACCOUNT", lambda: self.digits(r.randint(9, 18)).replace("0", "1", 1)),
            "dob": ("DATE_OF_BIRTH", self.date),
            # Not f.postcode: it starts some with 0, which no real PIN code does.
            "pin": ("PIN_CODE", lambda: str(r.randint(1, 9)) + self.digits(5)),
            "password": ("PASSWORD", lambda: f.password(length=12)),
            "otp": ("PASSWORD", lambda: self.digits(6)),
            "api_key": ("API_KEY", self.api_key),
            "order": (None, lambda: self.digits(10)),
            "sku": (None, lambda: self.letters(3) + self.digits(7)),
            "date": (None, self.date),
            "qty": (None, lambda: str(r.randint(2, 950))),
        }


def generate(templates: list[str], seed: int) -> list[dict]:
    slots = Generator(seed).slots()
    rows = []
    for i in range(SIZE):
        pii = []

        def fill(m: re.Match) -> str:
            entity, make = slots[m.group(1)]
            value = make()
            if entity:
                pii.append([entity, value])
            return value

        rows.append({"text": re.sub(r"\{(\w+)\}", fill, templates[i % len(templates)]), "pii": pii})
    return rows


def main():
    for name, templates, seed in [("dev", DEV_TEMPLATES, 1), ("test", TEST_TEMPLATES, 2)]:
        path = ROOT / "eval" / f"synthetic_{name}.jsonl"
        rows = generate(templates, seed)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{path.name}: {len(rows)} sentences, {sum(len(r['pii']) for r in rows)} PII spans")


if __name__ == "__main__":
    main()
