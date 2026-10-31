"""
Custom PII recognizers for Indian identifiers.

Presidio ships with recognizers for emails, credit cards, IPs etc., but not for
Aadhaar, PAN, UPI IDs or Indian mobile numbers. Each recognizer below is a
regex (fast, deterministic) plus, where the format allows it, a checksum
validator that rejects random numbers that merely *look* like the identifier.
"""
from presidio_analyzer import Pattern, PatternRecognizer

# ---------------------------------------------------------------------------
# Verhoeff checksum (used by Aadhaar). Tables are the standard dihedral-group
# multiplication (D), permutation (P) and inverse (INV) tables.
# ---------------------------------------------------------------------------
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string passes the Verhoeff checksum."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


class AadhaarRecognizer(PatternRecognizer):
    """12 digits, first digit 2-9, optional spaces/hyphens in 4-4-4 groups,
    and the last digit must be a valid Verhoeff check digit."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=[
                Pattern(
                    "aadhaar",
                    r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
                    0.5,
                )
            ],
            context=["aadhaar", "aadhar", "uid", "uidai"],
        )

    def validate_result(self, pattern_text: str):
        digits = "".join(ch for ch in pattern_text if ch.isdigit())
        # True -> confidence raised to max; False -> match discarded
        return len(digits) == 12 and verhoeff_valid(digits)


class PanRecognizer(PatternRecognizer):
    """PAN: 5 letters, 4 digits, 1 letter. The 4th letter encodes the holder
    type (P = person, C = company, ...), which we use to cut false positives."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_PAN",
            patterns=[
                Pattern("pan", r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b", 0.85)
            ],
            context=["pan", "permanent account number", "income tax"],
        )


class IndianPhoneRecognizer(PatternRecognizer):
    """Indian mobile numbers: optional +91 / 91 / 0 prefix, then 10 digits
    starting with 6-9 (optionally split 5-5 by a space or hyphen)."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_PHONE",
            patterns=[
                Pattern(
                    "in_mobile",
                    r"(?<![\w+])(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)",
                    0.7,
                )
            ],
            context=["phone", "mobile", "call", "contact", "whatsapp", "number"],
        )


class UpiRecognizer(PatternRecognizer):
    """UPI IDs look like name@bank (e.g. rahul.s@okaxis). Unlike emails there
    is no dot in the part after '@', which is how we tell them apart."""

    def __init__(self):
        super().__init__(
            supported_entity="UPI_ID",
            patterns=[
                Pattern(
                    "upi",
                    # not followed by ".com"-style domain (that's an email),
                    # but a sentence-ending full stop is fine
                    r"\b[a-zA-Z0-9._-]{2,256}@[a-zA-Z]{2,64}\b(?![\w@])(?!\.\w)",
                    0.6,
                )
            ],
            context=["upi", "gpay", "phonepe", "paytm", "vpa", "pay"],
        )


def custom_recognizers():
    return [
        AadhaarRecognizer(),
        PanRecognizer(),
        IndianPhoneRecognizer(),
        UpiRecognizer(),
    ]
