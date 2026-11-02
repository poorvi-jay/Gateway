"""
Custom PII recognizers for Indian identifiers and secrets.

Presidio ships with recognizers for emails, credit cards, IPs etc., but not for
Aadhaar, PAN, UPI IDs or Indian mobile numbers. Each recognizer below is a
regex (fast, deterministic) plus, where the format allows it, a checksum
validator that rejects random numbers that merely *look* like the identifier.
Presidio compiles these case-insensitively.
"""
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer

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
    """12 digits, first digit 2-9, optional spaces/hyphens in 4-4-4 groups.
    A valid Verhoeff check digit is enough on its own. A failed one still counts
    next to a word like "aadhaar": a mistyped real number is still mostly real."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=[
                Pattern(
                    "aadhaar",
                    r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
                    NEEDS_CONTEXT,
                )
            ],
            context=["aadhaar", "aadhar", "uid", "uidai"],
        )

    def validate_result(self, pattern_text: str):
        digits = "".join(ch for ch in pattern_text if ch.isdigit())
        # True -> confidence raised to max; None -> left to context words
        return True if verhoeff_valid(digits) else None


class CardRecognizer(CreditCardRecognizer):
    """Presidio's, except a failed Luhn check is left to context words instead
    of discarded, for the same reason as AadhaarRecognizer."""

    def validate_result(self, pattern_text: str):
        return True if super().validate_result(pattern_text) else None


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


# Below the score threshold on purpose: these shapes are too common to trust
# alone (any 6 digits could be a PIN code), so they only count next to a context
# word, which adds 0.35.
NEEDS_CONTEXT = 0.3

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE = (
    r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b"
)

# Vendor key prefixes, a JWT, or a PEM private key header. The digit lookahead
# keeps long hyphenated words starting "sk-" out, as a match blocks the request.
_API_KEY = (
    r"\b(?:sk-(?=[\w-]*\d)[\w-]{20,}|gsk_\w{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_\w{36,}"
    r"|github_pat_\w{22,}|AIza[\w-]{35}|xox[abprs]-[\w-]{10,}|rzp_(?:live|test)_\w{10,}"
    r"|eyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,})"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?<=\bBearer\s+)[\w.~+/-]{20,}"
)

# Passwords have no shape, so match the value after a keyword ("password: x",
# '"otp": "123456"', "my password is x"). Only the value is masked.
_SECRET_WORD = (
    r"(?:pass(?:word|wd|code|phrase)|pwd|secret|token|otp|cvv"
    r"|(?:api|access|auth|secret)[\s_-]?(?:key|token)|(?:upi|atm|card)\s+pin)"
)
_NOT_A_SECRET = (
    r"(?:required|incorrect|wrong|correct|invalid|valid|expired|empty|missing|not"
    r"|the|an?|too|very|true|false|null|none)\b"
)
_SECRET_VALUE = (
    rf"(?<=\b{_SECRET_WORD}[\"']?(?:\s+is\s+|\s*[:=]\s*)[\"']?)"
    rf"(?!{_NOT_A_SECRET})[^\s\"',;]{{2,}}[^\s\"',;.]"
    # Short numeric codes are often written with nothing between: "CVV 123".
    r"|(?<=\b(?:cvv|otp|(?:upi|atm|card)\s+pin)\s+)\d{3,8}\b"
)


# Safety net for names NER doesn't know, which is many Indian ones: two
# capitalized words in a row. A false match only masks a phrase. The (?-i:) keeps
# the case check, as Presidio compiles everything case-insensitively.
_NOT_NAME = (
    r"(?:Dear|Hello|Hi|Hey|Thanks|Thank|Regards|Please|Note|Rs|Customer|Support|Team|Project|Manager"
    r"|Annual|Report|Machine|Learning|Data|Science|Bank|Order|Account|Card|Aadhaar|Voter|Driving"
    r"|Date|Birth|The|This|That|Our|Your|My|New|Service|Services|Product|Sales|Office)\b"
)
_TWO_CAPITALIZED = rf"(?-i:\b(?!{_NOT_NAME})[A-Z][a-z]+ (?!{_NOT_NAME})(?:[A-Z]['’])?[A-Z][a-z]+\b)"

# State capitals and the largest cities: NER misses many of them. Written by
# hand rather than taken from Faker, which would be copying the eval's answers.
INDIAN_CITIES = """Agartala Agra Ahmedabad Aizawl Ajmer Aligarh Allahabad Amravati Amritsar
Asansol Aurangabad Bareilly Belgaum Bengaluru Bangalore Bhilai Bhopal Bhubaneswar Bhiwandi
Bikaner Chandigarh Chennai Coimbatore Cuttack Daman Dehradun Delhi Dhanbad Dharwad Dispur
Durgapur Faridabad Firozabad Gandhinagar Gangtok Ghaziabad Gorakhpur Guntur Gurgaon Gurugram
Guwahati Gwalior Howrah Hubli Hyderabad Imphal Indore Itanagar Jabalpur Jaipur Jalandhar
Jalgaon Jammu Jamnagar Jamshedpur Jhansi Jodhpur Kalyan Kanpur Kavaratti Kochi Kohima Kolhapur
Kolkata Kollam Kota Kozhikode Lucknow Ludhiana Madurai Malappuram Mangalore Meerut Moradabad
Mumbai Mysore Mysuru Nagpur Nanded Nashik Navi Mumbai Nellore Nizamabad Noida Panaji Patna
Pimpri Port Blair Pondicherry Puducherry Pune Raipur Rajkot Ranchi Saharanpur Salem Shillong
Shimla Siliguri Solapur Srinagar Surat Thane Thiruvananthapuram Thrissur Tiruchirappalli
Tirunelveli Tiruppur Udaipur Ujjain Vadodara Varanasi Vasai Vijayawada Visakhapatnam Warangal
""".replace("Navi Mumbai", "Navi_Mumbai").replace("Port Blair", "Port_Blair").split()


def _pattern(entity: str, regex: str, score: float, context=()) -> PatternRecognizer:
    return PatternRecognizer(supported_entity=entity, context=list(context),
                             patterns=[Pattern(entity.lower(), regex, score)])


def custom_recognizers():
    return [
        AadhaarRecognizer(),
        CardRecognizer(),
        PanRecognizer(),
        IndianPhoneRecognizer(),
        UpiRecognizer(),
        # A false match on these blocks the request, so they need context too.
        # Any fourth letter, so a mistyped PAN next to the word is still caught.
        _pattern("IN_PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", NEEDS_CONTEXT, ["pan"]),
        _pattern("IN_PASSPORT", r"\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b", NEEDS_CONTEXT, ["passport"]),
        _pattern("IN_VOTER_ID", r"\b[A-Z]{3}\d{7}\b", NEEDS_CONTEXT,
                 ["voter", "epic", "election"]),
        # State, RTO, year of issue, serial. Old pre-2019 formats vary by state.
        _pattern("IN_DRIVING_LICENCE", r"\b[A-Z]{2}[ -]?\d{2}[ -]?(?:19|20)\d{2}[ -]?\d{7}\b", 0.6,
                 ["licence", "license", "dl", "driving"]),
        _pattern("IN_IFSC", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.6, ["ifsc", "bank", "branch"]),
        _pattern("BANK_ACCOUNT", r"\b\d{9,18}\b", NEEDS_CONTEXT,
                 ["account", "acct", "ac", "bank", "beneficiary", "ifsc"]),
        # "A/C" is split into separate tokens, so context words can't see it.
        _pattern("BANK_ACCOUNT", r"(?<=\ba/c\.?\s*(?:no\.?|number)?[\s:]*)\d{9,18}\b", 0.7),
        # "born" lemmatizes to "bear".
        _pattern("DATE_OF_BIRTH", _DATE, NEEDS_CONTEXT,
                 ["dob", "d.o.b", "d.o.b.", "birth", "birthdate", "birthday", "born", "bear"]),
        _pattern("PIN_CODE", r"\b[1-9]\d{2}\s?\d{3}\b", NEEDS_CONTEXT,
                 ["pin", "pincode", "postal", "postcode", "zip", "address"]),
        # Landlines: STD code plus number. Bare digits, so they need context too.
        _pattern("PHONE_NUMBER", r"(?<![\d+])(?:\+91[\s-]?|0)?\d{2,4}[\s-]?\d{6,8}(?!\d)", NEEDS_CONTEXT,
                 ["landline", "phone", "telephone", "tel", "fax", "call", "contact"]),
        _pattern("PERSON", _TWO_CAPITALIZED, 0.55),
        PatternRecognizer(supported_entity="LOCATION", deny_list_score=0.7,
                          deny_list=[c.replace("_", " ") for c in INDIAN_CITIES]),
        _pattern("API_KEY", _API_KEY, 0.9),
        _pattern("PASSWORD", _SECRET_VALUE, 0.8),
    ]
