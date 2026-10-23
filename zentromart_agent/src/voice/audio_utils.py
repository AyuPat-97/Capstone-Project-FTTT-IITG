import re

DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",

    # Hindi / Hinglish spoken numbers (Latin + Devanagari)
    "ek": "1", "do": "2", "teen": "3", "chaar": "4", "char": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "che": "6", "saat": "7",
    "aath": "8", "ath": "8", "nau": "9", "shunya": "0",
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5",
    "पांच": "5", "छह": "6", "सात": "7", "आठ": "8", "नौ": "9", "शून्य": "0",

    # Marathi (Latin + Devanagari)
    "don": "2", "tin": "3", "paach": "5", "pach": "5", "saha": "6",
    "sat": "7", "naoo": "9",
    "दोन": "2", "पाच": "5", "सहा": "6", "नऊ": "9",

    # Bengali (bn)
    "shunyo": "0", "dui": "2", "chhoy": "6", "noy": "9",
    "শূন্য": "0", "এক": "1", "দুই": "2", "তিন": "3", "চার": "4",
    "পাঁচ": "5", "ছয়": "6", "সাত": "7", "আট": "8", "নয়": "9",

    # Gujarati (gu)
    "be": "2", "tran": "3", "chha": "6", "nav": "9", "meendu": "0",
    "શૂન્ય": "0", "મીંડું": "0", "એક": "1", "બે": "2", "ત્રણ": "3",
    "ચાર": "4", "પાંચ": "5", "છ": "6", "સાત": "7", "આઠ": "8", "નવ": "9",

    # Kannada (kn)
    "sonne": "0", "ondu": "1", "eradu": "2", "mooru": "3", "naalku": "4",
    "aidu": "5", "aaru": "6", "elu": "7", "entu": "8", "ombattu": "9",
    "ಸೊನ್ನೆ": "0", "ಒಂದು": "1", "ಎರಡು": "2", "ಮೂರು": "3", "ನಾಲ್ಕು": "4",
    "ಐದು": "5", "ಆರು": "6", "ಏಳು": "7", "ಎಂಟು": "8", "ಒಂಬತ್ತು": "9",

    # Malayalam (ml)
    "poojyam": "0", "onnu": "1", "randu": "2", "moonnu": "3", "naalu": "4",
    "anju": "5", "ezhu": "7", "ettu": "8", "ombathu": "9",
    "പൂജ്യം": "0", "ഒന്ന്": "1", "രണ്ട്": "2", "മൂന്ന്": "3", "നാല്": "4",
    "അഞ്ച്": "5", "ആറ്": "6", "ഏഴ്": "7", "എട്ട്": "8", "ഒമ്പത്": "9",

    # Punjabi (pa)
    "sifar": "0", "ikk": "1", "tinn": "3", "panj": "5", "satt": "7", "atth": "8",
    "ਸਿਫ਼ਰ": "0", "ਸ਼ੂੰਨ": "0", "ਇੱਕ": "1", "ਦੋ": "2", "ਤਿੰਨ": "3",
    "ਚਾਰ": "4", "ਪੰਜ": "5", "ਛੇ": "6", "ਸੱਤ": "7", "ਅੱਠ": "8", "ਨੌਂ": "9",

    # Tamil (ta)
    "poojyam": "0", "poojjiyam": "0", "saibar": "0", "ondru": "1", "irandu": "2",
    "moondru": "3", "naangu": "4", "aindhu": "5", "onbadhu": "9",
    "பூஜ்ஜியம்": "0", "பூஜ்யம்": "0", "சைபர்": "0",
    "ஒன்று": "1", "ஒன்னு": "1", "இரண்டு": "2", "ரெண்டு": "2",
    "மூன்று": "3", "மூணு": "3", "நான்கு": "4", "நாலு": "4",
    "ஐந்து": "5", "அஞ்சு": "5", "ஆறு": "6", "ஏழு": "7", "எட்டு": "8", "ஒன்பது": "9",

    # Telugu (te)
    "sunna": "0", "okati": "1", "moodu": "3", "naalugu": "4",
    "yedu": "7", "enimidi": "8", "tommidi": "9",
    "సున్నా": "0", "శూన్యం": "0", "ఒకటి": "1", "రెండు": "2", "మూడు": "3",
    "నాలుగు": "4", "ఐదు": "5", "ఆరు": "6", "ఏడు": "7",
    "ఎనిమిది": "8", "తొమ్మిది": "9",
}

UNICODE_DIGITS = {
    # Devanagari (Hindi, Marathi)
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    # Bengali
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
    # Gujarati
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
    # Gurmukhi (Punjabi)
    "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
    "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
    # Kannada
    "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
    "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
    # Malayalam
    "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
    "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
    # Tamil
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
    # Telugu
    "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
    "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
}

MONTH_NAMES_EN = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTH_NAMES_HI = [
    "", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"
]

MONTH_NAMES_MR = [
    "", "जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
    "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"
]

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def normalize_spoken_digits(text: str) -> str:
    """
    Convert spoken digit sequences and Indian numerals to standard ASCII digits,
    and collapse spaced digit groups (e.g. '8978 2837' -> '89782837').
    Preserves Indic vowel signs (matras) and combining characters during normalization.
    """
    if not text:
        return ""

    res = text
    # 1. Unicode Indian digits
    for k, v in UNICODE_DIGITS.items():
        res = res.replace(k, v)

    # 2. Digit words to digits (strip punctuation while preserving full Unicode combining marks)
    words = res.split()
    converted_words = []
    for w in words:
        clean_w = w.strip(".,!?:;\"'()[]{}«»-").lower()
        if clean_w in DIGIT_WORDS:
            converted_words.append(DIGIT_WORDS[clean_w])
        else:
            converted_words.append(w)
    res = " ".join(converted_words)

    # 3. Collapse spaced sequences of 4+4 digits or isolated single digits
    # E.g. "8 9 7 8 2 8 3 7" -> "89782837"
    # E.g. "8978 2837" -> "89782837"
    res = re.sub(r"(?<=\b\d{4})\s+(?=\d{4}\b)", "", res)
    for _ in range(8):
        res = re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", res)

    return res

def clean_for_tts(text: str, lang: str = "en") -> str:
    """
    Prepare agent reply text for Gnani TTS:
    - Remove markdown formatting
    - Replace currency symbol with words ('499 rupees' or '499 रुपये')
    - Format YYYY-MM-DD dates for natural speech ('the 27th of October 2026' or '27 ऑक्टोबर 2026')
    - Space 8-digit order IDs into 4+4 groups ('8978 2837')
    - Remove section ID references like '[RET-2]'
    """
    if not text:
        return ""

    out = text

    # Remove section citations e.g. [RET-2], [ESC-1], [products.csv:ELE-0012]
    out = re.sub(r"\[[A-Za-z0-9_.:\-]+\]", "", out)

    # Replace currency symbols
    if lang.startswith("hi") or lang.startswith("mr"):
        out = re.sub(r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 रुपये", out)
        out = re.sub(r"\bRs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 रुपये", out, flags=re.IGNORECASE)
        out = re.sub(r"\bINR\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 रुपये", out, flags=re.IGNORECASE)
    else:
        out = re.sub(r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 rupees", out)
        out = re.sub(r"\bRs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 rupees", out, flags=re.IGNORECASE)
        out = re.sub(r"\bINR\s*(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 rupees", out, flags=re.IGNORECASE)

    # Format YYYY-MM-DD dates
    def _format_date(m):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            if lang.startswith("mr"):
                return f"{day} {MONTH_NAMES_MR[month]} {year}"
            if lang.startswith("hi"):
                return f"{day} {MONTH_NAMES_HI[month]} {year}"
            return f"the {ordinal(day)} of {MONTH_NAMES_EN[month]} {year}"
        return m.group(0)

    out = re.sub(r"\b(20\d{2})-(\d{2})-(\d{2})\b", _format_date, out)

    # Format 8-digit order IDs to spaced groups of 4 for clear TTS readout
    # E.g. "89782837" -> "8978 2837"
    out = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", out)

    # Remove markdown symbols
    out = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", out)  # bold/italic
    out = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", out)    # code
    out = re.sub(r"^#+\s*", "", out, flags=re.MULTILINE) # headings
    out = re.sub(r"^\s*[-*•]\s+", "", out, flags=re.MULTILINE) # bullets
    out = re.sub(r"^\s*\d+\.\s+", "", out, flags=re.MULTILINE) # numbers

    # Normalize whitespace
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n+", ". ", out)
    out = re.sub(r"\n", " ", out)

    return out.strip()
