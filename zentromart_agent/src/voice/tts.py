import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import GNANI_TTS_MODEL, GNANI_TTS_VOICE_DEFAULT

# Language -> BCP-47 tag for Gnani
LANG_TO_CODE = {
    "en": "en-IN",
    "hi": "hi-IN",
    "hi-en": "hi-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "mr-en": "mr-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

_tts_client = None

def get_tts_client():
    global _tts_client
    if _tts_client is None:
        from gnani.tts import GnaniTTSClient
        _tts_client = GnaniTTSClient()
    return _tts_client

def synthesize(
    text: str,
    language: str,
    output_file: str,
    voice: str = GNANI_TTS_VOICE_DEFAULT,
    model: str = GNANI_TTS_MODEL
) -> str:
    """
    Synthesize text to speech using Gnani TTS.
    Always pass model explicitly ('timbre-v2.5' or 'timbre-v3.0').
    language: language code key in LANG_TO_CODE (e.g. en, hi, ta).
    """
    client = get_tts_client()
    lang_code = LANG_TO_CODE.get(language, "en-IN")
    client.synthesize(
        text,
        voice=voice,
        model=model,
        language=lang_code,
        output_file=output_file,
    )
    return output_file
