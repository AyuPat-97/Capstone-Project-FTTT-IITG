import sys
from pathlib import Path
from typing import Dict, Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_stt_client = None

def get_stt_client():
    global _stt_client
    if _stt_client is None:
        from gnani.stt import GnaniSTTClient
        _stt_client = GnaniSTTClient()
    return _stt_client

def transcribe(audio_path: str, language_code: str = "en-IN") -> Dict[str, Any]:
    """
    Transcribe audio_path (WAV, <= 60 s) with Gnani STT.
    Returns {"transcript": str, "language_code": str, "confidence": float | None}.
    If transcript is empty or whitespace-only, returns {"transcript": None}.
    language_code accepts any of: en-IN, hi-IN, bn-IN, gu-IN, kn-IN,
    ml-IN, mr-IN, pa-IN, ta-IN, te-IN.
    en-IN also handles English mixed with Hindi (returns Latin script).
    """
    client = get_stt_client()
    result = client.transcribe(audio_path, language_code=language_code)
    transcript = result.get("transcript", "").strip() if result else ""
    if not transcript:
        return {"transcript": None, "language_code": language_code, "confidence": None}
    return {
        "transcript": transcript,
        "language_code": result.get("language_code", language_code),
        "confidence": result.get("confidence"),
    }
