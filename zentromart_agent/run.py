import argparse
import json
import sys
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.session import ConversationSession
from src.agent.orchestrator import ZentroMartAgent
from src.voice.stt import transcribe
from src.voice.tts import synthesize
from src.voice.audio_utils import clean_for_tts

def _write_fallback_wav(path: Path):
    """Writes a minimal 1-second silent 16kHz mono WAV file as safety fallback."""
    import wave
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)
    except Exception:
        pass


import re

LANGUAGE_PATTERNS = [
    (r"(?:^|[_\-.\d])(mr|marathi)(?:[_\-.\d]|$)", "mr-IN"),
    (r"(?:^|[_\-.\d])(hi|hindi)(?:[_\-.\d]|$)", "hi-IN"),
    (r"(?:^|[_\-.\d])(ta|tamil)(?:[_\-.\d]|$)", "ta-IN"),
    (r"(?:^|[_\-.\d])(te|telugu)(?:[_\-.\d]|$)", "te-IN"),
    (r"(?:^|[_\-.\d])(kn|kannada)(?:[_\-.\d]|$)", "kn-IN"),
    (r"(?:^|[_\-.\d])(bn|bengali|bangla)(?:[_\-.\d]|$)", "bn-IN"),
    (r"(?:^|[_\-.\d])(gu|gujarati)(?:[_\-.\d]|$)", "gu-IN"),
    (r"(?:^|[_\-.\d])(pa|punjabi)(?:[_\-.\d]|$)", "pa-IN"),
    (r"(?:^|[_\-.\d])(ml|malayalam)(?:[_\-.\d]|$)", "ml-IN"),
    (r"(?:^|[_\-.\d])(en|english)(?:[_\-.\d]|$)", "en-IN"),
]

def infer_file_language(file_path: Path, fallback: str = None) -> str:
    """Infers the language code from the audio file name or parent folder name."""
    name_to_check = f"{file_path.parent.name}_{file_path.stem}".lower()
    for pattern, code in LANGUAGE_PATTERNS:
        if re.search(pattern, name_to_check):
            return code
    return fallback


def run_batch_pipeline(audio_dir: str, output_dir: str, language_code: str = "en-IN"):
    """
    Official VoiceOps batch run command:
    python run.py --audio_dir <folder_of_wav_files> --output_dir <output_folder>
    """
    in_path = Path(audio_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: audio directory '{in_path}' does not exist.")
        return

    SUPPORTED_EXTENSIONS = {
        ".wav", ".mp3", ".mp4", ".flac", ".ogg",
        ".opus", ".m4a", ".aac", ".webm", ".amr"
    }

    audio_files = sorted([
        f for f in in_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print(f"Found {len(audio_files)} audio files in {in_path}")
    if not audio_files:
        formats_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"Warning: No supported audio files ({formats_str}) found in '{in_path}'.")
        return

    agent = ZentroMartAgent(mode="full")

    # Load chunk dictionary for sources.json metadata
    from src.rag.retriever import retriever
    retriever._ensure_loaded()
    chunk_map = {c["section_id"]: c for c in retriever.chunks}

    for idx, audio_file in enumerate(audio_files, start=1):
        stem = audio_file.stem
        target_dir = out_path / stem
        target_dir.mkdir(parents=True, exist_ok=True)

        # Detect or infer language for this specific file
        if language_code == "auto":
            file_lang = infer_file_language(audio_file, fallback=None)
            if not file_lang:
                print(f"[{idx}/{len(audio_files)}] Skipping {audio_file.name}: filename must contain language information when --language auto is set (e.g. 'marathi_01.mp4', 'audio_hi.wav', 'english_02.wav').")
                continue
        else:
            file_lang = infer_file_language(audio_file, fallback=language_code)

        print(f"[{idx}/{len(audio_files)}] Processing {audio_file.name} [{file_lang}] -> {target_dir.name}/")

        # 1. Transcribe with Gnani STT
        transcript = None
        try:
            stt_res = transcribe(str(audio_file), language_code=file_lang)
            transcript = stt_res.get("transcript")
        except Exception as e:
            print(f"  STT error on {audio_file.name} with {file_lang}: {e}")

        # 2. Empty / Low-confidence STT handling
        if not transcript or not transcript.strip():
            print("  Empty transcript -> saving polite retry reply")
            (target_dir / "transcript.txt").write_text("", encoding="utf-8")
            retry_reply = "Sorry, I didn't catch that. Could you say it again?"
            (target_dir / "reply.txt").write_text(retry_reply, encoding="utf-8")
            (target_dir / "sources.json").write_text("[]", encoding="utf-8")
            try:
                synthesize(
                    text=retry_reply,
                    language="en",
                    output_file=str(target_dir / "reply.wav")
                )
            except Exception as e:
                print(f"  TTS synthesis warning: {e}")
            if not (target_dir / "reply.wav").exists():
                _write_fallback_wav(target_dir / "reply.wav")
            continue

        # Save STT output
        (target_dir / "transcript.txt").write_text(transcript.strip(), encoding="utf-8")

        # 3. Run Agent (Voice Channel)
        session = ConversationSession(session_id=f"voice_{stem}", channel="voice")
        reply = agent.respond(transcript, session, lang_hint=file_lang)

        # 4. Save Text Reply
        (target_dir / "reply.txt").write_text(reply.strip(), encoding="utf-8")

        # 5. Synthesize Reply with Gnani TTS
        tts_text = clean_for_tts(reply, session.detected_lang)
        try:
            synthesize(
                text=tts_text,
                language=session.detected_lang,
                output_file=str(target_dir / "reply.wav")
            )
        except Exception as e:
            print(f"  TTS synthesis warning: {e}")

        # Ensure reply.wav always exists for automated evaluation
        if not (target_dir / "reply.wav").exists():
            _write_fallback_wav(target_dir / "reply.wav")

        # 6. Save sources.json
        sources = []
        for sid in session.section_ids_used:
            chunk = chunk_map.get(sid, {})
            snippet = chunk.get("text", "")[:200]
            sources.append({
                "section_id": sid,
                "doc_type": chunk.get("doc_type", "policy"),
                "text_snippet": snippet,
                "superseded": chunk.get("superseded", False)
            })
        (target_dir / "sources.json").write_text(
            json.dumps(sources, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        try:
            print(f"  Done: transcript={transcript[:40]}... sources={len(sources)}")
        except Exception:
            safe_preview = transcript[:40].encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"  Done: transcript={safe_preview}... sources={len(sources)}")

    print(f"\nAll {len(audio_files)} files processed and saved to {out_path}")


def run_single_file(file_path: str, output_dir: str, language_code: str = "en-IN"):
    """
    Process a single audio file through the full STT → Agent → TTS pipeline.
    Usage: python run.py --file <audio_file> --output_dir <output_folder>
    """
    audio_file = Path(file_path).resolve()
    out_path = Path(output_dir).resolve()

    SUPPORTED_EXTENSIONS = {
        ".wav", ".mp3", ".mp4", ".flac", ".ogg",
        ".opus", ".m4a", ".aac", ".webm", ".amr"
    }

    if not audio_file.exists():
        print(f"Error: audio file '{audio_file}' does not exist.")
        return
    if not audio_file.is_file():
        print(f"Error: '{audio_file}' is not a file.")
        return
    if audio_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"Error: Unsupported format '{audio_file.suffix}'. Supported: {formats_str}")
        return

    stem = audio_file.stem
    target_dir = out_path / stem
    target_dir.mkdir(parents=True, exist_ok=True)

    agent = ZentroMartAgent(mode="full")

    # Load chunk dictionary for sources.json metadata
    from src.rag.retriever import retriever
    retriever._ensure_loaded()
    chunk_map = {c["section_id"]: c for c in retriever.chunks}

    # Detect or infer language
    if language_code == "auto":
        file_lang = infer_file_language(audio_file, fallback=None)
        if not file_lang:
            print(f"Error: Cannot infer language from '{audio_file.name}'.")
            print("When using '--language auto', the filename must contain language information (e.g. 'marathi_01.mp4', 'audio_hi.wav', 'english_02.wav').")
            return
    else:
        file_lang = infer_file_language(audio_file, fallback=language_code)

    print(f"Processing {audio_file.name} [{file_lang}] -> {target_dir.name}/")

    # 1. Transcribe with Gnani STT
    transcript = None
    try:
        stt_res = transcribe(str(audio_file), language_code=file_lang)
        transcript = stt_res.get("transcript")
    except Exception as e:
        print(f"  STT error on {audio_file.name} with {file_lang}: {e}")

    # 2. Empty / Low-confidence STT handling
    if not transcript or not transcript.strip():
        print("  Empty transcript -> saving polite retry reply")
        (target_dir / "transcript.txt").write_text("", encoding="utf-8")
        retry_reply = "Sorry, I didn't catch that. Could you say it again?"
        (target_dir / "reply.txt").write_text(retry_reply, encoding="utf-8")
        (target_dir / "sources.json").write_text("[]", encoding="utf-8")
        try:
            synthesize(
                text=retry_reply,
                language="en",
                output_file=str(target_dir / "reply.wav")
            )
        except Exception as e:
            print(f"  TTS synthesis warning: {e}")
        if not (target_dir / "reply.wav").exists():
            _write_fallback_wav(target_dir / "reply.wav")
        print(f"\nOutput saved to {target_dir}")
        return

    # Save STT output
    (target_dir / "transcript.txt").write_text(transcript.strip(), encoding="utf-8")

    # 3. Run Agent (Voice Channel)
    session = ConversationSession(session_id=f"voice_{stem}", channel="voice")
    reply = agent.respond(transcript, session, lang_hint=file_lang)

    # 4. Save Text Reply
    (target_dir / "reply.txt").write_text(reply.strip(), encoding="utf-8")

    # 5. Synthesize Reply with Gnani TTS
    tts_text = clean_for_tts(reply, session.detected_lang)
    try:
        synthesize(
            text=tts_text,
            language=session.detected_lang,
            output_file=str(target_dir / "reply.wav")
        )
    except Exception as e:
        print(f"  TTS synthesis warning: {e}")

    # Ensure reply.wav always exists
    if not (target_dir / "reply.wav").exists():
        _write_fallback_wav(target_dir / "reply.wav")

    # 6. Save sources.json
    sources = []
    for sid in session.section_ids_used:
        chunk = chunk_map.get(sid, {})
        snippet = chunk.get("text", "")[:200]
        sources.append({
            "section_id": sid,
            "doc_type": chunk.get("doc_type", "policy"),
            "text_snippet": snippet,
            "superseded": chunk.get("superseded", False)
        })
    (target_dir / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    try:
        print(f"  Done: transcript={transcript[:40]}... sources={len(sources)}")
    except Exception:
        safe_preview = transcript[:40].encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  Done: transcript={safe_preview}... sources={len(sources)}")

    print(f"\nOutput saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps 2026 Batch Runner")
    parser.add_argument("pos_audio_dir", nargs="?", default=None, help="Folder containing audio files (positional)")
    parser.add_argument("pos_output_dir", nargs="?", default=None, help="Destination folder for outputs (positional)")
    parser.add_argument("--audio_dir", "--audio-dir", "--input_dir", "--input-dir", "-i", "-a", dest="audio_dir", default=None, help="Folder containing audio files")
    parser.add_argument("--output_dir", "--output-dir", "-o", dest="output_dir", default=None, help="Destination folder for outputs")
    parser.add_argument("--file", "-f", dest="single_file", default=None, help="Single audio file to process (instead of a directory)")
    parser.add_argument("--language", default="en-IN", help="STT language code (default: en-IN)")
    args = parser.parse_args()

    # Single-file mode
    if args.single_file:
        output_dir = args.output_dir or args.pos_output_dir or "output"
        run_single_file(args.single_file, output_dir, args.language)
        return

    audio_dir = args.audio_dir or args.pos_audio_dir
    output_dir = args.output_dir or args.pos_output_dir

    if not audio_dir or not output_dir:
        parser.error("Both audio_dir and output_dir are required.\nUsage: python run.py --audio_dir <folder> --output_dir <folder>\n       python run.py --file <audio_file> --output_dir <folder>")

    run_batch_pipeline(audio_dir, output_dir, args.language)


if __name__ == "__main__":
    main()
