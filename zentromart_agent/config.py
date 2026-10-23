import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Automatically load environment variables from .env
load_dotenv(BASE_DIR / ".env")
load_dotenv()

# Fixed dataset date — NEVER use datetime.now() or date.today() in business logic
TODAY = date(2026, 10, 20)

# Check potential locations for dataset
_candidates = [
    Path(os.environ.get("DATASET_DIR", "")),
    BASE_DIR.parent / "Capstone_Project_Dataset_FTTT_2026",
    BASE_DIR.parent.parent / "Capstone_Project_Dataset_FTTT_2026",
    BASE_DIR / "Capstone_Project_Dataset_FTTT_2026",
    Path.cwd() / "Capstone_Project_Dataset_FTTT_2026",
    Path("c:/Users/DELL/OneDrive/Desktop/Capstone_project/Capstone_Project_Dataset_FTTT_2026"),
]

DATASET_DIR = None
for c in _candidates:
    try:
        if c and c.exists():
            DATASET_DIR = c.resolve()
            break
    except Exception:
        pass

if not DATASET_DIR:
    DATASET_DIR = (BASE_DIR.parent / "Capstone_Project_Dataset_FTTT_2026").resolve()

DATA_DIR = DATASET_DIR / "data"
KB_DIR = DATASET_DIR / "kb"
TOOLS_DIR = DATASET_DIR / "tools"
PRACTICE_DIR = DATASET_DIR / "practice"
RAG_CACHE = BASE_DIR / ".rag_cache"

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_FALLBACK = "BAAI/bge-small-en-v1.5"
GEMINI_PRIMARY = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.5-flash-lite")
GEMINI_LITE = os.environ.get("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")
GNANI_TTS_MODEL = "timbre-v2.5"  # ALWAYS name explicitly
GNANI_TTS_VOICE_DEFAULT = "Pranav"  # Default voice per preference

# RAG
RETRIEVAL_TOP_K = 8
RRF_K = 60
MAX_CTX_TOKENS = 1500

# Agent
MAX_TOKENS_CHAT = 512
MAX_TOKENS_VOICE = 256
MAX_TOOL_ROUNDS = 3
VERIFY_MAX_TRIES = 1   # one retry = 2 total attempts
HIGH_VALUE_THRESHOLD = 5000  # High-value escalation threshold in INR
