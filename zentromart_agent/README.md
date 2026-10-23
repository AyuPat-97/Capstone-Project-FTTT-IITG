Github URL : https://github.com/AyuPat-97/Capstone-Project-FTTT-IITG.git

# ZentroMart VoiceOps Agent

AI-powered multimodal customer support voice and text agent for **ZentroMart** (an Indian e-commerce company), handling order management and CRM/helpdesk support across 10 Indian languages.

Built for the **IIT Guwahati Consulting & Analytics Club × Gnani.ai VoiceOps 2026 Capstone Challenge**.

---

## 🚀 Architecture Highlights

- **Multilingual Ears & Voice**: Native STT & TTS powered by **Gnani.ai Speech Models** (`gnani-vachana` SDK with `timbre-v2.5` / `timbre-v3.0`).
- **Hybrid Document Grounding**: Dense vector retrieval (**BAAI/bge-m3** with FAISS L2-normalized cosine index) + Sparse lexical retrieval (**BM25Okapi**) combined via **Reciprocal Rank Fusion (RRF, k=60)** with temporal policy version-filtering (pre/post August 1, 2026 windows).
- **Deterministic Security & Verification Gate**: Caller identity verification state machine directly implemented in application code (requires last 4 digits of phone or 6-digit delivery pincode before order details are revealed; max 1 retry).
- **Hard Guardrails & Escalation**: Deterministic checks for high-value damage (> ₹5,000), safety hazards (swelling batteries, fires), repeat contacts ( ≥ 2 in past 7 days ), overdue refunds, and legal threats.
- **Accurate Business Math**: Strict Monday–Saturday business day arithmetic minus warehouse holiday closures; partial return and coupon discount math (REF-4).

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Synthetic Dataset directory `Capstone_Project_Dataset_FTTT_2026/` placed as a sibling to `zentromart_agent/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
In `.env`:
```env
GNANI_API_KEY=your_gnani_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Build RAG Index
Index the 21 knowledge base markdown policies and product catalog:
```bash
python src/rag/ingest.py
```
*(Expected output: `Indexed 154 chunks from 22 documents`)*

---

## 🧪 Evaluation Harness

Run the evaluation steps locally:
```bash
# Step 1: Prompt-only baseline on 52 practice questions
python eval/step1_baseline.py

# Step 2: Full RAG-grounded evaluation on practice questions (outputs side-by-side comparison)
python eval/step2_rag.py

# Step 3: Multi-turn conversation and tool calling evaluation (23 conversations, 96 turns)
python eval/step3_conversations.py
```

---

## ⚡ Run Audio Pipeline (Batch & Single File)

### Batch Execution (Organizer Submission Command)
The automated grading pipeline executes on a folder:
```bash
# Multi-language mode: Filenames must contain language information (example: marathi_01.mp4)
python run.py --audio_dir <folder_of_audio_files> --output_dir <output_folder> --language auto




# Standard flagged command (for English-only audio inputs)
python run.py --audio_dir <folder_of_audio_files> --output_dir <output_folder>
# Or positional syntax
python run.py <folder_of_audio_files> <output_folder>
```

### Single Audio File Execution
To process a single audio file directly through the STT → Agent → TTS pipeline:
```bash
# Multi-language mode via filename language tag (e.g. marathi_01.mp4)
python run.py -f <path_to_audio_file> -o <output_folder> --language auto

# Explicit regional language code
python run.py -f <path_to_audio_file> -o <output_folder> --language hi-IN

# Standard command (for English-only audio input)
python run.py --file <path_to_audio_file> --output_dir <output_folder>

# Short form (output defaults to output/ if --output_dir is omitted)
python run.py -f <path_to_audio_file> -o <output_folder>
```

### Features:
- **All 10 Official Gnani Audio Formats**: Supports `.wav`, `.mp3`, `.mp4`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.webm`, `.amr` (case-insensitive).
- **Filename-Based Automatic Language Routing**: When `--language auto` is set, the pipeline inspects the filename for language tokens (e.g. `marathi`, `_mr`, `hindi`, `_hi`, `tamil`, `_ta`, `english`, `_en`) and routes directly to the corresponding Indian language model (`mr-IN`, `hi-IN`, `ta-IN`, etc.). Files are required to have language information in their names when using `--language auto`.
- **Fail-Safe Generation**: Always produces all 4 artifacts per audio file:
  - `transcript.txt` — Gnani STT speech transcription
  - `reply.txt` — Grounded agent text reply in the customer's native language
  - `reply.wav` — Natural synthesized voice reply via Gnani TTS (voice="Pranav")
  - `sources.json` — Section IDs, policy types, and snippets grounding the reply

---

## 🧪 Unified Regression & Unit Tests

Run all unit tests in a single command:
```bash
python tests/run_all.py
```
Validates:
1. Business day calendar & holiday arithmetic (skipping Sundays and 19 October 2026 holiday).
2. Identity verification gating & PII withholding (4-digit phone suffix / 6-digit delivery pincode).
3. Hybrid RAG retrieval, BM25+FAISS RRF, and temporal policy version filtering.

---

## 🖥️ Interactive Web UI

Launch the Gradio interface featuring both Text Chat and Voice I/O:
```bash
cd zentromart_agent
python ui/app.py
```
Visit `http://localhost:7860` in your web browser.

---

## 📁 Repository Structure

```
zentromart_agent/
├── README.md                  # Complete architecture, setup & execution guide
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
├── run.py                     # Voice evaluation runner (batch & single-file, 10 audio formats)
├── config.py                  # Fixed reference date (2026-10-20), threshold (₹5,000) & paths
├── src/
│   ├── rag/
│   │   ├── ingest.py          # Markdown & product chunker + FAISS & BM25 builder
│   │   ├── retriever.py       # Dense + sparse RRF fusion & version filter
│   │   └── rewriter.py        # Customer vocab -> document vocab translator
│   ├── data_layer/
│   │   ├── loader.py          # CSV reader (dtype=str)
│   │   ├── pincode_lookup.py  # Pincode serviceability & COD lookup
│   │   ├── product_lookup.py  # Product SKU catalog lookup
│   │   ├── business_days.py   # Mon-Sat business day calendar & holiday math
│   │   └── refund_calc.py     # REF-4 partial refund & coupon discount math
│   ├── agent/
│   │   ├── session.py         # In-memory ConversationSession dataclass
│   │   ├── system_prompt.py   # Dynamic per-turn system prompt builder (native multilingual directives)
│   │   ├── tools_wrapper.py   # Function declarations & tool executors
│   │   ├── orchestrator.py    # Main agent loop (text & voice channels)
│   │   └── guardrails.py      # Deterministic escalation triggers (> ₹5,000, safety, legal, repeats)
│   └── voice/
│       ├── stt.py             # Gnani STT client wrapper (supports 10 Indian languages)
│       ├── tts.py             # Gnani TTS client wrapper (voice="Pranav", model="timbre-v2.5")
│       └── audio_utils.py     # Spoken digit normalizer & TTS text cleaner
├── ui/
│   └── app.py                 # Gradio interactive web app (Chat + Voice)
├── eval/
│   ├── step1_baseline.py      # Prompt-only baseline evaluator
│   ├── step2_rag.py           # RAG evaluator & comparison reporter
│   ├── step3_conversations.py # Multi-turn conversations evaluator
│   └── metrics.py             # Fact coverage, decision match & recall metrics
└── tests/
    ├── run_all.py             # Unified test runner
    ├── test_business_days.py
    ├── test_retriever.py
    └── test_verification_gate.py
```
