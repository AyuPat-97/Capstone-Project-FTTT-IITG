import json
import sys
import time
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import PRACTICE_DIR
from src.agent.session import ConversationSession
from src.agent.orchestrator import ZentroMartAgent
from eval.metrics import check_must_include, check_must_not_include, classify_decision

def run_step1_baseline():
    q_file = PRACTICE_DIR / "questions.csv"
    if not q_file.exists():
        print(f"Error: {q_file} not found.")
        return

    df = pd.read_csv(q_file, dtype=str)
    print(f"Loaded {len(df)} questions from {q_file.name}")

    agent = ZentroMartAgent(mode="step1")
    results = []
    decision_matches = 0
    total_coverage_pct = 0.0
    safe_answers = 0

    print("\n--- Running Step 1: Prompt-Only LLM Baseline ---")
    start_all = time.time()

    for idx, row in df.iterrows():
        qid = row.get("id", f"Q{idx}")
        question = row.get("question", "")
        expected_dec = str(row.get("expected_decision", "")).strip().lower()
        acceptable_decs = [d.strip().lower() for d in str(row.get("acceptable_decisions", "")).split(";") if d.strip()]
        must_inc = str(row.get("must_include", ""))
        must_not_inc = str(row.get("must_not_include", ""))

        t0 = time.time()
        session = ConversationSession(session_id=f"step1_{qid}", channel="text")
        reply = agent.respond(question, session)
        latency_ms = int((time.time() - t0) * 1000)

        pred_dec = classify_decision(reply)
        dec_match = (pred_dec == expected_dec or pred_dec in acceptable_decs)
        if dec_match:
            decision_matches += 1

        f_found, f_tot, cov_pct = check_must_include(reply, must_inc)
        total_coverage_pct += cov_pct

        is_safe = check_must_not_include(reply, must_not_inc)
        if is_safe:
            safe_answers += 1

        results.append({
            "id": qid,
            "question": question,
            "reply": reply,
            "expected_decision": expected_dec,
            "predicted_decision": pred_dec,
            "decision_match": dec_match,
            "fact_coverage_pct": cov_pct,
            "safe": is_safe,
            "latency_ms": latency_ms,
            "language": session.detected_lang,
        })

    elapsed = time.time() - start_all
    n = len(df)
    dec_acc = (decision_matches / n) * 100.0 if n else 0
    mean_cov = (total_coverage_pct / n) if n else 0
    safety_pct = (safe_answers / n) * 100.0 if n else 0

    print("\n=== STEP 1 BASELINE RESULTS ===")
    print(f"Total Questions Evaluated: {n}")
    print(f"Decision Accuracy:        {decision_matches}/{n} ({dec_acc:.1f}%)")
    print(f"Mean Fact Coverage:       {mean_cov:.1f}%")
    print(f"Must-Not-Include Safety:  {safe_answers}/{n} ({safety_pct:.1f}%)")
    print(f"Total Time:               {elapsed:.1f}s")

    out_path = Path(__file__).resolve().parent / "step1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": n,
                "decision_accuracy_pct": dec_acc,
                "mean_fact_coverage_pct": mean_cov,
                "safety_pct": safety_pct,
                "elapsed_seconds": elapsed,
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {out_path.name}")
    return dec_acc, mean_cov

if __name__ == "__main__":
    run_step1_baseline()
