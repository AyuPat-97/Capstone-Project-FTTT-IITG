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
from eval.metrics import (
    check_must_include,
    check_must_not_include,
    classify_decision,
    section_recall,
)

def run_step2_rag():
    q_file = PRACTICE_DIR / "questions.csv"
    if not q_file.exists():
        print(f"Error: {q_file} not found.")
        return

    df = pd.read_csv(q_file, dtype=str)
    print(f"Loaded {len(df)} questions from {q_file.name}")

    agent = ZentroMartAgent(mode="step2")
    results = []
    decision_matches = 0
    total_coverage_pct = 0.0
    safe_answers = 0
    section_hits = 0

    print("\n--- Running Step 2: Document-Grounded RAG Pipeline ---")
    start_all = time.time()

    for idx, row in df.iterrows():
        qid = row.get("id", f"Q{idx}")
        question = row.get("question", "")
        expected_dec = str(row.get("expected_decision", "")).strip().lower()
        acceptable_decs = [d.strip().lower() for d in str(row.get("acceptable_decisions", "")).split(";") if d.strip()]
        must_inc = str(row.get("must_include", ""))
        must_not_inc = str(row.get("must_not_include", ""))
        gold_ev = str(row.get("gold_evidence", ""))

        t0 = time.time()
        session = ConversationSession(session_id=f"step2_{qid}", channel="text")
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

        sec_hit = section_recall(session.section_ids_used, gold_ev)
        if sec_hit:
            section_hits += 1

        results.append({
            "id": qid,
            "question": question,
            "reply": reply,
            "expected_decision": expected_dec,
            "predicted_decision": pred_dec,
            "decision_match": dec_match,
            "fact_coverage_pct": cov_pct,
            "safe": is_safe,
            "section_hit": sec_hit,
            "sections_retrieved": session.section_ids_used,
            "gold_evidence": gold_ev,
            "latency_ms": latency_ms,
            "language": session.detected_lang,
        })

    elapsed = time.time() - start_all
    n = len(df)
    dec_acc = (decision_matches / n) * 100.0 if n else 0
    mean_cov = (total_coverage_pct / n) if n else 0
    safety_pct = (safe_answers / n) * 100.0 if n else 0
    recall_pct = (section_hits / n) * 100.0 if n else 0

    print("\n=== STEP 2 RAG RESULTS ===")
    print(f"Total Questions Evaluated: {n}")
    print(f"Decision Accuracy:        {decision_matches}/{n} ({dec_acc:.1f}%)")
    print(f"Mean Fact Coverage:       {mean_cov:.1f}%")
    print(f"Section Recall @8:        {section_hits}/{n} ({recall_pct:.1f}%)")
    print(f"Must-Not-Include Safety:  {safe_answers}/{n} ({safety_pct:.1f}%)")
    print(f"Total Time:               {elapsed:.1f}s")

    # Load Step 1 baseline numbers if available
    step1_file = Path(__file__).resolve().parent / "step1_results.json"
    if step1_file.exists():
        with open(step1_file, "r", encoding="utf-8") as f:
            s1_data = json.load(f)
            s1_acc = s1_data["summary"]["decision_accuracy_pct"]
            s1_cov = s1_data["summary"]["mean_fact_coverage_pct"]
            print("\n=== STEP 1 vs STEP 2 COMPARISON (Report Deliverable) ===")
            print(f"Metric                 | Step 1 (Prompt-Only) | Step 2 (RAG) | Improvement")
            print(f"-----------------------|----------------------|--------------|------------")
            print(f"Decision Accuracy      | {s1_acc:18.1f}% | {dec_acc:10.1f}% | +{dec_acc - s1_acc:.1f}%")
            print(f"Fact Coverage          | {s1_cov:18.1f}% | {mean_cov:10.1f}% | +{mean_cov - s1_cov:.1f}%")
            print(f"Section Recall @8      |                  N/A | {recall_pct:10.1f}% | N/A")

    out_path = Path(__file__).resolve().parent / "step2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": n,
                "decision_accuracy_pct": dec_acc,
                "mean_fact_coverage_pct": mean_cov,
                "section_recall_pct": recall_pct,
                "safety_pct": safety_pct,
                "elapsed_seconds": elapsed,
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {out_path.name}")
    return dec_acc, mean_cov, recall_pct

if __name__ == "__main__":
    run_step2_rag()
