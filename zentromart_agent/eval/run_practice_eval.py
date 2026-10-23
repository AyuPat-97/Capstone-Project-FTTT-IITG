import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path("c:/Users/DELL/OneDrive/Desktop/Capstone_project/zentromart_agent").resolve()
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

def run_practice_eval(limit=None):
    q_file = PRACTICE_DIR / "questions.csv"
    with open(q_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    agent = ZentroMartAgent(mode="full")

    print(f"Loaded {len(rows)} questions from {q_file.name}")
    print("=" * 80)

    results = []
    decision_matches = 0
    safe_matches = 0
    total_coverage = 0.0
    section_hits = 0
    errors = []

    for idx, row in enumerate(rows, 1):
        qid = row.get("id")
        q_text = row.get("question")
        exp_dec = (row.get("expected_decision") or "").strip().lower()
        acc_decs = [d.strip().lower() for d in (row.get("acceptable_decisions") or "").split(";") if d.strip()]
        gold_ev = row.get("gold_evidence") or ""
        hard_neg = row.get("hard_negatives") or ""
        must_inc = row.get("must_include") or ""
        must_not_inc = row.get("must_not_include") or ""
        ref_ans = row.get("reference_answer") or ""
        exp_lang = (row.get("reply_language") or "en").strip()

        session = ConversationSession(session_id=f"eval_{qid}", channel="text")
        t0 = time.time()
        try:
            reply = agent.respond(q_text, session)
        except Exception as e:
            reply = f"ERROR: {e}"
        latency = int((time.time() - t0) * 1000)

        # 1. Decision check
        pred_dec = classify_decision(reply)
        dec_ok = (pred_dec == exp_dec or pred_dec in acc_decs)
        if dec_ok:
            decision_matches += 1

        # 2. Must-include check
        f_found, f_tot, cov_pct = check_must_include(reply, must_inc)
        total_coverage += cov_pct

        # 3. Must-not-include check
        safe_ok = check_must_not_include(reply, must_not_inc)
        if safe_ok:
            safe_matches += 1

        # 4. Section hit
        sec_ok = section_recall(session.section_ids_used, gold_ev)
        if sec_ok:
            section_hits += 1

        # Check errors / flags
        q_errors = []
        if not dec_ok:
            q_errors.append(f"Decision mismatch (predicted={pred_dec}, expected={exp_dec})")
        if cov_pct < 100.0 and f_tot > 0:
            q_errors.append(f"Missing must_include ({f_found}/{f_tot} facts, missing from: {must_inc})")
        if not safe_ok:
            q_errors.append(f"Contained forbidden must_not_include ({must_not_inc})")
        if not sec_ok and gold_ev:
            q_errors.append(f"Retriever missed gold section ({gold_ev})")

        res_entry = {
            "id": qid,
            "question": q_text,
            "reply": reply,
            "expected_decision": exp_dec,
            "predicted_decision": pred_dec,
            "decision_ok": dec_ok,
            "must_include": must_inc,
            "coverage_pct": cov_pct,
            "must_not_include": must_not_inc,
            "safe": safe_ok,
            "gold_evidence": gold_ev,
            "section_hit": sec_ok,
            "sections_used": list(session.section_ids_used),
            "reference_answer": ref_ans,
            "detected_lang": session.detected_lang,
            "expected_lang": exp_lang,
            "errors": q_errors,
            "latency_ms": latency
        }
        results.append(res_entry)

        status_flag = "PASS" if not q_errors else "FAIL"
        print(f"[{idx:02d}/{len(rows):02d}] {qid} ({status_flag}) - {q_text[:60]}")
        if q_errors:
            errors.append(res_entry)
            for err in q_errors:
                print(f"   -> {err}")
            print(f"   Reply: {reply[:120]}...")
            print(f"   Ref:   {ref_ans[:120]}...")

        # Pace requests to respect 15 RPM Free Tier limit
        time.sleep(4.5)

    # Summary
    n = len(rows)
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY ({n} Questions):")
    print(f"Decision Accuracy: {decision_matches}/{n} ({decision_matches/n*100:.1f}%)")
    print(f"Mean Fact Coverage: {total_coverage/n:.1f}%")
    print(f"Safety (Must-Not-Include): {safe_matches}/{n} ({safe_matches/n*100:.1f}%)")
    print(f"Section Recall: {section_hits}/{n} ({section_hits/n*100:.1f}%)")
    print(f"Total Questions with Flaws: {len(errors)}/{n}")
    print("=" * 80)

    # Save results to file
    out_file = _ROOT / "eval" / "practice_questions_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as out_f:
        json.dump({
            "summary": {
                "total": n,
                "decision_accuracy": decision_matches / n * 100,
                "mean_fact_coverage": total_coverage / n,
                "safety_rate": safe_matches / n * 100,
                "section_recall": section_hits / n * 100,
                "error_count": len(errors)
            },
            "errors": errors,
            "all_results": results
        }, out_f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {out_file}")

if __name__ == "__main__":
    run_practice_eval()
