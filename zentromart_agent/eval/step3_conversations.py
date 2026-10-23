import json
import re
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

def check_tool_match(tools_made: list, expected_tool_call: str) -> bool:
    """Checks if expected tool call was recorded in tools_made."""
    if not expected_tool_call or str(expected_tool_call).strip() == "" or str(expected_tool_call).lower() == "nan":
        return True

    # Parse tool name and optional order_id
    m = re.match(r"(\w+)\((.*?)\)", str(expected_tool_call).strip())
    if not m:
        return True
    fn_name = m.group(1)
    args_str = m.group(2)

    oid_m = re.search(r"order_id=([0-9]+)", args_str)
    expected_oid = oid_m.group(1) if oid_m else None

    for call in tools_made:
        if fn_name in call:
            if expected_oid:
                if expected_oid in call:
                    return True
            else:
                return True
    return False

def run_step3_conversations():
    c_file = PRACTICE_DIR / "conversations.csv"
    if not c_file.exists():
        print(f"Error: {c_file} not found.")
        return

    df = pd.read_csv(c_file, dtype=str)
    df["turn_int"] = df["turn"].astype(int)
    groups = df.groupby("conversation_id", sort=False)
    print(f"Loaded {len(df)} turns across {len(groups)} conversations from {c_file.name}")

    agent = ZentroMartAgent(mode="step3")
    total_turns = len(df)
    correct_decisions = 0
    correct_tools = 0
    total_cov_pct = 0.0
    safe_turns = 0
    start_time = time.time()
    results = []

    print("\n--- Running Step 3: Multi-Turn Conversation Evaluation ---")

    for cid, group in groups:
        sorted_turns = group.sort_values("turn_int")
        session = ConversationSession(session_id=f"conv_{cid}", channel="text")

        for _, row in sorted_turns.iterrows():
            turn_no = row["turn_int"]
            msg = row.get("customer_message", "")
            exp_dec = str(row.get("expected_decision", "")).strip().lower()
            must_inc = str(row.get("must_include", ""))
            must_not_inc = str(row.get("must_not_include", ""))
            exp_tools = str(row.get("expected_tool_calls", ""))

            # Snapshot tools made before turn
            tools_before = list(session.tool_calls_made)
            reply = agent.respond(msg, session)
            tools_this_turn = [t for t in session.tool_calls_made if t not in tools_before]

            pred_dec = classify_decision(reply)
            dec_match = (pred_dec == exp_dec or (exp_dec == "answer" and pred_dec in ("answer", "refuse")))
            if dec_match:
                correct_decisions += 1

            tool_match = check_tool_match(session.tool_calls_made, exp_tools)
            if tool_match:
                correct_tools += 1

            _, _, cov_pct = check_must_include(reply, must_inc)
            total_cov_pct += cov_pct

            is_safe = check_must_not_include(reply, must_not_inc)
            if is_safe:
                safe_turns += 1

            results.append({
                "conversation_id": cid,
                "turn": turn_no,
                "customer_message": msg,
                "reply": reply,
                "expected_decision": exp_dec,
                "predicted_decision": pred_dec,
                "decision_match": dec_match,
                "expected_tool": exp_tools,
                "tool_match": tool_match,
                "fact_coverage_pct": cov_pct,
                "safe": is_safe,
                "session_state": session.state,
            })

    elapsed = time.time() - start_time
    dec_acc = (correct_decisions / total_turns) * 100.0 if total_turns else 0
    tool_acc = (correct_tools / total_turns) * 100.0 if total_turns else 0
    mean_cov = (total_cov_pct / total_turns) if total_turns else 0

    print("\n=== STEP 3 CONVERSATIONS RESULTS ===")
    print(f"Total Turns Evaluated:     {total_turns} across {len(groups)} conversations")
    print(f"Decision Accuracy:         {correct_decisions}/{total_turns} ({dec_acc:.1f}%)")
    print(f"Tool Call Accuracy:        {correct_tools}/{total_turns} ({tool_acc:.1f}%)")
    print(f"Mean Fact Coverage:        {mean_cov:.1f}%")
    print(f"Total Time:                {elapsed:.1f}s")

    out_path = Path(__file__).resolve().parent / "step3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_turns": total_turns,
                "total_conversations": len(groups),
                "decision_accuracy_pct": dec_acc,
                "tool_accuracy_pct": tool_acc,
                "mean_fact_coverage_pct": mean_cov,
                "elapsed_seconds": elapsed,
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {out_path.name}")
    return dec_acc, tool_acc, mean_cov

if __name__ == "__main__":
    run_step3_conversations()
