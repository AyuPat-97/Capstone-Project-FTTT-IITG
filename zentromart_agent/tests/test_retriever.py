import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.retriever import retrieve
from src.agent.session import ConversationSession

def test_retrieval():
    queries = [
        "How many days to return a phone or laptop?",
        "My delivery person did not show up for pickup",
        "Can I pay with cash on delivery?"
    ]

    for q in queries:
        chunks = retrieve(q, top_k=3)
        sids = [c["section_id"] for c in chunks]
        print(f"Query: '{q}'")
        print(f"  Top-3 Section IDs: {sids}")
        assert len(chunks) > 0, f"Expected chunks for query '{q}'"

    # Test version filter
    s_old = ConversationSession(session_id="test_old", order_data={"delivered_on": "2026-07-20"})
    old_chunks = retrieve("electronics return window", session=s_old, top_k=5)
    old_sids = [c["section_id"] for c in old_chunks]
    print(f"Old order (2026-07-20) electronics return query sections: {old_sids}")

    s_new = ConversationSession(session_id="test_new", order_data={"delivered_on": "2026-09-01"})
    new_chunks = retrieve("electronics return window", session=s_new, top_k=5)
    new_sids = [c["section_id"] for c in new_chunks]
    print(f"New order (2026-09-01) electronics return query sections: {new_sids}")

    print("ALL RETRIEVER TESTS PASSED!")

if __name__ == "__main__":
    test_retrieval()
