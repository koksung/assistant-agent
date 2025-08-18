from __future__ import annotations
from dotenv import dotenv_values, load_dotenv
load_dotenv()

import argparse
import asyncio
import io
import json
import os
from typing import Any, Dict, Optional

from fastapi import UploadFile

from app.llm.setup import LLMSetup
from app.orchestrator import prepare_user_task, conversation_loop, parse_plan
from app.tool_registry import initialize_tool_registry, set_global_registry
from app.users.session import UserSessionManager


os.environ.setdefault("MPLBACKEND", "Agg")
api_key = dotenv_values().get("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


def _pretty_beliefs(belief_state: Dict[str, Any]) -> str:
    belief = (belief_state or {}).get("belief", {})
    probs = belief.get("archetype_probs", {}) or {}
    prefs = belief.get("preferences", {}) or {}
    rc = (belief_state or {}).get("routing_ctx", {})
    lines = ["Archetype probabilities:"]
    for k, v in probs.items():
        lines.append(f"  - {k}: {v:.2f}")
    lines.append("Preferences:")
    for k, v in prefs.items():
        lines.append(f"  - {k}: {v}")
    if rc:
        lines.append("Routing ctx:")
        if rc.get("archetypes"):
            lines.append(f"  - archetypes: {rc['archetypes']}")
        if rc.get("preferences"):
            lines.append(f"  - prefs: {rc['preferences']}")
    return "\n".join(lines)


def _last_plan_from_session(session) -> Optional[str]:
    # orchestrator stores the plan as: session.add_message("orchestrator", plan)
    for msg in reversed(session.conversation_history):
        if msg.get("role") == "orchestrator":
            return str(msg.get("content", "")).strip()
    return None


def _print_plan(plan: Optional[str]) -> None:
    if not plan:
        print("Plan: <none>")
        return
    print("Plan (raw):")
    print(plan)
    print("\nParsed steps:")
    for i, step in enumerate(parse_plan(plan), 1):
        print(f"  {i}. token={step.get('token')}  input_target={step.get('input_target')}")


async def main():
    parser = argparse.ArgumentParser(description="Two-round smoke test for agentic LLM orchestrator.")
    parser.add_argument("--pdf", default="data/ddpm-short.pdf", help="Path to the PDF (round 1).")
    parser.add_argument("--q1", default="Give me an initial high-level summary of this paper.",
                        help="Round 1 user query.")
    parser.add_argument("--q2", default="Summarize methodology and key equations.",
                        help="Round 2 user query (follow-up).")
    parser.add_argument("--q3", default="Now focus on the math details and derivations. "
                                        "I prefer concise math-first summaries focusing on derivations; I’m familiar with diffusion models.",
                        help="Round 3 user query (follow-up).")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    # ---- 1) Init registry + LLMs (same config you use in main.py) ----------
    registry = initialize_tool_registry()
    set_global_registry(registry)

    llms = {
        "summarizer":   LLMSetup("summarizer_llm",   temperature=0.3).get_llm(),
        "orchestrator": LLMSetup("orchestrator_llm", temperature=0.7).get_llm(),
        "belief_updater": LLMSetup("belief_updater_llm", temperature=0.4).get_llm(),
    }

    # ---- 2) Create a session ------------------------------------------------
    session_manager = UserSessionManager()
    session = session_manager.get_or_create_session(user_id="smoke_user_1")

    # ---- 3) Round 1: with file + q1 ----------------------------------------
    print("=== ROUND 1 ===")
    with open(args.pdf, "rb") as f:
        file_bytes = f.read()
    upload = UploadFile(filename=os.path.basename(args.pdf), file=io.BytesIO(file_bytes))

    result1 = await prepare_user_task(upload, user_query=args.q1, session=session, llms=llms)

    print("\n--- Beliefs AFTER Round 1 ---")
    print(_pretty_beliefs(session.belief_state))

    plan1 = _last_plan_from_session(session)
    print("\n--- Orchestrator Plan Round 1 ---")
    _print_plan(plan1)

    print("\n--- Summarizer Output Round 1 (truncated) ---")
    s1 = str(result1)
    print(s1[:800] + ("..." if len(s1) > 800 else ""))

    # ---- 4) Round 2: follow-up q2 (no file) --------------------------------
    print("\n=== ROUND 2 ===")
    result2 = await conversation_loop(args.q2, session.task, session, llms)

    print("\n--- Beliefs AFTER Round 2 ---")
    print(_pretty_beliefs(session.belief_state))

    plan2 = _last_plan_from_session(session)
    print("\n--- Orchestrator Plan Round 2 ---")
    _print_plan(plan2)

    print("\n--- Summarizer Output Round 2 (truncated) ---")
    s2 = str(result2)
    print(s2[:800] + ("..." if len(s2) > 800 else ""))

    # ---- 4) Round 3: follow-up q3 (no file) --------------------------------
    print("\n=== ROUND 3 ===")
    result3 = await conversation_loop(args.q3, session.task, session, llms)

    print("\n--- Beliefs AFTER Round 3 ---")
    print(_pretty_beliefs(session.belief_state))

    plan3 = _last_plan_from_session(session)
    print("\n--- Orchestrator Plan Round 3 ---")
    _print_plan(plan3)

    print("\n--- Summarizer Output Round 3 (truncated) ---")
    s3 = str(result3)
    print(s3[:800] + ("..." if len(s3) > 800 else ""))

    # Optional: dump full session to a JSON file for inspection
    safe_history = [{**m, "content": str(m.get("content", ""))} for m in session.conversation_history]
    dump = {
        "belief_state": session.belief_state,
        "conversation_history": safe_history,
        "round1_plan": plan1,
        "round2_plan": plan2,
        "round3_plan": plan3,
        "round1_output": s1,
        "round2_output": s2,
        "round3_output": s3
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/smoke_two_rounds.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    print("\nSaved full run to logs/smoke_two_rounds.json")


if __name__ == "__main__":
    asyncio.run(main())
