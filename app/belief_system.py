# app/belief_system.py
from __future__ import annotations

import json, re
from typing import Dict, Any, List
import numpy as np

# keep your belief_state if you need it elsewhere
belief_state = np.array([0.5, 0.5, 0.5])

ARCH_KEYS = ["explorer", "deep_dive_analyst", "summary_seeker", "math_oriented"]

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_json_loads(s: str) -> Dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def _uniform_probs() -> Dict[str, float]:
    n = len(ARCH_KEYS)
    return {k: 1.0 / n for k in ARCH_KEYS}

def _heuristic_preferences(text: str) -> Dict[str, Any]:
    t = (text or "").lower()

    math_kw = ("equation", "equations", "math", "derivation", "proof", "latex", "formula")
    detail_kw = ("methodology", "implementation", "details", "ablation", "hyperparameter", "architecture")
    summary_kw = ("summary", "overview", "tl;dr", "high-level", "executive")
    tone_concise_kw = ("brief", "short", "concise", "tl;dr")
    tone_structured_kw = ("bullet", "outline", "structured")

    # preferred_focus
    if any(k in t for k in math_kw):
        focus = "math"
    elif any(k in t for k in detail_kw):
        focus = "methods"
    else:
        focus = "results" if "result" in t or "conclusion" in t else ""

    # depth_preference
    if any(k in t for k in summary_kw):
        depth = "high-level"
    elif any(k in t for k in detail_kw + math_kw):
        depth = "technical"
    else:
        depth = ""

    # tone
    if any(k in t for k in tone_concise_kw):
        tone = "concise"
    elif any(k in t for k in tone_structured_kw):
        tone = "structured"
    else:
        tone = ""

    # familiar_with (very light heuristics—extend as you like)
    fam_terms = [
        ("diffusion", "diffusion"),
        ("variational", "variational inference"),
        ("bayes", "bayesian"),
        ("transformer", "transformers"),
        ("gradient", "gradients"),
        ("markov", "markov chains"),
    ]
    familiar_with = [label for kw, label in fam_terms if kw in t]

    # curious_about: extract noun-ish keywords (ultra-simple)
    curious_about: List[str] = []
    for w in ("methodology", "equations", "results", "limitations", "datasets", "ablation", "theory"):
        if w in t:
            curious_about.append(w)

    return {
        "curious_about": curious_about,
        "preferred_focus": focus,
        "familiar_with": familiar_with,
        "depth_preference": depth,
        "tone": tone,
    }

def _heuristic_archetype_probs(text: str, prefs: Dict[str, Any]) -> Dict[str, float]:
    t = (text or "").lower()
    score = {k: 0.0 for k in ARCH_KEYS}
    if prefs.get("preferred_focus") == "math" or any(k in t for k in ("equation", "derivation", "proof", "latex", "math")):
        score["math_oriented"] += 2.5
    if prefs.get("preferred_focus") in ("methods",) or any(k in t for k in ("methodology", "implementation", "details", "ablation")):
        score["deep_dive_analyst"] += 1.8
    if prefs.get("depth_preference") == "high-level" or any(k in t for k in ("summary", "overview", "tl;dr", "high-level")):
        score["summary_seeker"] += 1.6
    # if none of the above, nudge explorer
    if not any(v > 0 for v in score.values()):
        score["explorer"] += 1.0

    s = sum(score.values())
    return {k: (v / s) if s > 0 else 1.0 / len(ARCH_KEYS) for k, v in score.items()}

def update_beliefs(text_or_response: str, curr_belief, belief_updater_llm) -> Dict[str, Any]:
    """
    Robust belief updater:
    - Tries LLM JSON
    - Falls back to heuristics for both prefs and archetype probs
    - Always returns non-empty 'belief' and 'routing_ctx'
    """
    archetype_probs = curr_belief["belief"].get("archetype_probs", {})
    current_belief_state = ", ".join(f"{k}:{v:.4f}" for k, v in archetype_probs.items())

    prompt = f"""
         You are a belief modeling assistant. Your task is to estimate how likely a user is to belong to each archetype,
    based on the input they've provided.

    TASK: Calculate posterior probabilities P(archetype|evidence) using Bayesian updating for each archetype:
        P(archetype | user_input) = P(user_input | archetype) × P(archetype)
       
    For each archetype, assess:
    1. LIKELIHOOD P(user_input|archetype): How likely would this archetype ask this specific question?
       - math_oriented: High for equations, proofs, technical details, derivations, formalisms
       - visual_learner: High for diagrams, figures, visual explanations, wants high-level summaries, avoids complexity
       - explorer: High for overviews, abstracts, key points, curious, open-ended, novelty-seeking
       - deep_dive_analyst: High for comprehensive analysis, thorough exploration, detail-oriented, seeks technical depth
    2. PRIOR P(archetype): Use `current_belief_state` as prior
    3. Calculate unnormalized posterior: likelihood × prior
    4. NORMALIZE: Ensure all probabilities sum to 1.0
    
    Also infer preferences:
    - curious_about: key topics the user is interested in
    - preferred_focus: 'methods' | 'results' | 'math' | ...
    - depth_preference: "high-level" | "detailed" | "technical"
    - tone: "concise" | "elaborate" | "structured"
    - familiar_with: ML concepts the user understands
    
    **Respond in pure JSON ONLY (NO code fences, NO commentary).
    
    {{
      "belief": {{
        "archetype_probs": {{
          "explorer": 0.0,
          "deep_dive_analyst": 0.0,
          "summary_seeker": 0.0,
          "math_oriented": 0.0
        }},
        "preferences": {{
          "curious_about": [],
          "preferred_focus": "",
          "familiar_with": [],
          "depth_preference": "",
          "tone": ""
        }}
      }}
    }}
    
    current_belief_state:
    {current_belief_state}
    
    User input:
    \"\"\"{text_or_response}\"\"\"
    """.strip()

    try:
        raw_obj = belief_updater_llm.invoke(prompt)
        raw = getattr(raw_obj, "content", raw_obj)  # handle AIMessage
        parsed = raw if isinstance(raw, dict) else _safe_json_loads(_strip_code_fences(str(raw)))
    except (Exception, ):
        parsed = None

    # Start with heuristics
    prefs = _heuristic_preferences(text_or_response)
    probs = _heuristic_archetype_probs(text_or_response, prefs)

    # If LLM parsed okay, overlay its values
    if parsed and isinstance(parsed, dict) and "belief" in parsed:
        b = parsed["belief"] or {}
        llm_prefs = b.get("preferences") or {}
        llm_probs = b.get("archetype_probs") or {}

        # Fill from LLM only if non-empty
        for k in ("curious_about", "preferred_focus", "familiar_with", "depth_preference", "tone"):
            v = llm_prefs.get(k)
            if v not in (None, "", []):
                prefs[k] = v

        if isinstance(llm_probs, dict) and any(isinstance(v, (int, float)) and v > 0 for v in llm_probs.values()):
            # normalize
            vals = {k: max(0.0, float(llm_probs.get(k, 0.0))) for k in ARCH_KEYS}
            s = sum(vals.values())
            probs = {k: (v / s) if s > 0 else 1.0 / len(ARCH_KEYS) for k, v in vals.items()}

    # Derive routing tags
    tags: List[str] = []
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][0] if ranked else None
    if probs.get("math_oriented", 0) >= 0.35 or top == "math_oriented":
        tags.append("math_heavy")
    if probs.get("deep_dive_analyst", 0) >= 0.35 or top == "deep_dive_analyst":
        tags.append("detail_oriented")
    if probs.get("summary_seeker", 0) >= 0.35 or top == "summary_seeker":
        tags.append("prefers_summary")

    routing_prefs = {
        "prefer_local": (prefs.get("tone", "").lower() == "concise")
    }
    # cheap doc hints (the orchestrator also augments these later)
    t = (text_or_response or "").lower()
    routing_prefs["doc_has_equations"] = any(k in t for k in ("equation", "derivation", "proof", "latex"))
    routing_prefs["has_complex_layout"] = any(k in t for k in ("table", "figure", "diagram", "two-column", "layout"))

    return {
        "belief": {
            "archetype_probs": probs,
            "preferences": prefs,
        },
        "routing_ctx": {
            "archetypes": tags,
            "preferences": routing_prefs,
        }
    }
