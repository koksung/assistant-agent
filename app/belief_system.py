# app/belief_system.py
from __future__ import annotations

import json
import math
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from app.main import llms

# Initial random belief vector (currently unused; keep or remove)
belief_state = np.array([0.5, 0.5, 0.5])  # [methodology, theory, application]

ARCH_KEYS = ["explorer", "deep_dive_analyst", "summary_seeker", "math_oriented"]

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ``` fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        # Try to recover: find the first {...} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def _normalize_probs(d: Dict[str, float]) -> Dict[str, float]:
    # clamp negatives, handle missing, renormalize
    vals = [max(0.0, float(d.get(k, 0.0))) for k in ARCH_KEYS]
    s = sum(vals)
    if s <= 0:
        # uniform fallback
        return {k: 1.0 / len(ARCH_KEYS) for k in ARCH_KEYS}
    return {k: v / s for k, v in zip(ARCH_KEYS, vals)}

def _derive_archetype_tags(arch_probs: Dict[str, float], top_k: int = 2, thresh: float = 0.35) -> List[str]:
    # Map model archetypes → router archetype tags
    ranked = sorted(arch_probs.items(), key=lambda kv: kv[1], reverse=True)
    tags: List[str] = []
    for i, (name, p) in enumerate(ranked[:top_k]):
        if p >= thresh or i == 0:
            if name == "math_oriented":
                tags.append("math_heavy")
            if name == "deep_dive_analyst":
                tags.append("detail_oriented")
            if name == "summary_seeker":
                tags.append("prefers_summary")
            if name == "explorer":
                tags.append("curiosity_driven")
    return tags

def _derive_preferences(pref_blob: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    prefs: Dict[str, Any] = {}

    # Pass through known fields
    for k in ("curious_about", "preferred_focus", "familiar_with", "depth_preference", "tone"):
        if k in pref_blob and pref_blob[k] not in (None, "", []):
            prefs[k] = pref_blob[k]

    # Router-friendly hints
    depth = (pref_blob.get("depth_preference") or "").lower()
    tone = (pref_blob.get("tone") or "").lower()

    # Slightly opinionated mappings; tune as you like
    prefs["prefer_local"] = tone in {"concise"}  # concise users often want snappy/local ops
    if depth in {"technical", "detailed"}:
        prefs["want_more_detail"] = True

    # Doc-aware hints (cheap heuristics)
    text = user_text.lower()
    prefs["doc_has_equations"] = any(w in text for w in ("equation", "latex", "derivation", "proof"))
    prefs["has_complex_layout"] = any(w in text for w in ("figure", "table", "two-column", "layout", "diagram"))

    return prefs

def update_beliefs(text_or_response: str) -> Dict[str, Any]:
    """
    Extract structured belief updates including probabilistic archetype classification.
    Returns:
      {
        "belief": {...},                 # raw normalized belief JSON
        "routing_ctx": {                 # minimal dict for ToolCallContext(**routing_ctx)
           "archetypes": [...],
           "preferences": {...}
        }
      }
    """
    prompt = f"""
You are a belief modeling assistant. Your task is to estimate how likely a user is to belong to each archetype,
based on the input they've provided.

Treat this as a probabilistic inference problem. For each archetype, estimate:
    P(user_input | archetype) × P(archetype)
Assume a uniform prior across the 4 types. Output posterior probabilities.

Archetypes:
- explorer: curious, open-ended, novelty-seeking
- deep_dive_analyst: detail-oriented, seeks technical depth
- summary_seeker: wants high-level summaries, avoids complexity
- math_oriented: prefers equations, derivations, or formalisms

Also infer preferences:
- curious_about: key topics the user is interested in
- preferred_focus: 'methods' | 'results' | 'math' | ...
- depth_preference: "high-level" | "detailed" | "technical"
- tone: "concise" | "elaborate" | "structured"
- familiar_with: ML concepts the user understands

Respond in pure JSON only:
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

User input:
\"\"\"{text_or_response}\"\"\"
""".strip()

    try:
        belief_updater_llm = llms["belief_updater"]
        raw = belief_updater_llm.invoke(prompt)
        if not isinstance(raw, str):
            # Some LLM wrappers return dicts already
            parsed = raw
        else:
            parsed = _safe_json_loads(_strip_code_fences(raw))

        if not parsed or "belief" not in parsed:
            return {"belief": {}, "routing_ctx": {"archetypes": [], "preferences": {}}}

        belief = parsed.get("belief", {})
        probs = _normalize_probs(belief.get("archetype_probs", {}))
        prefs = belief.get("preferences", {}) or {}

        # Derived router signals
        archetype_tags = _derive_archetype_tags(probs)
        router_prefs = _derive_preferences(prefs, text_or_response)

        normalized = {
            "belief": {
                "archetype_probs": probs,
                "preferences": {
                    "curious_about": prefs.get("curious_about", []),
                    "preferred_focus": prefs.get("preferred_focus", ""),
                    "familiar_with": prefs.get("familiar_with", []),
                    "depth_preference": prefs.get("depth_preference", ""),
                    "tone": prefs.get("tone", ""),
                }
            },
            "routing_ctx": {
                "archetypes": archetype_tags,
                "preferences": router_prefs
            }
        }
        return normalized

    except Exception:
        # Don’t raise in the belief path—just return neutral defaults
        return {"belief": {}, "routing_ctx": {"archetypes": [], "preferences": {}}}
