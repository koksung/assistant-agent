from typing import List, Dict, Any, Optional
from pprint import pformat
from pydantic import BaseModel, Field
from app.users.user_task import UserTask


def merge_and_smooth_probs(
    current: Dict[str, float],
    new: Dict[str, float],
    alpha: float = 0.7  # Higher = favor new insight more
) -> Dict[str, float]:
    """
    Exponential smoothing + renormalize.
    """
    if not new:
        return current or {}
    all_keys = set(current) | set(new)
    merged = {
        k: alpha * float(new.get(k, 0.0)) + (1 - alpha) * float(current.get(k, 0.0))
        for k in all_keys
    }
    total = sum(merged.values())
    if total > 0:
        merged = {k: round(v / total, 6) for k, v in merged.items()}
    return merged


def _uniq_preserve_order(xs: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in xs:
        if x in (None, "", []):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _merge_preferences(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge list/scalar prefs carefully:
    - Lists: union, de-dup, drop empties
    - Scalars: overwrite only if incoming is non-empty
    """
    cur = dict(current or {})
    inc = incoming or {}

    # Known list fields
    for key in ("curious_about", "familiar_with"):
        cur[key] = _uniq_preserve_order(list(cur.get(key, [])) + list(inc.get(key, [])))

    # Known scalar fields
    for key in ("preferred_focus", "depth_preference", "tone"):
        val = inc.get(key)
        if val not in (None, "", []):
            cur[key] = val

    # Pass through any extra custom keys from incoming (same non-empty rule)
    for k, v in inc.items():
        if k in ("curious_about", "familiar_with", "preferred_focus", "depth_preference", "tone"):
            continue
        if isinstance(v, list):
            cur[k] = _uniq_preserve_order(list(cur.get(k, [])) + v)
        else:
            if v not in (None, "", []):
                cur[k] = v

    return cur


def _merge_routing_ctx(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge routing_ctx:
    - archetypes: union list
    - preferences: OR booleans, keep truthy scalars, do not overwrite with falsy
    """
    cur = dict(current or {})
    inc = incoming or {}

    # Archetype tags
    cur_arch = _uniq_preserve_order(list(cur.get("archetypes", [])) + list(inc.get("archetypes", [])))

    # Preferences
    cur_p = dict(cur.get("preferences", {}) or {})
    inc_p = dict(inc.get("preferences", {}) or {})

    # Booleans we want to OR
    for bkey in ("doc_has_equations", "has_complex_layout", "prefer_local"):
        if bkey in inc_p:
            cur_p[bkey] = bool(inc_p[bkey] or cur_p.get(bkey, False))

    # For any other keys: only overwrite if incoming is truthy
    for k, v in inc_p.items():
        if k in ("doc_has_equations", "has_complex_layout", "prefer_local"):
            continue
        if v not in (None, "", [], False):
            cur_p[k] = v

    return {"archetypes": cur_arch, "preferences": cur_p}


class ConversationContext(BaseModel):
    user_id: str
    belief_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    task: Optional[UserTask] = None

    # hold a running compressed summary of older turns
    running_summary: str = ""

    def update_beliefs(self, new_insight: Dict[str, Any]):
        if not new_insight:
            return

        # Ensure containers exist
        if "belief" not in self.belief_state:
            self.belief_state["belief"] = {"archetype_probs": {}, "preferences": {}}

        # --- Merge archetype probabilities (smooth) ---
        cur_probs = self.belief_state["belief"].get("archetype_probs", {}) or {}
        new_probs = (new_insight.get("belief") or {}).get("archetype_probs", {}) or {}
        if new_probs:
            self.belief_state["belief"]["archetype_probs"] = merge_and_smooth_probs(cur_probs, new_probs, alpha=0.7)

        # --- Merge preferences (non-empty overwrite) ---
        cur_prefs = self.belief_state["belief"].get("preferences", {}) or {}
        new_prefs = (new_insight.get("belief") or {}).get("preferences", {}) or {}
        self.belief_state["belief"]["preferences"] = _merge_preferences(cur_prefs, new_prefs)

        # --- Merge routing_ctx (keep + OR flags) ---
        cur_rc = self.belief_state.get("routing_ctx", {}) or {}
        new_rc = new_insight.get("routing_ctx", {}) or {}
        if new_rc or cur_rc:
            self.belief_state["routing_ctx"] = _merge_routing_ctx(cur_rc, new_rc)

    def add_message(self, role: str, content: Any):
        # Always store string content for safe serialization
        self.conversation_history.append({"role": role, "content": str(content)})

    def compact_history(
            self,
            summarizer_llm,
            *,
            max_chars: int = 12000,  # overall budget for convo text
            keep_last: int = 2,  # keep the most recent N turns verbatim
            summary_target_chars: int = 1200
        ):
        """
        If conversation_history is too long, compress older messages into `running_summary`,
        keep only the last `keep_last` turns uncompressed.
        """
        # quick size estimate
        total_len = len(self.running_summary) + sum(len(m.get("content", "")) for m in self.conversation_history)
        if total_len <= max_chars or len(self.conversation_history) <= keep_last:
            return  # nothing to do

        old = self.conversation_history[:-keep_last]
        recent = self.conversation_history[-keep_last:]

        # Build a compacting prompt
        old_text = "\n".join(f"[{m['role']}] {m['content']}" for m in old)
        compress_prompt = f"""
        You are compressing a chat history. Produce a concise summary that preserves:
        - user goals/preferences
        - tool decisions/reasons
        - key findings/answers so far
        - open questions / next steps
    
        Limit to ~{summary_target_chars} characters. Output plain text only.
    
        Existing running summary (may be empty):
        \"\"\"{self.running_summary}\"\"\"
    
        New chunk to merge:
        \"\"\"{old_text}\"\"\"
        """.strip()

        try:
            new_sum = summarizer_llm.invoke(compress_prompt)
            new_sum = getattr(new_sum, "content", new_sum)
            new_sum = str(new_sum).strip()
            # Update running summary and collapse history
            self.running_summary = new_sum
            self.conversation_history = [{"role": "system",
                                          "content": f"Conversation summary:\n{self.running_summary}"}] + recent
        except (Exception, ):
            # Fail-safe: keep only last turns to protect token budget
            self.conversation_history = recent

    def set_task(self, task: UserTask):
        self.task = task

    def get_primary_archetype(self) -> Optional[str]:
        probs = (self.belief_state.get("belief") or {}).get("archetype_probs", {})
        if not probs:
            return None
        return max(probs.items(), key=lambda x: x[1])[0]

    def describe(self) -> str:
        return (
            f"User: {self.user_id}\nBeliefs:\n{pformat(self.belief_state)}\nRecent Messages:\n"
            + "\n".join([f"[{m['role']}] {m['content'][:60]}..." for m in self.conversation_history[-3:]])
        )


class UserSessionManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationContext] = {}

    def get_or_create_session(self, user_id: str) -> ConversationContext:
        if user_id not in self.sessions:
            self.sessions[user_id] = ConversationContext(user_id=user_id)
        return self.sessions[user_id]

    def update_session(self, session: ConversationContext):
        self.sessions[session.user_id] = session

    def clear_session(self, user_id: str):
        if user_id in self.sessions:
            del self.sessions[user_id]
