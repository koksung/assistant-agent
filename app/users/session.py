from typing import List, Dict, Any, Optional
from pprint import pformat
from pydantic import BaseModel
from app.users.user_task import UserTask


def merge_and_smooth_probs(
        current: Dict[str, float],
        new: Dict[str, float],
        alpha: float = 0.7  # Higher = favor new insight more
    ) -> Dict[str, float]:
    """
    Merge two probability dictionaries using exponential smoothing and normalize.

    :param current: existing probability distribution
    :param new: new probability distribution from belief update
    :param alpha: smoothing factor (0 < alpha <= 1)
    :return: smoothed and normalized probability distribution
    """
    all_keys = set(current) | set(new)
    merged = {
        k: round(alpha * new.get(k, 0.0) + (1 - alpha) * current.get(k, 0.0), 6)
        for k in all_keys
    }

    total = sum(merged.values())
    if total > 0:
        merged = {k: round(v / total, 6) for k, v in merged.items()}

    return merged


class ConversationContext(BaseModel):
    user_id: str
    belief_state: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    task: Optional[UserTask] = None

    def update_beliefs(self, new_insight: Dict[str, Any]):
        new_belief = new_insight.get("belief", {})
        current = self.belief_state.setdefault("belief", {})

        # Merge archetype probabilities using smoothing
        new_probs = new_belief.get("archetype_probs", {})
        current_probs = current.get("archetype_probs", {})
        if new_probs:
            current["archetype_probs"] = merge_and_smooth_probs(current_probs, new_probs, alpha=0.7)

        # Merge preferences
        prefs = current.setdefault("preferences", {})
        for k, v in new_belief.get("preferences", {}).items():
            if isinstance(v, list):
                prefs[k] = list(set(prefs.get(k, []) + v))
            else:
                prefs[k] = v

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def set_task(self, task: UserTask):
        self.task = task

    def get_primary_archetype(self) -> Optional[str]:
        probs = self.belief_state.get("belief", {}).get("archetype_probs", {})
        if not probs:
            return None
        return max(probs.items(), key=lambda x: x[1])[0]

    def describe(self) -> str:
        return f"User: {self.user_id}\nBeliefs:\n{pformat(self.belief_state)}\nRecent Messages:\n" + \
            "\n".join([f"[{m['role']}] {m['content'][:60]}..." for m in self.conversation_history[-3:]])


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
