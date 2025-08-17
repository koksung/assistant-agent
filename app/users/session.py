from typing import List, Dict, Any
from pydantic import BaseModel

class ConversationContext(BaseModel):
    user_id: str
    belief_state: Dict[str, Any]
    conversation_history: List[Dict[str, str]]  # [{role: 'user', content: '...'}, ...]

    def update_beliefs(self, new_insight: Dict[str, Any]):
        self.belief_state.update(new_insight)

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})


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
