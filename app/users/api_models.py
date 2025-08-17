from pydantic import BaseModel
from typing import Optional


class InteractionRequest(BaseModel):
    user_id: str
    user_query: str
    file_path: Optional[str] = None  # optional in later turns
