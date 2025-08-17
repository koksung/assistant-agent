from pydantic import BaseModel


class InteractionRequest(BaseModel):
    user_id: str
    user_query: str
    file_path: str  # Full path to the local PDF file
