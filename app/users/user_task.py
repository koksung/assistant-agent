from typing import Optional, Dict
from pydantic import BaseModel

class UserTask(BaseModel):
    file_path: str                     # Local file path after upload
    raw_text: Dict[str, str]           # Text extracted from the PDF
    user_query: Optional[str] = None   # Optional instruction from the user
    task_type: str = "summarize"       # Default task
    metadata: Optional[Dict] = None    # File name, upload time, etc.

    def get_full_text(self) -> str:
        return "\n\n".join(
            f"# {section_name.replace('_', ' ').title()}\n{section_text.strip()}"
            for section_name, section_text in self.raw_text.items()
        )
