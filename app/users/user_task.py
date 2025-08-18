from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class UserTask(BaseModel):
    file_path: str                               # Local file path after upload
    raw_text: str = ""                           # Prompt-safe text version (always a string)
    user_query: Optional[str] = None             # Optional instruction from the user
    task_type: str = "summarize"                 # Default task
    metadata: Optional[Dict[str, Any]] = None    # File name, upload time, etc.
    cache: Dict[str, Any] = Field(default_factory=dict)
    # cache may contain:
    #   - "extracted_text": longform string (same as or longer than raw_text)
    #   - "extracted_struct": structured dict from extractor

    def get_full_text(self) -> str:
        """
        Return the best available longform text for summarization/retrieval.
        Priority:
          1) cache['extracted_text'] if present
          2) raw_text (string)
          3) a synthesized string from cache['extracted_struct'] if available
        """
        # 1) Prefer explicitly cached long text
        if isinstance(self.cache.get("extracted_text"), str) and self.cache["extracted_text"].strip():
            return self.cache["extracted_text"]

        # 2) Fallback to normalized string
        if isinstance(self.raw_text, str) and self.raw_text.strip():
            return self.raw_text

        # 3) Last resort: synthesize from structured dict
        struct = self.cache.get("extracted_struct")
        if isinstance(struct, dict) and struct:
            parts = []
            for k, v in struct.items():
                if isinstance(v, str) and v.strip():
                    parts.append(f"# {str(k).replace('_', ' ').title()}\n{v.strip()}")
                elif isinstance(v, list):
                    # join simple string lists conservatively
                    string_items = [str(x) for x in v if isinstance(x, (str, int, float))]
                    if string_items:
                        parts.append(f"# {str(k).replace('_', ' ').title()}\n" + "\n".join(string_items))
            if parts:
                return "\n\n".join(parts)

        # If absolutely nothing usable, return empty string
        return ""
