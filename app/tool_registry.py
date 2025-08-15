from app.prompts.summarizer_prompt import get_prompt
from app.tools.local import summarizer_tool

def get_tools():
    return {
        "summarizer": summarizer_tool.Summarizer(get_prompt())
    }
