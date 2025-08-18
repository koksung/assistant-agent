from pydantic import BaseModel, Field
from app.prompts.summarizer_prompt import CHUNK_USER_TEMPLATE, ONE_SHOT_USER_TEMPLATE


class GenerateSummaryInput(BaseModel):
    context: str = Field(..., description="Summarises text.")


def summarize_abstract(text_to_be_summarized, summarizer_llm):
    user_prompt = ONE_SHOT_USER_TEMPLATE.format(
        target_words=150,
        style="neutral technical",
        format_hint="3–5 sentences",
        text=text_to_be_summarized
    )

    # build messages for LLM
    messages = [
        {"role": "system", "content": "You are a helpful & competent research assistant. "
                                      "Summarise the abstract of this academic paper."},
        {"role": "user", "content": user_prompt},
    ]

    result = summarizer_llm.invoke(messages)
    return result.content


def generate_summary(context_chunk, summarizer_llm):
    user_prompt = CHUNK_USER_TEMPLATE.format(
        chunk = context_chunk
    )
    messages = [
        {"role": "system", "content": "You are a helpful & competent research assistant. "
                                      "Provide a competent summary."},
        {"role": "user", "content": user_prompt},
    ]
    result = summarizer_llm.invoke(messages)
    return result.content
