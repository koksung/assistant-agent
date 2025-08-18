from app.prompts.summarizer_prompt import ONE_SHOT_USER_TEMPLATE

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
