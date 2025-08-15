def get_prompt():
    with open("app/prompts/summarizer_prompt.txt") as f:
        return f.read()
