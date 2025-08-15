from langchain_openai import ChatOpenAI


class LLMSetup:
    def __init__(self, name, model: str = "gpt-3.5-turbo", temperature: float = 0.3, max_tokens: int = 700):
            self.name = name
            self.llm = ChatOpenAI(
                model=model,  # or "gpt-3.5-turbo" or "gpt-4o"
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def get_llm(self):
        return self.llm
