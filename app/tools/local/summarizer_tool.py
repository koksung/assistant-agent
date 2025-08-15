from langchain.llms import OpenAI

class Summarizer:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.llm = OpenAI(temperature=0.3)

    def run(self, text, belief_vector):
        formatted_prompt = self.prompt_template + f"\nBelief vector: {belief_vector}\n\nPaper:\n{text}"
        return self.llm.predict(formatted_prompt)
