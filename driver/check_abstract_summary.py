from dotenv import dotenv_values, load_dotenv
load_dotenv()

import os
from app.llm.setup import LLMSetup
from app.tools.local.abstract_extractor import abstract_summary

api_key = dotenv_values().get("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


def check_abstract_summary(pdf_path):
    summarizer = LLMSetup("summarizer_llm", temperature=0.4).get_llm()
    output = abstract_summary(pdf_path, summarizer)
    print(output)


if __name__ == "__main__":
    file_path = "C:/Users/Zeus/PycharmProjects/assistant-agent/data/ddpm-short.pdf"
    check_abstract_summary(file_path)
