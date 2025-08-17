from dotenv import dotenv_values, load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager

from app.llm.setup import LLMSetup
from app.orchestrator import process_paper
from app.tool_registry import initialize_tool_registry, set_global_registry
from app.utils.logger import get_logger

os.environ["OPENAI_API_KEY"] = dotenv_values().get("OPENAI_API_KEY")
logger = get_logger(__name__)


llms = {
    # Low-temp for summarizer (precise, factual)
    "summarizer": LLMSetup("summarizer_llm", temperature=0.3).get_llm(),
    # Higher-temp for creative or generative reasoning
    "orchestrator": LLMSetup("orchestrator_llm", temperature=0.7).get_llm(),
    # Max creativity (e.g., for analogies or hypotheses)
    "creative": LLMSetup("explorer_llm", temperature=0.9).get_llm()
}


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    registry = initialize_tool_registry()
    set_global_registry(registry)  # Set the global registry

    logger.info("Available tools:\n")
    for name, tool in registry.list_all_tools().items():
        logger.info(tool.describe())

    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    result = await process_paper(file)
    return {"summary": result}


# uvicorn app.main:app --reload
