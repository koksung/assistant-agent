from dotenv import dotenv_values, load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.llm.setup import LLMSetup
from app.orchestrator import prepare_user_task
from app.tool_registry import initialize_tool_registry, set_global_registry
from app.users.session import UserSessionManager
from app.users.api_models import InteractionRequest
from app.utils.logger import get_logger

os.environ["OPENAI_API_KEY"] = dotenv_values().get("OPENAI_API_KEY")
logger = get_logger(__name__)

# Initialize LLMs
llms = {
    "summarizer": LLMSetup("summarizer_llm", temperature=0.3).get_llm(),
    "orchestrator": LLMSetup("orchestrator_llm", temperature=0.7).get_llm(),
    "creative": LLMSetup("explorer_llm", temperature=0.9).get_llm()
}

# Global registry + session manager
session_manager = UserSessionManager()

@asynccontextmanager
async def lifespan(fapp: FastAPI):
    registry = initialize_tool_registry()
    set_global_registry(registry)
    logger.info("Available tools:\n")
    for name, tool in registry.list_all_tools().items():
        logger.info(tool.describe())
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.post("/interact/")
async def interact(request: InteractionRequest):
    logger.info(f"New interaction from user: {request.user_id} | file: {request.file_path}")
    session = session_manager.get_or_create_session(request.user_id)

    # Open the file manually (simulating UploadFile)
    with open(request.file_path, "rb") as f:
        from fastapi import UploadFile
        file = UploadFile(filename=request.file_path, file=f)

        result = await prepare_user_task(
            file=file,
            user_query=request.user_query,
            session=session,
            llms=llms
        )

    session_manager.update_session(session)
    return JSONResponse(content={"response": result})
