from dotenv import dotenv_values, load_dotenv
load_dotenv()

import io
import os
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.llm.setup import LLMSetup
from app.orchestrator import conversation_loop, prepare_user_task
from app.tool_registry import initialize_tool_registry, set_global_registry
from app.users.session import UserSessionManager
from app.users.api_models import InteractionRequest
from app.utils.logger import get_logger

# Optional: only set if present in .env
api_key = dotenv_values().get("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

logger = get_logger(__name__)

# Initialize LLMs (✅ removed trailing space)
llms = {
    "summarizer":   LLMSetup("summarizer_llm",   temperature=0.3).get_llm(),
    "orchestrator": LLMSetup("orchestrator_llm", temperature=0.7).get_llm(),
    "belief_updater": LLMSetup("belief_updater_llm", temperature=0.4).get_llm(),
}

# Global registry + session manager
session_manager = UserSessionManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry = initialize_tool_registry()
    set_global_registry(registry)
    logger.info("Available tools:")
    for name, tool in registry.list_all_tools().items():
        logger.info(tool.describe())
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.post("/interact/")
async def interact(request: InteractionRequest):
    logger.info(f"New interaction from user: {request.user_id} | file: {request.file_path}")
    session = session_manager.get_or_create_session(request.user_id)

    # Round 1: user provides a file_path (server-side path). Load and wrap as UploadFile.
    if request.file_path:
        try:
            with open(request.file_path, "rb") as f:
                file_bytes = await asyncio.to_thread(f.read)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="file_path not found on server.")
        file = UploadFile(filename=request.file_path.split("/")[-1], file=io.BytesIO(file_bytes))
        result = await prepare_user_task(file, user_query=request.user_query, session=session, llms=llms)
    else:
        if not session.task:
            raise HTTPException(status_code=400, detail="No active document for this session.")
        result = await conversation_loop(request.user_query, session.task, session, llms)

    session_manager.update_session(session)
    return JSONResponse(content={"response": result})
