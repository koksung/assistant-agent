from fastapi import FastAPI, UploadFile, File
from app.orchestrator import process_paper

app = FastAPI()

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    result = await process_paper(file)
    return {"summary": result}
