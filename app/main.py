from fastapi import FastAPI, UploadFile, File
from app.orchestrator import process_paper

app = FastAPI()

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    result = await process_paper(file)
    return {"summary": result}


if __name__ == "__main__":
    import sys
    sys.path.append("..")  # or absolute path to root
