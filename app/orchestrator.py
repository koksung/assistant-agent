import os
from app.tool_registry import get_tools
from app.belief_system import update_beliefs
from app.tools.local.pdf_extractor import extract_text_from_pdf

async def process_paper(file):
    content = await file.read()
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    # Step 1: Extract text
    pdf_text = extract_text_from_pdf(file_path)

    # Step 2: Run belief update (stub)
    belief_vector = update_beliefs(pdf_text)

    # Step 3: Use summarization tool
    tools = get_tools()
    summary_tool = tools["summarizer"]
    summary = summary_tool.run(pdf_text, belief_vector)

    return summary
