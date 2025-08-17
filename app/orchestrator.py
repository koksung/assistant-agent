import re
from typing import List, Dict

from app.tool_registry import get_tools
from app.belief_system import update_beliefs
from app.tools.local.pdf_extractor import extract_pdf_text
from app.users.user_task import UserTask
from app.users.session import ConversationContext
from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_plan(plan_str: str) -> List[Dict]:
    steps = []
    pattern = r"Use '([\w_]+)'(?: on (.*))?"
    for line in plan_str.splitlines():
        match = re.search(pattern, line)
        if match:
            tool_name = match.group(1)
            target = match.group(2) if match.group(2) else ""
            steps.append({
                "tool": tool_name,
                "input_target": target.strip()
            })
    return steps


def build_tool_input(step: Dict, task: UserTask, session: ConversationContext) -> Dict:
    target = step.get("input_target", "").lower()

    if "raw text" in target or "extracted text" in target:
        return {"text": task.raw_text}
    if "user query" in target:
        return {"query": task.user_query}
    if "beliefs" in target:
        return {"beliefs": session.belief_state}
    if "file path" in target:
        return {"pdf_path": task.file_path}

    return {
        "text": task.raw_text,
        "query": task.user_query,
        "beliefs": session.belief_state,
        "pdf_path": task.file_path
    }


def build_summary_prompt(
    task: UserTask,
    intermediate_outputs: List[Dict],
    belief_state: Dict,
    user_input: str
) -> str:
    intermediate_summary = ""
    for item in intermediate_outputs:
        tool_name = item.get("tool")
        output = item.get("output")
        intermediate_summary += f"\n[{tool_name} Output]:\n{output}\n"

    return f"""You are an academic research assistant helping summarize scientific papers.
    The user said: "{user_input}"
    Your understanding of the user archetype so far: {belief_state}
    
    Here are intermediate tool outputs you can use:
    {intermediate_summary}
    
    Now, summarize the paper accordingly. Be precise and match the user's preferences.
    Output should be structured in clean markdown with clear section headings.
    """.strip()


async def prepare_user_task(file, user_query: str, session: ConversationContext, llms: Dict):
    content = await file.read()
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract text
    pdf_text = extract_pdf_text(file_path)

    task = UserTask(
        file_path=file_path,
        raw_text=pdf_text,
        user_query=user_query,
        task_type="summarize",
        metadata={"filename": file.filename}
    )

    # Add user message
    session.add_message("user", user_query)

    # Step 3: Initial belief update (only if not already present)
    if not session.belief_state:
        session.update_beliefs(update_beliefs(pdf_text))

    # Step 4-7: Conversational reasoning loop
    response = await conversation_loop(user_query, task, session, llms)
    return response


async def conversation_loop(user_input: str, task: UserTask, session: ConversationContext, llms: Dict):
    tools = get_tools()
    orchestrator_llm = llms["orchestrator"]
    summarizer_llm = llms["summarizer"]

    preview = task.get_full_text()[:1000]

    # Step 4: Plan via CoT
    cot_prompt = f"""You are an LLM researcher assistant.
    User wants: "{user_input}"
    Current paper content has: {preview}...
    
    User archetype: {session.belief_state}
    
    What tools should be used to fulfill the user's intent?
    Think step-by-step. Return a structured plan.
    Available tools: {list(tools.keys())}
    """.strip()
    plan = orchestrator_llm.invoke(cot_prompt)

    session.add_message("orchestrator", plan)

    # Step 5: Tool execution
    intermediate_outputs = []
    for step in parse_plan(plan):
        tool_name = step.get("tool")
        if tool_name not in tools:
            continue
        tool = tools[tool_name]
        input_data = build_tool_input(step, task, session)
        logger.info(f"[Tool]: Executing {tool_name} with inputs: {input_data}")
        output = tool.tool.invoke(input_data)
        intermediate_outputs.append({"tool": tool_name, "output": output})

    # Step 6: Generate final summary
    summary_prompt = build_summary_prompt(task, intermediate_outputs, session.belief_state, user_input)
    response = summarizer_llm.invoke(summary_prompt)

    # Step 7: Update beliefs
    session.update_beliefs(update_beliefs(response))
    session.add_message("summarizer", response)

    return response
