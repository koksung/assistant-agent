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
    # Extract belief components
    belief = belief_state.get("belief", {})
    archetypes = belief.get("archetype_probs", {})
    preferences = belief.get("preferences", {})

    # Format archetypes and preferences
    formatted_archetypes = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v:.2f}" for k, v in archetypes.items()
    ) or "Unknown"

    formatted_preferences = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in preferences.items()
    ) or "Not specified"

    # Assemble intermediate tool outputs
    intermediate_summary = ""
    for item in intermediate_outputs:
        tool_name = item.get("tool", "unknown_tool")
        output = item.get("output", "").strip()
        intermediate_summary += f"\n### Output from `{tool_name}`\n{output}\n"

    # Final prompt to summarizer LLM
    return f"""
    You are an academic research assistant helping summarize scientific papers.
    
    ## User Query
    "{user_input}"
    
    ## User Belief Profile
    
    ### Archetype Probabilities
    {formatted_archetypes}
    
    ### User Preferences
    {formatted_preferences}
    
    ## Contextual Instructions
    
    - Align your summary to the user's intent and preferences.
    - Adjust tone, length, and depth based on belief profile.
    - Emphasize methods, results, math, or novelty **as appropriate**.
    
    ## Tool Outputs for Reference
    {intermediate_summary}
    
    ## Your Task
    
    Write a clean, markdown-formatted summary of the paper. Include:
    - Appropriate section headers (e.g., ## Abstract, ## Methodology)
    - Highlights aligned with the user’s archetype
    - Clear structure, precise explanations, and user-adaptive tone
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

    session.set_task(task)  # Now stored for later rounds
    session.add_message("user", user_query)

    # Initial belief update (only if not already present)
    if not session.belief_state:
        insight = update_beliefs(user_query)
        session.update_beliefs(insight)

    # Conversational reasoning loop
    response = await conversation_loop(user_query, task, session, llms)
    return response


async def conversation_loop(user_input: str, task: UserTask, session: ConversationContext, llms: Dict):
    tools = get_tools()
    orchestrator_llm = llms["orchestrator"]
    summarizer_llm = llms["summarizer"]

    preview = task.get_full_text()[:1000]
    archetypes = session.belief_state.get("belief", {}).get("archetype_probs", {})
    preferences = session.belief_state.get("belief", {}).get("preferences", {})
    formatted_archetypes = "\n".join(
        f"- {k}: {v:.2f}" for k, v in archetypes.items()
    ) or "Unknown"
    formatted_prefs = "\n".join(
        f"- {k}: {v}" for k, v in preferences.items()
    ) or "No explicit preferences"

    # Step 4: Plan via CoT
    cot_prompt = f"""
    You are a highly capable orchestration agent that can choose the right tools to help a user understand or summarize a scientific paper.

    ## User Input:
    "{user_input}"

    ## Paper Preview:
    {preview}

    ## User Belief Profile:

    ### Archetype Probabilities:
    {formatted_archetypes}

    ### Known Preferences:
    {formatted_prefs}

    ## Instructions:
    Based on the user’s goal and belief profile above:
    - Think step-by-step.
    - Determine which tools to use (in order).
    - Prioritize tools that match the user's archetype and preferences.
    - Be adaptive: if the user is math-oriented, prioritize equation tools; if they are a summary-seeker, go straight to summarization.

    ## Available Tools:
    {list(tools.keys())}

    Output your step-by-step reasoning as a structured plan.
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
    full_context = "\n".join([f"{m['role']}: {m['content']}" for m in session.conversation_history])
    session.update_beliefs(update_beliefs(full_context))
    session.add_message("summarizer", response)

    return response
