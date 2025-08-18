def format_tool_descriptions(tools: dict) -> str:
    return "\n\n".join([
        f"### {desc['name']}\n"
        f"- Purpose: {desc['purpose']}\n"
        f"- Inputs: {', '.join(f'{k}: {v}' for k, v in desc['inputs'].items())}\n"
        f"- Outputs: {desc['outputs']}\n"
        f"- Strengths: {desc['strengths']}\n"
        f"- Limitations: {desc['limitations']}\n"
        f"- Cost: {desc['cost']}\n"
        f"- Latency: {desc['latency']}"
        for desc in [t.describe() for t in tools.values()]
    ])
