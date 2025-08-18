from typing import Dict, Any

def format_tool_descriptions(tools: Dict[str, Any]) -> str:
    """
    Returns a markdown block describing tools. Accepts ToolProfile or any object
    with a .describe() that returns a dict. Tolerates missing fields.
    """
    lines = []
    for name, profile in tools.items():
        desc = profile.describe() if hasattr(profile, "describe") else {}
        nm = desc.get("name", name)
        purpose = desc.get("purpose", "")
        inputs = desc.get("inputs", {}) or {}
        strengths = desc.get("strengths", "")
        limitations = desc.get("limitations", "")
        caps = desc.get("capabilities", []) or []
        requires_net = desc.get("requires_network", False)

        # Be resilient to different cost/latency fields
        cost = desc.get("cost") or desc.get("cost_label") or desc.get("cost_hint") or "Unknown"
        latency = desc.get("latency") or desc.get("latency_label") or f"~{desc.get('latency_hint_ms', '?')} ms"

        inputs_str = ", ".join(f"{k}: {v}" for k, v in inputs.items()) if inputs else "–"
        caps_str = ", ".join(caps) if caps else "–"

        lines.append(
            f"- **{nm}**: {purpose}\n"
            f"  - Inputs: {inputs_str}\n"
            f"  - Capabilities: {caps_str}\n"
            f"  - Strengths: {strengths}\n"
            f"  - Limitations: {limitations}\n"
            f"  - Cost: {cost}\n"
            f"  - Latency: {latency}\n"
            f"  - Network: {'Yes' if requires_net else 'No'}"
        )
    return "\n".join(lines)
