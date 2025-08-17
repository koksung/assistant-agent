import json
import numpy as np

from app.main import llms
from typing import Dict


# Initial random belief vector
belief_state = np.array([0.5, 0.5, 0.5])  # [methodology, theory, application]

def update_beliefs(text_or_response: str) -> Dict:
    """
    Extracts structured belief updates including probabilistic archetype classification.
    """
    prompt = f"""
    You are a belief modeling assistant. Your task is to estimate how likely a user is to belong to each archetype,
    based on the input they've provided.

    Treat this as a probabilistic inference problem. For each archetype, estimate:

        P(user_input | archetype) × P(archetype)

    Assume prior P(archetype) is uniform across the 4 types. Your job is to provide the *posterior probabilities* after observing the user's input.

    ---

    Archetypes:
    - explorer: curious, open-ended, novelty-seeking
    - deep_dive_analyst: detail-oriented, seeks technical depth
    - summary_seeker: wants high-level summaries, avoids complexity
    - math_oriented: prefers equations, derivations, or formalisms

    Also infer **preferences** from the input, such as:
    - curious_about: key topics the user is interested in
    - preferred_focus: e.g., 'methods', 'results', 'math'
    - depth_preference: "high-level", "detailed", or "technical"
    - tone: "concise", "elaborate", or "structured"
    - familiar_with: ML concepts the user understands

    ---

    Respond in **pure JSON** format using this structure:

    {{
      "belief": {{
        "archetype_probs": {{
          "explorer": float (0–1),
          "deep_dive_analyst": float (0–1),
          "summary_seeker": float (0–1),
          "math_oriented": float (0–1)
        }},
        "preferences": {{
          "curious_about": [...],
          "preferred_focus": "...",
          "familiar_with": [...],
          "depth_preference": "...",
          "tone": "..."
        }}
      }}
    }}

    Base your probabilities **only** on the user input below.

    Input:
    \"\"\"{text_or_response}\"\"\"
    """
    try:
        belief_updater_llm = llms["belief_updater"]
        result = belief_updater_llm.invoke(prompt)
        return json.loads(result) if isinstance(result, str) else result
    except (Exception, ):
        return {}

