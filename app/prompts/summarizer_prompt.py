DEFAULT_SYSTEM = """You are a precise research assistant.
Summarize technical text faithfully and concisely.
- Prefer concrete claims over hype
- Use plain language unless math/notation is essential
- If numbers/metrics are present, keep them
- Do not invent citations or facts
- If the source is ambiguous, hedge carefully
"""

# For chunk-level (map) summaries
CHUNK_USER_TEMPLATE = """You will receive some gathered context of a conversation and bits of a larger document.
Write a concise summary of ONLY these context as info.

Requirements:
- 3–6 bullet points
- Preserve key claims, methods, and important numbers if present
- No intro/outro text—just bullets
- Do not reference 'this chunk' or 'this section'

Chunk:
---
{chunk}
---
"""

# For final (reduce) summary over the partials
REDUCE_USER_TEMPLATE = """You will receive bullet summaries from multiple chunks of the same document.
Synthesize them into a single coherent summary for the whole document.

Controls:
- Target length: {target_words} words
- Style: {style}
- Format: {format_hint}

Guidelines:
- Merge duplicate points
- Preserve important numbers, datasets, methods
- Avoid speculation or invented details
- If uncertainty remains (e.g., missing results), state it briefly

Chunk bullet summaries:
---
{bullets}
---
"""

# One-shot for short texts (skip map-reduce)
ONE_SHOT_USER_TEMPLATE = """Summarize the following text.

Controls:
- Target length: {target_words} words
- Style: {style}
- Format: {format_hint}

Text:
---
{text}
---
"""
