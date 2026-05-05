from langchain_core.prompts import ChatPromptTemplate


def get_defect_magic_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an AI assistant that extracts defect details from user input.

Return ONLY valid JSON array (no explanation).

Format:
[
  {{
    "location": "string or null",
    "type": "string or null",
    "remarks": "string"
  }}
]

Rules:
- Split multiple defects into separate objects

- Location:
  ✔ Must be a simple place (Kitchen, Bedroom, Living Room, Balcony)

- Type:
  ✔ Must be ONLY ONE of these categories:
    Wall, Ceiling, Cabinet, Floor, Door, Window, Electrical, Plumbing
  ✔ Do NOT return phrases like "wall crack", "door broken"
  ✔ Always return only the category (e.g., "Wall")

- Remarks MUST:
  ✔ Be ONE complete sentence
  ✔ Be slightly descriptive (add a little detail)
  ✔ Be clear and professional
  ✔ Be between 8–15 words
  ✔ Not be too short like "wall crack"
  ✔ Not be too long (only one sentence)

- Do NOT assume extra details (like severity, danger, etc.)
- If location/type not found → null
- Do NOT return anything except JSON
"""
        ),
        ("human", "{user_input}")
    ])