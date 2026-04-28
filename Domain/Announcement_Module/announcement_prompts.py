from langchain_core.prompts import ChatPromptTemplate


def get_announcement_route_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an Announcement Domain Router Agent.

Your task is to decide what announcement action is required.

Possible outputs (return ONLY ONE value):
- announcement_search : listing or finding announcements

Rules:
- The query is already confirmed to be announcement-related
- Do NOT return greetings
- Do NOT explain anything
- Return ONLY the action keyword
"""
            ),
            ("human", "{user_query}")
        ]
    )