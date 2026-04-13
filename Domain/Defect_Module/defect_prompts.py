from langchain_core.prompts import ChatPromptTemplate

def get_defect_route_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Defect Domain Router Agent.

Your task is to decide what defect action is required.

Possible outputs (return ONLY ONE value):
- defect_search : listing or finding defects
- defect_status_update : updating or closing a defect
- defect_inspection : inspection scheduling or inspection updates

Rules:
- The query is already confirmed to be defect-related
- Do NOT return greetings or clarification
- Do NOT explain your decision
- Return ONLY the action keyword
"""
            ),
            ("human", "{user_query}")
        ]
    )
