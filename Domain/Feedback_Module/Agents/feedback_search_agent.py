from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from Core.model_registry import get_chat_model


# =====================================================
# STATE
# =====================================================
class FeedbackSearchState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    response: Dict[str, Any]
    token: str
    login_id: int


# =====================================================
# LLM
# =====================================================
llm = get_chat_model()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Feedback Search Agent.

Extract filters from user query.

Return ONLY JSON:

{{
  "filters": {{
    "fromdate": null,
    "todate": null,
    "unit": null,
    "ticket": null,
    "status": null,
    "category": null,
    "building": null,
    "filter": null
  }}
}}

Rules:
- status: 1=Active, 2=Inactive, 3=Faulty, 4=Loss, 5=Stolen
- filter: created_at / fb_option / status
- Dates: YYYY-MM-DD
"""
        ),
        ("human", "{user_query}")
    ]
)

parser = JsonOutputParser()


# =====================================================
# INPUT MODEL
# =====================================================
class FeedbackSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    ticket: Optional[str] = None
    status: Optional[int] = None
    category: Optional[int] = None
    building: Optional[str] = None
    filter: Optional[str] = None

    token: str
    login_id: int


# =====================================================
# NODE
# =====================================================
async def feedback_search_node(state: FeedbackSearchState) -> Dict[str, Any]:
    from MCP.defect_mcp_server import search_feedback

    # =============================
    # LLM FILTER EXTRACTION
    # =============================
    chain = prompt | llm | parser

    llm_output = await chain.ainvoke({
        "user_query": state["user_query"]
    })

    filters = llm_output.get("filters", {})

    print(f"\n📋 [Feedback Search] Filters: {filters}")

    # =============================
    # CLEAN FILTERS
    # =============================
    cleaned_filters = {
        k: None if v in ["null", None, ""] else v
        for k, v in filters.items()
    }
    

    print(f"🧹 Cleaned Filters: {cleaned_filters}")
    print("🔍 Calling Feedback API...")

    try:
        result_obj = await search_feedback(
            FeedbackSearchInput(
                **cleaned_filters,
                token=state["token"],
                login_id=state["login_id"]
            )
        )

        # Handle dict / pydantic
        if hasattr(result_obj, "model_dump"):
            result_dict = result_obj.model_dump()
        else:
            result_dict = result_obj

        print(f"📦 Raw API Response Keys: {list(result_dict.keys())}")

    except Exception as e:
        print(f"❌ [Feedback Search] Error: {e}")
        return {
            **state,
            "response": {"message": str(e), "total": 0}
        }

    # =====================================================
    # RESPONSE PARSE
    # =====================================================
    records = result_dict.get("data", [])

    formatted = []

    for r in records:
        sub = r.get("submissions") or {}
        unit_info = r.get("unit_info") or {}
        user_info = r.get("user_info") or {}

        formatted.append({
            "ticket": sub.get("ticket"),
            "subject": sub.get("subject"),
            "notes": sub.get("notes"),
            "status": sub.get("status"),
            "category": (sub.get("getoption") or {}).get("feedback_option"),
            "unit": unit_info.get("unit"),
            "block": unit_info.get("building"),
            "user": user_info.get("name"),
            "created_at": sub.get("created_at"),
        })

    print(f"✅ Parsed {len(formatted)} feedback records")

    return {
        **state,
        "response": {
            "table": {
                "columns": [
                    "ticket",
                    "subject",
                    "notes",
                    "status",
                    "category",
                    "unit",
                    "block",
                    "user",
                    "created_at"
                ],
                "rows": formatted
            },
            "total": len(formatted)
        }
    }


# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(FeedbackSearchState)

workflow.add_node("feedback_search", feedback_search_node)

workflow.set_entry_point("feedback_search")

workflow.add_edge("feedback_search", END)

graph = workflow.compile()

print("✅ Feedback Search Agent Ready")