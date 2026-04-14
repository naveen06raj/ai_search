from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import asyncio

from Core.model_registry import get_chat_model

# =====================================================
# STATE
# =====================================================
class DefectSearchState(TypedDict):
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
You are a Defect Search Agent.
Extract defect search filters from the user query.

Return ONLY valid JSON:

{{
  "filters": {{
    "fromdate": null,
    "todate": null,
    "status": null,
    "block_no": null,
    "unit": null,
    "ticket": null,
    "location": null,
    "type": null
  }}
}}

Rules:
- status: 0=open, 1=closed, 2=on schedule, 3=in progress
- Dates: YYYY-MM-DD
- Use null if missing
"""
        ),
        ("human", "{user_query}")
    ]
)

parser = JsonOutputParser()

# =====================================================
# INPUT MODEL
# =====================================================
class DefectSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    ticket: Optional[str] = None
    status: Optional[int] = None
    location: Optional[str] = None
    type: Optional[str] = None
    block_no: Optional[int] = None

    token: str
    login_id: int

# =====================================================
# STATUS MAP
# =====================================================
STATUS_MAP = {
    0: "OPEN",
    1: "CLOSED",
    2: "ON SCHEDULE",
    3: "IN PROGRESS",
    4: "COMPLETED - PENDING RESIDENT UPDATE"
}

# =====================================================
# NODE
# =====================================================
async def defect_search_node(state: DefectSearchState) -> Dict[str, Any]:
    from MCP.defect_mcp_server import search_defects

    # 🔥 STEP 1: LLM CALL (ASYNC)
    chain = prompt | llm | parser

    llm_output = await chain.ainvoke({
        "user_query": state["user_query"]
    })

    filters = llm_output["filters"]

    print(f"\n📋 [Defect Search] Extracted filters: {filters}")

    # 🔧 CLEAN FILTERS
    cleaned_filters = {}
    for key, value in filters.items():
        cleaned_filters[key] = None if value in ["null", None] else value

    # Convert block_no to int
    if isinstance(cleaned_filters.get("block_no"), str):
        try:
            cleaned_filters["block_no"] = int(cleaned_filters["block_no"])
        except:
            cleaned_filters["block_no"] = None

    status_filter = cleaned_filters.get("status")

    print(f"🔍 [Defect Search] Calling internal function: search_defects")

    try:
        # 🔥 STEP 2: API CALL (ASYNC)
        result_obj = await search_defects(
            DefectSearchInput(
                **cleaned_filters,
                token=state.get("token"),
                login_id=state.get("login_id")
            )
        )

        result_dict = result_obj.model_dump()

    except Exception as e:
        print(f"❌ [Defect Search] Internal Tool Error: {e}")
        return {
            **state,
            "response": {"message": f"Tool error: {str(e)}", "total": 0}
        }

    # =====================================================
    # POST PROCESSING
    # =====================================================
    records = result_dict.get("records", [])
    original_count = len(records)

    if status_filter in STATUS_MAP:
        target_status = STATUS_MAP[status_filter]
        print(f"🎯 Applying status filter: {target_status}")

        records = [
            r for r in records if r.get("status") == target_status
        ]
        status_summary = {target_status: len(records)}
    else:
        status_summary = result_dict.get("status_summary", {})

    total = len(records)

    formatted_records = [
        {
            "ticket_no": r.get("ticket_no", ""),
            "status": r.get("status", ""),
            "block": r.get("block", ""),
            "unit_no": r.get("unit_no", ""),
            "submitted_date": r.get("submitted_date", ""),
            "appointment_status": r.get("appointment_status", ""),
            "completion_date": r.get("completion_date", ""),
        }
        for r in records
    ]

    return {
        **state,
        "response": {
            "table": {
                "columns": [
                    "ticket_no", "status", "block", "unit_no",
                    "submitted_date", "appointment_status", "completion_date"
                ],
                "rows": formatted_records,
            },
            "chart": {
                "type": "bar",
                "data": status_summary,
            },
            "total": total,
            "filter_applied": status_filter is not None,
            "original_count": original_count if status_filter is not None else None,
        }
    }

# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(DefectSearchState)
workflow.add_node("defect_search", defect_search_node)
workflow.set_entry_point("defect_search")
workflow.add_edge("defect_search", END)

graph = workflow.compile()

print("✅ Defect Search Agent compiled (FIXED)")