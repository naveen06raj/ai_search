from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from datetime import datetime, timedelta
import re

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
# DATE PARSER
# =====================================================
def parse_dates_from_query(query: str):
    query = query.lower()
    today = datetime.today()

    fromdate = None
    todate = None

    if any(word in query for word in ["today", "now", "still"]):
        todate = today

    if "yesterday" in query:
        fromdate = today - timedelta(days=1)
        todate = fromdate

    if "last 7 days" in query:
        fromdate = today - timedelta(days=7)
        todate = today

    if "last month" in query:
        first_day_this_month = today.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        fromdate = last_day_last_month.replace(day=1)
        todate = last_day_last_month

    match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{4})", query)
    if match:
        month_str, year = match.groups()
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        month = month_map[month_str]
        fromdate = datetime(int(year), month, 1)

    return {
        "fromdate": fromdate.strftime("%Y-%m-%d") if fromdate else None,
        "todate": todate.strftime("%Y-%m-%d") if todate else None
    }

# =====================================================
# HELPERS
# =====================================================
def normalize_defect_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve nested defect format if the upstream API already returns it.
    Otherwise wrap flat rows into the nested structure.
    """
    if isinstance(record, dict) and "lists" in record:
        return {
            "lists": record.get("lists"),
            "user_info": record.get("user_info"),
            "unit_info": record.get("unit_info"),
            "inspection": record.get("inspection"),
        }

    return {
        "lists": {
            "ticket_no": record.get("ticket_no", ""),
            "status": record.get("status", ""),
            "block": record.get("block", ""),
            "unit_no": record.get("unit_no", ""),
            "submitted_date": record.get("submitted_date", ""),
            "appointment_status": record.get("appointment_status", ""),
            "completion_date": record.get("completion_date", "")
        },
        "user_info": None,
        "unit_info": None,
        "inspection": None
    }

# =====================================================
# NODE
# =====================================================
async def defect_search_node(state: DefectSearchState) -> Dict[str, Any]:
    from MCP.defect_mcp_server import search_defects

    # STEP 1: LLM CALL
    chain = prompt | llm | parser
    llm_output = await chain.ainvoke({
        "user_query": state["user_query"]
    })

    filters = llm_output["filters"]
    print(f"\n📋 [Defect Search] Extracted filters: {filters}")

    # CLEAN FILTERS
    cleaned_filters = {}
    for key, value in filters.items():
        cleaned_filters[key] = None if value in ["null", None] else value

    # DATE FIX
    date_fix = parse_dates_from_query(state["user_query"])

    if date_fix["fromdate"]:
        cleaned_filters["fromdate"] = date_fix["fromdate"]

    if date_fix["todate"]:
        cleaned_filters["todate"] = date_fix["todate"]

    # VALIDATE RANGE
    fd = cleaned_filters.get("fromdate")
    td = cleaned_filters.get("todate")

    if fd and td and fd > td:
        print("⚠️ Fixing invalid date range")
        cleaned_filters["todate"] = datetime.today().strftime("%Y-%m-%d")

    # Convert block_no to int
    if isinstance(cleaned_filters.get("block_no"), str):
        try:
            cleaned_filters["block_no"] = int(cleaned_filters["block_no"])
        except Exception:
            cleaned_filters["block_no"] = None

    status_filter = cleaned_filters.get("status")

    print("🔍 [Defect Search] Calling internal function: search_defects")

    try:
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
            "response": {
                "data": [],
                "chart": {"type": "bar", "data": {}},
                "total": 0,
                "filter_applied": False,
                "original_count": None
            }
        }

    # POST PROCESSING
    records = result_dict.get("records", [])
    original_count = len(records)

    if status_filter in STATUS_MAP:
        target_status = STATUS_MAP[status_filter]
        print(f"🎯 Applying status filter: {target_status}")

        records = [r for r in records if r.get("status") == target_status]
        status_summary = {target_status: len(records)}
    else:
        status_summary = result_dict.get("status_summary", {})

    total = len(records)

    formatted_data = [normalize_defect_record(r) for r in records]

    return {
        **state,
        "response": {
            "data": formatted_data,
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

print("✅ Defect Search Agent compiled (updated nested response + chart + total)")