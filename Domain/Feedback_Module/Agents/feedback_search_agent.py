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

    chain = prompt | llm | parser

    llm_output = await chain.ainvoke({
        "user_query": state["user_query"]
    })

    filters = llm_output.get("filters", {})
    print(f"\n📋 [Feedback Search] Filters: {filters}")

    # CLEAN FILTERS
    cleaned_filters = {
        k: None if v in ["null", None, ""] else v
        for k, v in filters.items()
    }

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

    # CATEGORY MAP
    category_map = {
        "security": 1,
        "plumbing": 2,
        "lift": 32,
        "cleaning": 5,
        "electrical": 6
    }

    cat = cleaned_filters.get("category")
    if isinstance(cat, str):
        mapped = category_map.get(cat.lower())
        if mapped:
            cleaned_filters["category"] = mapped
        else:
            cleaned_filters["category"] = None

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

        if hasattr(result_obj, "model_dump"):
            result_dict = result_obj.model_dump()
        else:
            result_dict = result_obj

    except Exception as e:
        print(f"❌ [Feedback Search] Error: {e}")
        return {
            **state,
            "response": {"message": str(e), "total": 0}
        }

    # RAW RECORDS
    records = result_dict.get("data", [])
    formatted = []

    for r in records:
        submissions = dict(r.get("submissions") or {})
        option = r.get("option") or submissions.get("getoption") or {}
        user_info = r.get("user_info") or submissions.get("user") or {}
        unit_info = r.get("unit_info") or {}

        # Keep the nested structure exactly like you want
        submissions["getoption"] = option
        submissions["user"] = user_info

        formatted.append({
            "submissions": submissions,
            "option": option,
            "user_info": user_info,
            "unit_info": unit_info
        })

    print(f"✅ Parsed {len(formatted)} feedback records")

    return {
        **state,
        "response": {
            "data": formatted,
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