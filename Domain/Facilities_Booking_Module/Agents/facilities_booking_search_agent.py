from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import httpx
from datetime import datetime, timedelta
import re

from Core.model_registry import get_chat_model


# =====================================================
# STATE
# =====================================================
class FacilitiesBookingSearchState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    response: Dict[str, Any]
    token: str
    login_id: int


# =====================================================
# LLM + PROMPT
# =====================================================
llm = get_chat_model()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Facilities Booking Search Agent.

Extract filters from user query.

Return ONLY JSON:

{{
  "filters": {{
    "fromdate": null,
    "todate": null,
    "unit": null,
    "status": null,
    "category": null,
    "building": null
  }}
}}

Rules:
- category = facility type (BBQ, Game Room, Swimming Pool)
- Dates format: YYYY-MM-DD
- DO NOT return text
"""
        ),
        ("human", "{user_query}")
    ]
)

parser = JsonOutputParser()


# =====================================================
# INPUT MODEL
# =====================================================
class FacilitiesBookingSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    status: Optional[int] = None
    category: Optional[int] = None   # type_id
    building: Optional[str] = None

    token: str
    login_id: int


# =====================================================
# DATE PARSER
# =====================================================
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def month_start_end(year: int, month: int):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    return start, end

def parse_dates_from_query(query: str):
    query = query.lower()
    today = datetime.today()

    fromdate = None
    todate = None

    if any(word in query for word in ["today", "now", "still"]):
        fromdate = today.replace(hour=0, minute=0, second=0, microsecond=0)
        todate = today

    elif "yesterday" in query:
        fromdate = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        todate = today

    elif "last 7 days" in query:
        fromdate = today - timedelta(days=7)
        todate = today

    elif any(word in query for word in ["this month", "current month"]):
        fromdate = datetime(today.year, today.month, 1)
        todate = today

    elif any(word in query for word in ["last month", "previous month"]):
        year = today.year
        month = today.month - 1
        if month == 0:
            month = 12
            year -= 1
        fromdate, todate = month_start_end(year, month)

    elif any(word in query for word in [
        "last 3 months",
        "last three months",
        "recent months",
        "latest months"
    ]):
        year = today.year
        month = today.month - 2
        while month <= 0:
            month += 12
            year -= 1
        fromdate = datetime(year, month, 1)
        todate = today

    else:
        match = re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+month)?(?:\s+(\d{4}))?",
            query
        )
        if match:
            month_str, year_str = match.groups()
            month = MONTH_MAP[month_str]
            year = int(year_str) if year_str else today.year
            fromdate, todate = month_start_end(year, month)

    return {
        "fromdate": fromdate.strftime("%Y-%m-%d") if fromdate else None,
        "todate": todate.strftime("%Y-%m-%d") if todate else None
    }


# =====================================================
# CATEGORY MAP API
# =====================================================
async def get_facility_category_map(token: str, login_id: int) -> Dict[str, int]:
    url = "https://aerea.panzerplayground.com/api/ops/v4/facilityoptions"

    payload = {"login_id": login_id}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=payload, headers=headers)

    data = response.json()
    return {v.lower(): int(k) for k, v in data.get("options", {}).items()}


# =====================================================
# NODE
# =====================================================
async def facilities_booking_search_node(state: FacilitiesBookingSearchState) -> Dict[str, Any]:
    from MCP.defect_mcp_server import search_facilities_booking

    chain = prompt | llm | parser

    # LLM
    try:
        llm_output = await chain.ainvoke({"user_query": state["user_query"]})
    except Exception as e:
        print(f"⚠️ LLM failed: {e}")
        llm_output = {"filters": {}}

    filters = llm_output.get("filters", {})
    print(f"\n📋 [Facility Booking] Filters: {filters}")

    # CLEAN
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

    # CATEGORY FIX
    try:
        category_map = await get_facility_category_map(
            state.get("token"),
            state.get("login_id")
        )

        cat = cleaned_filters.get("category")
        if isinstance(cat, str):
            cat_lower = cat.lower()

            for name, cid in category_map.items():
                if cat_lower in name:
                    cleaned_filters["category"] = cid
                    break
            else:
                cleaned_filters.pop("category", None)

    except Exception as e:
        print(f"⚠️ Category mapping failed: {e}")

    print(f"🧹 Cleaned Filters: {cleaned_filters}")

    # API CALL
    try:
        result_obj = await search_facilities_booking(
            FacilitiesBookingSearchInput(
                **cleaned_filters,
                token=state.get("token"),
                login_id=state.get("login_id")
            )
        )

        result_dict = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj

    except Exception as e:
        return {
            **state,
            "response": {"message": str(e), "total": 0}
        }

    # FORMAT RESPONSE
    records = result_dict.get("data", [])
    if not records and isinstance(result_dict.get("table"), dict):
        records = result_dict.get("table", {}).get("rows", [])

    formatted = []

    for r in records:
        submissions = r.get("submissions") or {}
        facility_type = r.get("type") or {}
        unit_info = r.get("unit_info") or {}
        user_info = r.get("user_info") or {}

        formatted.append({
            "submissions": submissions,
            "type": facility_type,
            "unit_info": unit_info,
            "user_info": user_info
        })

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
workflow = StateGraph(FacilitiesBookingSearchState)

workflow.add_node("facilities_booking_search", facilities_booking_search_node)

workflow.set_entry_point("facilities_booking_search")

workflow.add_edge("facilities_booking_search", END)

graph = workflow.compile()

print("✅ Facilities Booking Search Agent Ready")