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
    category: Optional[int] = None
    building: Optional[str] = None

    token: str
    login_id: int


# =====================================================
# DATE PARSER (🔥 NEW)
# =====================================================
def parse_dates_from_query(query: str):
    query = query.lower()
    today = datetime.today()

    fromdate = None
    todate = None

    # today / now
    if any(word in query for word in ["today", "now", "still"]):
        todate = today

    # yesterday
    if "yesterday" in query:
        fromdate = today - timedelta(days=1)
        todate = fromdate

    # last 7 days
    if "last 7 days" in query:
        fromdate = today - timedelta(days=7)
        todate = today

    # last month
    if "last month" in query:
        first_day_this_month = today.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        fromdate = last_day_last_month.replace(day=1)
        todate = last_day_last_month

    # jan 2026
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
        k: v for k, v in filters.items()
        if v not in ["null", None, ""]
    }

    # =====================================================
    # 🔥 DATE FIX (IMPORTANT)
    # =====================================================
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

    # =====================================================
    # CATEGORY FIX
    # =====================================================
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

    # =====================================================
    # API CALL
    # =====================================================
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

    # =====================================================
    # FORMAT RESPONSE
    # =====================================================
    records = result_dict.get("data", [])

    formatted = []

    for r in records:
        sub = r.get("submissions") or {}
        typ = r.get("type") or {}
        unit = r.get("unit_info") or {}
        user = r.get("user_info") or {}

        formatted.append({
            "booking_id": sub.get("id"),
            "facility": typ.get("facility_type"),
            "booking_date": sub.get("booking_date"),
            "booking_time": sub.get("booking_time"),
            "status": sub.get("status"),
            "unit": unit.get("unit"),
            "block": unit.get("building"),
            "user": user.get("first_name"),
        })

    return {
        **state,
        "response": {
            "table": {
                "columns": [
                    "booking_id", "facility", "booking_date",
                    "booking_time", "status", "unit", "block", "user"
                ],
                "rows": formatted
            },
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