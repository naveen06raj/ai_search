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
class AnnouncementSearchState(TypedDict):
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
You are an Announcement Search Agent.

Extract filters from the user query.

Return ONLY valid JSON:

{{
  "filters": {{
    "startdate": null,
    "enddate": null,
    "roles": null,
    "status": null
  }}
}}

Rules:
- Dates format: YYYY-MM-DD
- roles = role name (Tenant, Owner, etc.)
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
class AnnouncementSearchInput(BaseModel):
    startdate: Optional[str] = None
    enddate: Optional[str] = None
    roles: Optional[int] = None
    status: Optional[int] = None

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

    startdate = None
    enddate = None

    if any(word in query for word in ["today", "now", "still"]):
        startdate = today.replace(hour=0, minute=0, second=0, microsecond=0)
        enddate = today

    elif "yesterday" in query:
        startdate = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        enddate = today

    elif "last 7 days" in query:
        startdate = today - timedelta(days=7)
        enddate = today

    elif any(word in query for word in ["this month", "current month"]):
        startdate = datetime(today.year, today.month, 1)
        enddate = today

    elif any(word in query for word in ["last month", "previous month"]):
        year = today.year
        month = today.month - 1
        if month == 0:
            month = 12
            year -= 1
        startdate, enddate = month_start_end(year, month)

    elif any(word in query for word in [
        "last 3 months",
        "last three months",
        "recent months",
        "latest months"
    ]):
        # from 2 months back up to today (3 months including current month)
        year = today.year
        month = today.month - 2
        while month <= 0:
            month += 12
            year -= 1
        startdate = datetime(year, month, 1)
        enddate = today

    else:
        match = re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+month)?(?:\s+(\d{4}))?",
            query
        )
        if match:
            month_str, year_str = match.groups()
            month = MONTH_MAP[month_str]
            year = int(year_str) if year_str else today.year
            startdate, enddate = month_start_end(year, month)

    return {
        "startdate": startdate.strftime("%Y-%m-%d") if startdate else None,
        "enddate": enddate.strftime("%Y-%m-%d") if enddate else None
    }


# =====================================================
# NODE
# =====================================================
async def announcement_search_node(state: AnnouncementSearchState) -> Dict[str, Any]:
    from MCP.defect_mcp_server import search_announcements, get_roles_list

    # STEP 1: LLM
    chain = prompt | llm | parser

    llm_output = await chain.ainvoke({
        "user_query": state["user_query"]
    })

    filters = llm_output.get("filters", {})
    print(f"\n📋 [Announcement] Filters: {filters}")

    # CLEAN FILTERS
    cleaned_filters = {
        k: None if v in ["null", None, ""] else v
        for k, v in filters.items()
    }

    # DATE FIX
    date_fix = parse_dates_from_query(state["user_query"])

    if date_fix["startdate"]:
        cleaned_filters["startdate"] = date_fix["startdate"]

    if date_fix["enddate"]:
        cleaned_filters["enddate"] = date_fix["enddate"]

    # GET ROLES MAP
    roles_map = await get_roles_list(
        token=state.get("token"),
        login_id=state.get("login_id")
    )

    # Convert role name -> id if needed
    role_name = cleaned_filters.get("roles")
    if isinstance(role_name, str):
        for k, v in roles_map.items():
            if str(v).lower() == role_name.lower():
                cleaned_filters["roles"] = int(k)
                break

    print(f"🧹 Cleaned Filters: {cleaned_filters}")

    # CALL API
    try:
        result_obj = await search_announcements(
            AnnouncementSearchInput(
                **cleaned_filters,
                token=state.get("token"),
                login_id=state.get("login_id")
            )
        )

        result_dict = result_obj if isinstance(result_obj, dict) else result_obj.model_dump()

    except Exception as e:
        print(f"❌ Error: {e}")
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
        formatted.append({
            "id": r.get("id"),
            "account_id": r.get("account_id"),
            "title": r.get("title"),
            "document": r.get("document"),
            "notes": r.get("notes"),
            "roles": r.get("roles"),
            "upload": r.get("upload"),
            "upload_2": r.get("upload_2"),
            "upload_3": r.get("upload_3"),
            "upload_4": r.get("upload_4"),
            "upload_5": r.get("upload_5"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at")
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
workflow = StateGraph(AnnouncementSearchState)

workflow.add_node("announcement_search", announcement_search_node)

workflow.set_entry_point("announcement_search")

workflow.add_edge("announcement_search", END)

graph = workflow.compile()

print("✅ Announcement Search Agent Ready")