from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from Core.model_registry import get_chat_model


# =====================================================
# STATE
# =====================================================
class DefectSearchState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    response: Dict[str, Any]


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
# MCP CLIENT
# =====================================================
mcp_client = MultiServerMCPClient(
    {
        "defect": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8002/mcp"
        }
    }
)


# =====================================================
# INPUT MODEL FOR MCP TOOL
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


# =====================================================
# STATUS MAP FOR POST-FILTERING
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
    # 1. Extract filters using LLM
    chain = prompt | llm | parser
    llm_output = await chain.ainvoke(
        {"user_query": state["user_query"]}
    )

    filters = llm_output["filters"]
    
    # Debug: Show what filters we extracted
    print(f"\n📋 [Defect Search] Extracted filters from query:")
    for key, value in filters.items():
        print(f"  {key}: {value}")
    
    # Clean up the filters
    cleaned_filters = {}
    for key, value in filters.items():
        if value == "null" or value is None:
            cleaned_filters[key] = None
        else:
            cleaned_filters[key] = value
    
    # Convert block_no to int if present
    if cleaned_filters.get("block_no") and isinstance(cleaned_filters["block_no"], str):
        try:
            cleaned_filters["block_no"] = int(cleaned_filters["block_no"])
        except ValueError:
            cleaned_filters["block_no"] = None
    
    # Store the status filter for post-filtering
    status_filter = cleaned_filters.get("status")

    # 2. Call MCP tool with proper Pydantic model input
    tools = await mcp_client.get_tools()
    search_tool = next(t for t in tools if t.name == "search_defects")

    tool_input = DefectSearchInput(**cleaned_filters)
    
    print(f"\n🔍 [Defect Search] Calling MCP tool with: {tool_input.model_dump()}")

    tool_result = await search_tool.ainvoke(
        {"input": tool_input.model_dump()}
    )

    # Handle the result - it's wrapped in a list by LangGraph
    if isinstance(tool_result, list) and len(tool_result) > 0:
        result_item = tool_result[0]
    else:
        result_item = tool_result
    
    # Extract the JSON string from the 'text' field
    if isinstance(result_item, dict) and 'text' in result_item:
        try:
            result_dict = json.loads(result_item['text'])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            result_dict = {}
    else:
        result_dict = result_item if isinstance(result_item, dict) else {}
    
    # 3. POST-FILTERING: Apply status filter locally if API didn't filter
    records = result_dict.get("records", [])
    original_count = len(records)
    
    if status_filter is not None and status_filter in STATUS_MAP:
        target_status = STATUS_MAP[status_filter]
        print(f"\n🎯 [Defect Search] Applying status filter: {status_filter} -> '{target_status}'")
        print(f"   Records before filtering: {original_count}")
        
        # Filter records by status
        filtered_records = [
            record for record in records 
            if record.get("status") == target_status
        ]
        
        records = filtered_records
        print(f"   Records after filtering: {len(records)}")
        
        # Update status_summary to only show filtered status
        status_summary = {target_status: len(records)}
    else:
        status_summary = result_dict.get("status_summary", {})
    
    total = len(records)
    
    print(f"\n📊 [Defect Search] Final results: {total} records")
    
    # Format records for the table
    formatted_records = []
    for record in records:
        formatted_records.append({
            "ticket_no": record.get("ticket_no", ""),
            "status": record.get("status", ""),
            "block": record.get("block", ""),
            "unit_no": record.get("unit_no", ""),
            "submitted_date": record.get("submitted_date", ""),
            "appointment_status": record.get("appointment_status", ""),
            "completion_date": record.get("completion_date", ""),
        })

    # Return the FULL state with response
    return {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "response": {
            "table": {
                "columns": [
                    "ticket_no",
                    "status",
                    "block",
                    "unit_no",
                    "submitted_date",
                    "appointment_status",
                    "completion_date",
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

print("✅ Defect Search Agent compiled")