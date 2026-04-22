from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from .facilities_booking_prompts import get_facilities_booking_route_prompt

# import TASK AGENT
from .Agents.facilities_booking_search_agent import graph as facilities_booking_search_graph


# =====================================================
# STATE
# =====================================================
class FacilitiesBookingRouterState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    token: str
    login_id: int
    facilities_booking_action: str
    response: Dict[str, Any]


# =====================================================
# LLM & PROMPT
# =====================================================
llm = get_chat_model()
route_prompt = get_facilities_booking_route_prompt()


# =====================================================
# ROUTER NODE
# =====================================================
def facilities_booking_router_node(state: FacilitiesBookingRouterState) -> FacilitiesBookingRouterState:
    chain = route_prompt | llm | StrOutputParser()

    action = (
        chain.invoke({"user_query": state["user_query"]})
        .strip()
        .lower()
    )

    print(f"\n🎯 [Facilities Router] Action determined: {action}")

    return {
        **state,
        "facilities_booking_action": action,
    }


# =====================================================
# ROUTE DECISION
# =====================================================
def facilities_booking_route_decision(state: FacilitiesBookingRouterState):
    action = state["facilities_booking_action"]

    if action == "facilities_booking_search":
        return "facilities_booking_search"

    return "facilities_booking_search"   # fallback


# =====================================================
# TASK NODE
# =====================================================
async def facilities_booking_search_node(state: FacilitiesBookingRouterState) -> FacilitiesBookingRouterState:
    print(f"\n🚀 [Facilities Router] Executing search for: {state['user_query']}")

    result = await facilities_booking_search_graph.ainvoke({
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    })

    total = result.get("response", {}).get("total", 0)

    print(f"✅ [Facilities Router] Search completed, found {total} results")

    return {
        **state,
        "response": result.get("response", {})
    }


# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(FacilitiesBookingRouterState)

workflow.add_node("facilities_booking_router", facilities_booking_router_node)

workflow.add_node("facilities_booking_search", facilities_booking_search_node)

workflow.add_conditional_edges(
    "facilities_booking_router",
    facilities_booking_route_decision,
    {
        "facilities_booking_search": "facilities_booking_search",
    }
)

workflow.add_edge("facilities_booking_search", END)

workflow.set_entry_point("facilities_booking_router")

graph = workflow.compile()

print("✅ Facilities Booking Router Agent compiled successfully")