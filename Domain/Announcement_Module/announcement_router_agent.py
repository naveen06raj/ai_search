from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from .announcement_prompts import get_announcement_route_prompt

# import TASK AGENT
from .Agents.announcement_search_agent import graph as announcement_search_graph


# =====================================================
# STATE
# =====================================================
class AnnouncementRouterState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    token: str
    login_id: int
    announcement_action: str
    response: Dict[str, Any]


# =====================================================
# LLM & PROMPT
# =====================================================
llm = get_chat_model()
route_prompt = get_announcement_route_prompt()


# =====================================================
# ROUTER NODE
# =====================================================
def announcement_router_node(state: AnnouncementRouterState) -> AnnouncementRouterState:
    chain = route_prompt | llm | StrOutputParser()

    action = (
        chain.invoke({"user_query": state["user_query"]})
        .strip()
        .lower()
    )

    print(f"\n🎯 [Announcement Router] Action determined: {action}")

    return {
        **state,
        "announcement_action": action,
    }


# =====================================================
# ROUTE DECISION
# =====================================================
def announcement_route_decision(state: AnnouncementRouterState):
    action = state["announcement_action"]

    if action == "announcement_search":
        return "announcement_search"

    return "announcement_search"  # fallback


# =====================================================
# TASK NODE
# =====================================================
async def announcement_search_node(state: AnnouncementRouterState) -> AnnouncementRouterState:
    print(f"\n🚀 [Announcement Router] Executing search for: {state['user_query']}")

    result = await announcement_search_graph.ainvoke({
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    })

    total = result.get("response", {}).get("total", 0)

    print(f"✅ [Announcement Router] Search completed, found {total} results")

    return {
        **state,
        "response": result.get("response", {})
    }


# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(AnnouncementRouterState)

workflow.add_node("announcement_router", announcement_router_node)

workflow.add_node("announcement_search", announcement_search_node)

workflow.add_conditional_edges(
    "announcement_router",
    announcement_route_decision,
    {
        "announcement_search": "announcement_search",
    }
)

workflow.add_edge("announcement_search", END)

workflow.set_entry_point("announcement_router")

graph = workflow.compile()

print("✅ Announcement Router Agent compiled successfully")