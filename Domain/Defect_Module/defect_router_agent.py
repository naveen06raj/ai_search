from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from .defect_prompts import get_defect_route_prompt

# import TASK AGENTS
from .Agents.defect_search_agent import graph as defect_search_graph
# future agents will be added later


# =====================================================
# STATE
# =====================================================
class DefectRouterState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    defect_action: str
    response: Dict[str, Any]  # This will hold the final response


# =====================================================
# LLM & PROMPT
# =====================================================
llm = get_chat_model()
route_prompt = get_defect_route_prompt()


# =====================================================
# ROUTER NODE (DECISION ONLY)
# =====================================================
def defect_router_node(state: DefectRouterState) -> DefectRouterState:
    chain = route_prompt | llm | StrOutputParser()

    action = (
        chain.invoke({"user_query": state["user_query"]})
        .strip()
        .lower()
    )

    print(f"\n🎯 [Defect Router] Action determined: {action}")
    
    return {
        **state,
        "defect_action": action,
    }


# =====================================================
# ROUTE DECISION
# =====================================================
def defect_route_decision(state: DefectRouterState):
    action = state["defect_action"]

    if action in {
        "defect_search",
        "defect_status_update",
        "defect_inspection",
    }:
        return action

    # Safe fallback
    return "defect_search"


# =====================================================
# TASK NODES
# =====================================================
async def defect_search_node(state: DefectRouterState) -> DefectRouterState:
    """Execute defect search and return updated state with response"""
    print(f"\n🚀 [Defect Router] Executing defect search for: {state['user_query']}")
    
    # Call the defect search graph
    search_result = await defect_search_graph.ainvoke({
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", [])
    })
    
    print(f"✅ [Defect Router] Search completed, found {search_result.get('response', {}).get('total', 0)} results")
    
    return {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "defect_action": state["defect_action"],
        "response": search_result.get("response", {})
    }


async def defect_update_node(state: DefectRouterState) -> DefectRouterState:
    """Placeholder for defect status update"""
    print(f"\n⚠️ [Defect Router] Defect status update not implemented yet")
    return {
        **state,
        "response": {"message": "Defect status update functionality coming soon"}
    }


async def defect_inspection_node(state: DefectRouterState) -> DefectRouterState:
    """Placeholder for defect inspection"""
    print(f"\n⚠️ [Defect Router] Defect inspection not implemented yet")
    return {
        **state,
        "response": {"message": "Defect inspection functionality coming soon"}
    }


# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(DefectRouterState)

# Level 2 router
workflow.add_node("defect_router", defect_router_node)

# Level 3 task agents (updated to async functions)
workflow.add_node("defect_search", defect_search_node)
workflow.add_node("defect_status_update", defect_update_node)
workflow.add_node("defect_inspection", defect_inspection_node)

# ROUTING
workflow.add_conditional_edges(
    "defect_router",
    defect_route_decision,
    {
        "defect_search": "defect_search",
        "defect_status_update": "defect_status_update",
        "defect_inspection": "defect_inspection",
    }
)

# Each task ends the graph
workflow.add_edge("defect_search", END)
workflow.add_edge("defect_status_update", END)
workflow.add_edge("defect_inspection", END)

workflow.set_entry_point("defect_router")

graph = workflow.compile()

print("✅ Defect Router Agent compiled successfully")