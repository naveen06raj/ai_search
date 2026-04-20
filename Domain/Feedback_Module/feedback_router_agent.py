from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from .feedback_prompts import get_feedback_route_prompt

# import TASK AGENTS
from .Agents.feedback_search_agent import graph as feedback_search_graph


# =====================================================
# STATE
# =====================================================
class FeedbackRouterState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    token: str
    login_id: int
    feedback_action: str
    response: Dict[str, Any]


# =====================================================
# LLM & PROMPT
# =====================================================
llm = get_chat_model()
route_prompt = get_feedback_route_prompt()


# =====================================================
# ROUTER NODE
# =====================================================
def feedback_router_node(state: FeedbackRouterState) -> FeedbackRouterState:
    chain = route_prompt | llm | StrOutputParser()

    action = (
        chain.invoke({"user_query": state["user_query"]})
        .strip()
        .lower()
    )

    print(f"\n🎯 [Feedback Router] Action determined: {action}")
    
    return {
        **state,
        "feedback_action": action,
    }


# =====================================================
# ROUTE DECISION
# =====================================================
def feedback_route_decision(state: FeedbackRouterState):
    action = state["feedback_action"]

    if action in {
        "feedback_search",
        "feedback_summary",
        "feedback_update",
    }:
        return action

    return "feedback_search"


# =====================================================
# TASK NODES
# =====================================================
async def feedback_search_node(state: FeedbackRouterState) -> FeedbackRouterState:
    print(f"\n🚀 [Feedback Router] Executing feedback search for: {state['user_query']}")

    result = await feedback_search_graph.ainvoke({
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    })

    total = result.get("response", {}).get("total", 0)

    print(f"✅ [Feedback Router] Search completed, found {total} results")

    return {
        **state,
        "response": result.get("response", {})
    }


async def feedback_summary_node(state: FeedbackRouterState) -> FeedbackRouterState:
    print(f"\n⚠️ [Feedback Router] Summary not implemented yet")

    return {
        **state,
        "response": {"message": "Feedback summary coming soon"}
    }


async def feedback_update_node(state: FeedbackRouterState) -> FeedbackRouterState:
    print(f"\n⚠️ [Feedback Router] Update not implemented yet")

    return {
        **state,
        "response": {"message": "Feedback update coming soon"}
    }


# =====================================================
# GRAPH
# =====================================================
workflow = StateGraph(FeedbackRouterState)

workflow.add_node("feedback_router", feedback_router_node)

workflow.add_node("feedback_search", feedback_search_node)
workflow.add_node("feedback_summary", feedback_summary_node)
workflow.add_node("feedback_update", feedback_update_node)

workflow.add_conditional_edges(
    "feedback_router",
    feedback_route_decision,
    {
        "feedback_search": "feedback_search",
        "feedback_summary": "feedback_summary",
        "feedback_update": "feedback_update",
    }
)

workflow.add_edge("feedback_search", END)
workflow.add_edge("feedback_summary", END)
workflow.add_edge("feedback_update", END)

workflow.set_entry_point("feedback_router")

graph = workflow.compile()

print("✅ Feedback Router Agent compiled successfully")