from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from Orchestration.orchestration_prompts import get_route_prompt, get_search_intent_prompt
from Domain.Defect_Module.Agents.defect_search_agent import graph as defect_search_graph
from Domain.Feedback_Module.Agents.feedback_search_agent import graph as feedback_search_graph
from Domain.Facilities_Booking_Module.Agents.facilities_booking_search_agent import graph as facilities_search_graph
from Domain.Announcement_Module.Agents.announcement_search_agent import graph as announcement_search_graph


# ------------------ STATE ------------------
class OrchestrationState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    route: str
    defect_action: str
    feedback_action: str
    facility_action: str
    announcement_action: str
    response: Dict[str, Any]

    token: str
    login_id: int
    current_module: Optional[str]


# ------------------ LLM ------------------
llm = get_chat_model()
route_prompt = get_route_prompt()
intent_prompt = get_search_intent_prompt()


# ------------------ HELPERS ------------------
def module_to_route(module: str) -> Optional[str]:
    module = (module or "").lower().strip()

    mapping = {
        "defect": "defect_domain",
        "feedback": "feedback_domain",
        "facility": "facility_booking_domain",
        "announcement": "announcement_domain",
    }

    return mapping.get(module)


def route_to_module(route: str) -> Optional[str]:
    route = (route or "").lower().strip()

    mapping = {
        "defect_domain": "defect",
        "feedback_domain": "feedback",
        "facility_booking_domain": "facility",
        "announcement_domain": "announcement",
    }

    return mapping.get(route)


# ------------------ ORCHESTRATOR NODE ------------------
def orchestration_node(state: OrchestrationState) -> OrchestrationState:
    user_query = state["user_query"]
    current_module = (state.get("current_module") or "").lower().strip()

    # 1) Detect intent first: search_query / general_question / unclear
    intent_chain = intent_prompt | llm | StrOutputParser()
    intent = intent_chain.invoke({
        "user_query": user_query
    }).strip().lower()

    print(f"\n🧠 [Orchestrator] Intent determined: {intent}")

    # 2) If it is not a search request, always clarify
    if intent in {"general_question", "unclear"}:
        route = "clarify_query"
        print(f"\n❓ [Orchestrator] Non-search question detected -> clarify_query")

    else:
        # 3) If current module is known, validate that the query belongs to it
        current_route = module_to_route(current_module)

        if current_route:
            # Ask LLM to detect the actual domain for the query
            route_chain = route_prompt | llm | StrOutputParser()
            detected_route = route_chain.invoke({
                "user_query": user_query,
                "chat_history": state.get("chat_history", [])
            }).strip().lower()

            if detected_route == "general_response":
                detected_route = "clarify_query"

            print(f"\n🎯 [Orchestrator] Current module detected: {current_module} -> {current_route}")
            print(f"🎯 [Orchestrator] Query detected route: {detected_route}")

            # 4) If user asks for a different module, block it
            if detected_route != current_route:
                route = "clarify_query"
                print(f"\n⚠️ [Orchestrator] Module mismatch -> clarify_query")
            else:
                route = current_route
                print(f"\n✅ [Orchestrator] Module matches query -> {route}")

        # 5) No module context, fall back to normal routing
        else:
            chain = route_prompt | llm | StrOutputParser()
            route = chain.invoke({
                "user_query": user_query,
                "chat_history": state.get("chat_history", [])
            }).strip().lower()

            if route == "general_response":
                route = "clarify_query"

            print(f"\n🎯 [Orchestrator] Route determined by LLM: {route}")

    return {
        "user_query": user_query,
        "chat_history": state.get("chat_history", []) + [
            AIMessage(content=f"Routing decision: {route}")
        ],
        "route": route,
        "defect_action": state.get("defect_action", ""),
        "feedback_action": state.get("feedback_action", ""),
        "facility_action": state.get("facility_action", ""),
        "announcement_action": state.get("announcement_action", ""),
        "response": state.get("response", {}),
        "token": state.get("token"),
        "login_id": state.get("login_id"),
        "current_module": current_module
    }


# ------------------ ROUTE DECISION ------------------
def route_decision(state: OrchestrationState):
    route = state["route"]

    if route == "general_response":
        return "clarify_query"

    if route in {
        "defect_domain",
        "device_management_domain",
        "feedback_domain",
        "facility_booking_domain",
        "announcement_domain",
        "clarify_query",
        "continue_conversation",
    }:
        return route

    return "error"


# ------------------ DOMAIN NODES ------------------
async def defect_domain_node(state: OrchestrationState):
    print(f"\n🚀 [Orchestrator] Entering Defect Domain for: {state['user_query']}")

    defect_router_input = {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    }

    
    defect_result = await defect_search_graph.ainvoke(defect_router_input)

    print(f"✅ [Orchestrator] Defect Domain completed, response ready")

    return {
        **state,
        "defect_action": defect_result.get("defect_action", ""),
        "response": defect_result.get("response", {})
    }


async def device_management_domain_node(state: OrchestrationState):
    print(f"\n⚠️ [Orchestrator] Device Management Domain not implemented")
    return {**state, "response": {"message": "Device management coming soon"}}


async def feedback_domain_node(state: OrchestrationState):
    print(f"\n🚀 [Orchestrator] Entering Feedback Domain for: {state['user_query']}")

    feedback_input = {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    }

    result = await feedback_search_graph.ainvoke(feedback_input)

    print(f"✅ [Orchestrator] Feedback Domain completed")

    return {
        **state,
        "feedback_action": result.get("feedback_action", ""),
        "response": result.get("response", {})
    }


async def facility_booking_domain_node(state: OrchestrationState):
    print(f"\n🚀 [Orchestrator] Entering Facility Domain for: {state['user_query']}")

    facility_input = {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    }

    result = await facilities_search_graph.ainvoke(facility_input)

    print(f"✅ [Orchestrator] Facility Domain completed")

    return {
        **state,
        "facility_action": result.get("facility_action", ""),
        "response": result.get("response", {})
    }


async def announcement_domain_node(state: OrchestrationState):
    print(f"\n🚀 [Orchestrator] Entering Announcement Domain for: {state['user_query']}")

    announcement_input = {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []),
        "token": state.get("token"),
        "login_id": state.get("login_id")
    }

    result = await announcement_search_graph.ainvoke(announcement_input)

    print(f"✅ [Orchestrator] Announcement Domain completed")

    return {
        **state,
        "announcement_action": result.get("announcement_action", ""),
        "response": result.get("response", {})
    }


async def clarify_query_node(state: OrchestrationState):
    print(f"\n❓ [Orchestrator] Asking for clarification")

    current_module = (state.get("current_module") or "").lower().strip()

    module_message_map = {
        "defect": "This is for defect search only. Please ask only defect-related questions.",
        "feedback": "This is for feedback search only. Please ask only feedback-related questions.",
        "facility": "This is for facility booking only. Please ask only facility-related questions.",
        "announcement": "This is for announcement search only. Please ask only announcement-related questions.",
    }

    message = module_message_map.get(
        current_module,
        "I can help only with module search questions. Please ask a filter or search question."
    )

    return {
        **state,
        "response": {
            "message": message
        }
    }


async def continue_conversation_node(state: OrchestrationState):
    print(f"\n💬 [Orchestrator] Continuing conversation")
    return {**state, "response": {"message": "Let's continue your search."}}


def error_node(state: OrchestrationState):
    print(f"\n❌ [Orchestrator] Error handling")
    return {
        **state,
        "response": {
            "message": "I apologize, but I encountered an error processing your request."
        }
    }


# ------------------ GRAPH ------------------
workflow = StateGraph(OrchestrationState)

workflow.add_node("orchestrator", orchestration_node)
workflow.add_node("defect_domain", defect_domain_node)
workflow.add_node("device_management_domain", device_management_domain_node)
workflow.add_node("feedback_domain", feedback_domain_node)
workflow.add_node("facility_booking_domain", facility_booking_domain_node)
workflow.add_node("announcement_domain", announcement_domain_node)
workflow.add_node("clarify_query", clarify_query_node)
workflow.add_node("continue_conversation", continue_conversation_node)
workflow.add_node("error", error_node)

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "defect_domain": "defect_domain",
        "device_management_domain": "device_management_domain",
        "feedback_domain": "feedback_domain",
        "facility_booking_domain": "facility_booking_domain",
        "announcement_domain": "announcement_domain",
        "clarify_query": "clarify_query",
        "continue_conversation": "continue_conversation",
        "error": "error",
    }
)

for node in [
    "defect_domain",
    "device_management_domain",
    "feedback_domain",
    "facility_booking_domain",
    "announcement_domain",
    "clarify_query",
    "continue_conversation",
    "error",
]:
    workflow.add_edge(node, END)

workflow.set_entry_point("orchestrator")

graph = workflow.compile()

print("✅ Orchestration State Graph compiled successfully")