from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from Orchestration.orchestration_prompts import get_route_prompt, get_greeting_prompt
from Domain.Defect_Module.defect_router_agent import graph as defect_router_graph



# ------------------ STATE ------------------
class OrchestrationState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    route: str
    defect_action: str 
    response: Dict[str, Any]  # Add response field


# ------------------ LLM ------------------
llm = get_chat_model()
route_prompt = get_route_prompt()


# ------------------ ORCHESTRATOR NODE ------------------
def orchestration_node(state: OrchestrationState) -> OrchestrationState:
    chain = route_prompt | llm | StrOutputParser()

    route = chain.invoke({
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", [])
    }).strip().lower()

    print(f"\n🎯 [Orchestrator] Route determined: {route}")
    
    return {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", []) + [
            AIMessage(content=f"Routing decision: {route}")
        ],
        "route": route,
        "defect_action": state.get("defect_action", ""),
        "response": state.get("response", {})
    }


# ------------------ ROUTE DECISION ------------------
def route_decision(state: OrchestrationState):
    route = state["route"]

    if route in {
        "defect_domain",
        "device_management_domain",
        "facility_booking_domain",
        "general_response",
        "clarify_query",
        "continue_conversation",
    }:
        return route

    return "error"


# ------------------ DOMAIN NODES ------------------
async def defect_domain_node(state: OrchestrationState):
    """
    This node hands over control to Defect Router Agent
    """
    print(f"\n🚀 [Orchestrator] Entering Defect Domain for: {state['user_query']}")

    defect_router_input = {
        "user_query": state["user_query"],
        "chat_history": state.get("chat_history", [])
    }

    defect_result = await defect_router_graph.ainvoke(defect_router_input)

    print(f"✅ [Orchestrator] Defect Domain completed, response ready")
    
    return {
        **state,
        "defect_action": defect_result["defect_action"],
        "response": defect_result.get("response", {})  # Get response from defect router
    }


def device_management_domain_node(state): 
    print(f"\n⚠️ [Orchestrator] Device Management Domain not implemented")
    return {**state, "response": {"message": "Device management coming soon"}}


def facility_booking_domain_node(state): 
    print(f"\n⚠️ [Orchestrator] Facility Booking Domain not implemented")
    return {**state, "response": {"message": "Facility booking coming soon"}}


def general_response_node(state): 
    print(f"\n💬 [Orchestrator] Generating general response for: {state['user_query']}")
    
    # Use the greeting prompt for better responses
    greeting_prompt = get_greeting_prompt()
    chain = greeting_prompt | llm | StrOutputParser()
    
    try:
        ai_response = chain.invoke({
            "query": state["user_query"]
        })
        
        # Return in expected response format
        return {
            **state, 
            "response": {
                "message": ai_response,
                "is_general_response": True
            }
        }
    except Exception as e:
        print(f"Error in general_response_node: {e}")
        # Fallback to simple response
        return {
            **state, 
            "response": {
                "message": "Hello! I'm here to help with defect management. You can ask me about defects, tickets, or maintenance issues.",
                "is_general_response": True
            }
        }


def clarify_query_node(state): 
    print(f"\n❓ [Orchestrator] Asking for clarification")
    return {**state, "response": {"message": "Could you please clarify your question?"}}


def continue_conversation_node(state): 
    print(f"\n💬 [Orchestrator] Continuing conversation")
    return {**state, "response": {"message": "Let's continue our conversation."}}


def error_node(state): 
    print(f"\n❌ [Orchestrator] Error handling")
    return {**state, "response": {"message": "I apologize, but I encountered an error processing your request."}}


# ------------------ GRAPH ------------------
workflow = StateGraph(OrchestrationState)

workflow.add_node("orchestrator", orchestration_node)
workflow.add_node("defect_domain", defect_domain_node)
workflow.add_node("device_management_domain", device_management_domain_node)
workflow.add_node("facility_booking_domain", facility_booking_domain_node)
workflow.add_node("general_response", general_response_node)
workflow.add_node("clarify_query", clarify_query_node)
workflow.add_node("continue_conversation", continue_conversation_node)
workflow.add_node("error", error_node)

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "defect_domain": "defect_domain",
        "device_management_domain": "device_management_domain",
        "facility_booking_domain": "facility_booking_domain",
        "general_response": "general_response",
        "clarify_query": "clarify_query",
        "continue_conversation": "continue_conversation",
        "error": "error",
    }
)

# END connections
for node in [
    "defect_domain",
    "device_management_domain",
    "facility_booking_domain",
    "general_response",
    "clarify_query",
    "continue_conversation",
    "error",
]:
    workflow.add_edge(node, END)

workflow.set_entry_point("orchestrator")

graph = workflow.compile()

print("✅ Orchestration State Graph compiled successfully")


# =====================================================
# TEST FUNCTION
# =====================================================
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from pprint import pprint

    test_cases = [
        "Show open defects for block 6"
    ]

    async def run_tests():
        for query in test_cases:
            print("\n" + "="*60)
            print(f"🧪 TESTING: {query}")
            print("="*60)
            
            result = await graph.ainvoke({
                "user_query": query,
                "chat_history": [HumanMessage(content="Hi")]
            })

            print(f"\n📊 RESULTS:")
            print(f"Route: {result.get('route')}")
            print(f"Defect Action: {result.get('defect_action')}")
            print(f"Response keys: {result.get('response', {}).keys()}")
            
            # Display table if available
            if "table" in result.get("response", {}):
                response = result["response"]
                print(f"\n📋 TABLE SUMMARY:")
                print(f"Total records: {response.get('total')}")
                print(f"Columns: {response['table']['columns']}")
                print(f"Number of rows: {len(response['table']['rows'])}")
                
                # Show first 3 records
                print(f"\n📝 First 3 records:")
                for i, record in enumerate(response['table']['rows'][:3], 1):
                    print(f"  {i}. Ticket: {record.get('ticket_no')} | Status: {record.get('status')} | Block: {record.get('block')}")
            
            if "chart" in result.get("response", {}):
                print(f"\n📊 CHART DATA: {result['response']['chart']['data']}")

    asyncio.run(run_tests())