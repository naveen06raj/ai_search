from typing import TypedDict, List, Dict, Any
import logging

from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

from Core.model_registry import get_chat_model
from ..defect_magic_prompts import get_defect_magic_prompt


# =========================
# STATE
# =========================
class DefectMagicState(TypedDict):
    user_input: str
    extracted: List[Dict[str, Any]]


# =========================
# INIT
# =========================
llm = get_chat_model()
prompt = get_defect_magic_prompt()
parser = JsonOutputParser()


# =========================
# NODE
# =========================
async def defect_magic_node(state: DefectMagicState):

    try:
        chain = prompt | llm | parser

        result = await chain.ainvoke({
            "user_input": state["user_input"]
        })

        logging.info(f"✨ [AI MAGIC] Extracted: {result}")

        # Safety: ensure list
        if not isinstance(result, list):
            result = [result]

        return {
            **state,
            "extracted": result
        }

    except Exception as e:
        logging.error(f"❌ AI Magic Error: {e}")

        # fallback response
        return {
            **state,
            "extracted": [
                {
                    "location": None,
                    "type": None,
                    "remarks": state["user_input"]
                }
            ]
        }


# =========================
# GRAPH
# =========================
workflow = StateGraph(DefectMagicState)

workflow.add_node("defect_magic", defect_magic_node)

workflow.set_entry_point("defect_magic")
workflow.add_edge("defect_magic", END)

graph = workflow.compile()

logging.info("✅ Defect AI Magic Autofill Agent Ready")