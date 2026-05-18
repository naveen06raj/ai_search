from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# ROUTER PROMPT TEMPLATE
# =====================================================
def get_route_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a GLOBAL AI ORCHESTRATION AGENT.\n"
            "Your job is to decide which DOMAIN should handle the user request.\n\n"

            "Available domains:\n"
            "- defect_domain : issues, defects, inspections, maintenance complaints\n"
            "- device_management_domain : devices, sensors, hardware, IoT, status\n"
            "- feedback_domain : feedback, complaints, user reports, submissions\n"
            "- facility_booking_domain : facility booking, amenities, reservations, BBQ, room booking, swimming pool, game room\n"
            "- announcement_domain : announcements, notices, updates, broadcasts, messages\n"
            "- clarify_query : greetings, general questions, missing or unclear information\n"
            "- continue_conversation : follow-up or refinement of previous request\n"
            "- error : invalid, unsafe, or unsupported request\n\n"

            "Routing Rules:\n"
            "- If query is about defects/issues → defect_domain\n"
            "- If query is about feedback/complaints → feedback_domain\n"
            "- If query is about facility booking (BBQ, room, pool, amenities, reservation, booking) → facility_booking_domain\n"
            "- If query is about announcements, notices, updates → announcement_domain\n"
            "- If query is a greeting or general question → clarify_query\n"
            "- If query is unclear or missing important information → clarify_query\n"
            "- If query is a follow-up to the previous request → continue_conversation\n\n"

            "STRICT RULES:\n"
            "- Return ONLY ONE value\n"
            "- Do NOT explain\n"
            "- Do NOT add extra text\n"
            "- Output must exactly match one option\n"
        ),
        (
            "human",
            "Chat history:\n{chat_history}\n\nUser query:\n{user_query}"
        )
    ])


# =====================================================
# SEARCH / GENERAL INTENT PROMPT
# =====================================================
def get_search_intent_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an intent classifier for a module-based search assistant.\n\n"
            "Classify the user query into exactly one of these labels:\n"
            "- search_query\n"
            "- general_question\n"
            "- unclear\n\n"
            "Rules:\n"
            "- search_query: the user asks to show, search, find, list, filter, or retrieve module data\n"
            "- general_question: greetings, identity questions, casual chat, or non-search questions\n"
            "- unclear: the request is too vague or missing important details\n\n"
            "Return only one label.\n"
            "Do not explain.\n"
            "Do not add extra text."
        ),
        (
            "human",
            "{user_query}"
        )
    ])


# =====================================================
# CLARIFICATION / SEARCH-ONLY PROMPT
# =====================================================
def get_greeting_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a search-only assistant for module-based search.\n"
            "You must not answer general knowledge questions.\n"
            "If the user greeting or question is not a module search request, reply with:\n"
            "I can help only with module search questions. Please ask a filter or search question.\n\n"
            "Keep the reply short and clear."
        ),
        (
            "human",
            "{query}"
        )
    ])