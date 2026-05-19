from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from datetime import datetime
import traceback
import logging
import asyncio

# MCP
from MCP.defect_mcp_server import fastmcp

# Move import here (IMPORTANT for performance)
from Orchestration.orchestration_agent import graph as orchestration_graph

from Domain.Defect_Module.Agents.defect_magic_autofill_agent import graph as defect_magic_graph
from MCP.defect_mcp_server import defect_magic_map


# =====================================================
# LOGGING (Production Ready)
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="AI Search API",
    description="Backend API for Defect & Feedback AI Search",
    version="2.1.0"
)


# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# REQUEST / RESPONSE MODELS
# =====================================================

class SearchRequest(BaseModel):
    query: str
    token: str
    login_id: int

    # Module context from frontend: defect / feedback / facility / announcement
    current_module: Optional[str] = None

    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AutofillRequest(BaseModel):
    text: str
    token: str


class SearchResponse(BaseModel):
    success: bool
    message: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: str

    # Optional fields for structured search responses
    chart: Optional[Dict[str, Any]] = None
    total: Optional[int] = None
    filter_applied: Optional[bool] = None
    original_count: Optional[int] = None


# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():
    return {
        "service": "AI Search API",
        "version": "2.1.0",
        "status": "running"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# =====================================================
# MAIN SEARCH API
# =====================================================

@app.post("/api/search", response_model=SearchResponse)
async def search_ai(request: SearchRequest):
    try:
        logger.info(f"🔍 QUERY: {request.query}")

        start_time = datetime.now()

        input_data = {
            "user_query": request.query,
            "chat_history": [],
            "token": request.token,
            "login_id": request.login_id,
            "current_module": request.current_module,
            "session_id": request.session_id,
        }

        # Timeout protection
        result = await asyncio.wait_for(
            orchestration_graph.ainvoke(input_data),
            timeout=30
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        route = result.get("route")
        response_data = result.get("response") or {}

        # =====================================================
        # DEFECT / FEEDBACK / ANNOUNCEMENT / FACILITY DOMAIN
        # return array directly under "data"
        # =====================================================
        if route in {"defect_domain", "feedback_domain", "announcement_domain", "facility_booking_domain"}:
            if isinstance(response_data, dict):
                final_data = response_data.get(
                    "data",
                    response_data.get("table", {}).get("rows", [])
                )
                chart_data = response_data.get("chart")
                total = response_data.get("total")
                filter_applied = response_data.get("filter_applied")
                original_count = response_data.get("original_count")
            else:
                final_data = response_data
                chart_data = None
                total = len(response_data) if isinstance(response_data, list) else 0
                filter_applied = False
                original_count = None

            record_count = len(final_data) if isinstance(final_data, list) else 0

            metadata = {
                "query": request.query,
                "execution_time_seconds": round(execution_time, 2),
                "route": route,
                "action": (
                    result.get("defect_action")
                    or result.get("feedback_action")
                    or result.get("facility_action")
                    or result.get("announcement_action")
                ),
                "record_count": record_count,
                "user_id": request.user_id,
                "session_id": request.session_id
            }

            logger.info(f"✅ Found {metadata['record_count']} records in {execution_time:.2f}s")

            return SearchResponse(
                success=True,
                message="Search completed",
                data=final_data,
                chart=chart_data,
                total=total,
                filter_applied=filter_applied,
                original_count=original_count,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )

        # =====================================================
        # OTHER MODULES
        # =====================================================
        record_count = (
            response_data.get("total", len(response_data.get("records", [])))
            if isinstance(response_data, dict)
            else 0
        )

        metadata = {
            "query": request.query,
            "execution_time_seconds": round(execution_time, 2),
            "route": route,
            "action": (
                result.get("defect_action")
                or result.get("feedback_action")
                or result.get("facility_action")
                or result.get("announcement_action")
            ),
            "record_count": record_count,
            "user_id": request.user_id,
            "session_id": request.session_id
        }

        logger.info(f"✅ Found {metadata['record_count']} records in {execution_time:.2f}s")

        return SearchResponse(
            success=True,
            message="Search completed",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    except asyncio.TimeoutError:
        logger.error("⏱️ Request Timeout")
        raise HTTPException(
            status_code=504,
            detail="Request timeout. Please try again."
        )

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# AI MAGIC AUTOFILL API
# =====================================================

@app.post("/api/defect/autofill")
async def defect_autofill(request: AutofillRequest):
    try:
        logger.info(f"✨ [AUTOFILL] Input: {request.text}")

        # STEP 1 → LLM
        result = await asyncio.wait_for(
            defect_magic_graph.ainvoke({
                "user_input": request.text
            }),
            timeout=10
        )

        extracted = result.get("extracted", [])

        # STEP 2 → MCP (WITH TOKEN)
        mapped = await defect_magic_map(
            extracted=extracted,
            token=request.token
        )

        return {
            "success": True,
            "message": "Autofill generated",
            "data": mapped,
            "timestamp": datetime.now().isoformat()
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Autofill timeout")

    except Exception as e:
        logger.error(f"❌ Autofill Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# EXAMPLES
# =====================================================

@app.get("/api/example-queries")
async def get_example_queries():
    return {
        "success": True,
        "examples": [
            "Show open defects for block 6",
            "Find defects with ticket number 25121137801",
            "Show all feedback for block 6",
            "Show complaints related to lift"
        ]
    }


# =====================================================
# MCP MOUNT
# =====================================================

app.mount("/mcp", fastmcp.streamable_http_app)


# =====================================================
# RUN
# =====================================================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )