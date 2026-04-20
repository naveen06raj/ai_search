# main.py

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

# 🔥 Move import here (IMPORTANT for performance)
from Orchestration.orchestration_agent import graph as orchestration_graph


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
    allow_origins=["*"],   # ⚠️ change in production
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

    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class SearchResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


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
# 🔥 MAIN SEARCH API
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
            "login_id": request.login_id
        }

        # 🔥 Timeout protection
        result = await asyncio.wait_for(
            orchestration_graph.ainvoke(input_data),
            timeout=15
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # 🔥 Safe response
        response_data = result.get("response") or {}

        metadata = {
            "query": request.query,
            "execution_time_seconds": round(execution_time, 2),
            "route": result.get("route"),
            "action": result.get("defect_action") or result.get("feedback_action"),
            "record_count": response_data.get("total", len(response_data.get("records", []))),
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
# EXAMPLES
# =====================================================

@app.get("/api/example-queries")
async def get_example_queries():
    return {
        "success": True,
        "examples": [
            "Show open defects for block 6",
            "Find defects with ticket number 25121137801",
            "Show all feedback for block 6",        # ✅ Added feedback example
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

port = int(os.environ.get("PORT", 8080))

uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=port,
)