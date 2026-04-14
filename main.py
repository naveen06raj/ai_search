# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from datetime import datetime
import traceback

from MCP.defect_mcp_server import fastmcp

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Defect AI API",
    description="Backend API for Defect Search and Management",
    version="2.0.0"
)

# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REQUEST / RESPONSE MODELS (UPDATED)
# =====================================================

class SearchRequest(BaseModel):
    query: str
    token: str              # 🔥 REQUIRED
    login_id: int           # 🔥 REQUIRED

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
        "service": "Defect AI API",
        "version": "2.0.0",
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
# 🔥 MAIN SEARCH API (FIXED)
# =====================================================

@app.post("/api/search", response_model=SearchResponse)
async def search_defects(request: SearchRequest):
    try:
        print(f"\n{'='*60}")
        print(f"🔍 QUERY: {request.query}")
        print(f"{'='*60}")

        start_time = datetime.now()

        # 🔥 Import here
        from Orchestration.orchestration_agent import graph as orchestration_graph

        # 🔥 IMPORTANT FIX: pass token + login_id
        input_data = {
            "user_query": request.query,
            "chat_history": [],
            "token": request.token,
            "login_id": request.login_id
        }

        # 🔥 Execute AI
        result = await orchestration_graph.ainvoke(input_data)

        execution_time = (datetime.now() - start_time).total_seconds()

        response_data = result.get("response", {})

        metadata = {
            "query": request.query,
            "execution_time_seconds": round(execution_time, 2),
            "route": result.get("route"),
            "action": result.get("defect_action"),
            "record_count": response_data.get("total", 0),
            "user_id": request.user_id,
            "session_id": request.session_id
        }

        print(f"✅ Found {response_data.get('total', 0)} records in {execution_time:.2f}s")

        return SearchResponse(
            success=True,
            message="Search completed",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())

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
            "Show all defects submitted in 2024"
        ]
    }


# =====================================================
# MCP MOUNT
# =====================================================

app.mount("/mcp", fastmcp.streamable_http_app)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )