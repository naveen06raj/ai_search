# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import traceback

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Defect AI API",
    description="Backend API for Defect Search and Management",
    version="1.0.0"
)

# 🔧 UPDATE THIS SECTION: Configure CORS for local development
origins = [
    "http://localhost:8501",  # Your Streamlit frontend's default port
    "http://127.0.0.1:8501",  # Also allow localhost IP
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Changed from "*" to the list above
    allow_credentials=True,
    allow_methods=["*"],    # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],    # Allows all headers
    expose_headers=["*"]    # Makes custom headers available to the frontend
)

# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================
class SearchRequest(BaseModel):
    query: str
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
# API ENDPOINTS
# =====================================================
@app.get("/")
async def root():
    return {
        "service": "Defect AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/api/search": "POST - Search defects",
            "/api/health": "GET - Health check",
            "/api/example-queries": "GET - Example queries"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Import here to avoid circular imports
        from Orchestration.orchestration_agent import graph as orchestration_graph
        
        test_result = await orchestration_graph.ainvoke({
            "user_query": "test",
            "chat_history": []
        })
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "orchestration": "working" if test_result else "not_working"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/search", response_model=SearchResponse)
async def search_defects(request: SearchRequest):
    """
    Main search endpoint
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔍 [API] Received search request: {request.query}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Import here to avoid circular imports
        from Orchestration.orchestration_agent import graph as orchestration_graph
        
        # Prepare input
        input_data = {
            "user_query": request.query,
            "chat_history": []
        }
        
        # Execute search
        result = await orchestration_graph.ainvoke(input_data)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Extract response data
        response_data = result.get("response", {})
        
        # Prepare metadata
        metadata = {
            "query": request.query,
            "execution_time_seconds": round(execution_time, 2),
            "route": result.get("route", "unknown"),
            "defect_action": result.get("defect_action", "unknown"),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "record_count": response_data.get("total", 0),
            "filters_applied": response_data.get("filter_applied", False)
        }
        
        print(f"✅ [API] Found {response_data.get('total', 0)} records in {execution_time:.2f}s")
        
        return SearchResponse(
            success=True,
            message=f"Found {response_data.get('total', 0)} defect records",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"\n❌ [API] Error: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/example-queries")
async def get_example_queries():
    examples = [
        "Show open defects for block 6",
        "Find defects with ticket number 25121137801",
        "Show all defects submitted in 2024"
    ]
    
    return {
        "success": True,
        "examples": examples,
        "count": len(examples)
    }

# =====================================================
# RUN SERVER (FIXED)
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING DEFECT AI API SERVER")
    print("="*60)
    print("\n📡 API ENDPOINTS:")
    print("  • Main URL: http://127.0.0.1:8000")
    print("  • API Docs: http://127.0.0.1:8000/docs")
    print("  • Health Check: http://127.0.0.1:8000/api/health")
    print("  • Search: POST http://127.0.0.1:8000/api/search")
    print("="*60)
    print("\n✅ Server starting...\n")
    
    # Run with proper configuration
    uvicorn.run(
        "main:app",  # Changed to string import
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )