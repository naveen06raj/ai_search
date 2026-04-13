"""
Defect MCP Server
- FastMCP
- HTTP (streamable) transport
- Safe schema
- Agent compatible
"""

import os
import requests
import uvicorn
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP


# =====================================================
# MCP SERVER
# =====================================================
fastmcp = FastMCP("defect-mcp")


# =====================================================
# CONFIG
# =====================================================
API_URL = "https://aerea.panzerplayground.com/api/ops/v4/defectssearch"

BEARER_TOKEN = os.getenv(
    "AEREA_BEARER_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6ImE3NGFjN2JiODk1Yjc5ZTI1MTVhMGYxYTE1NzcyMzkxZGUwNWU0YTk5YmE3NjYwNzgxYTgwYmZmZjkyMTQ0MzMzYTRjNWQ1YWEyZjM3OGFhIn0.eyJhdWQiOiIxIiwianRpIjoiYTc0YWM3YmI4OTViNzllMjUxNWEwZjFhMTU3NzIzOTFkZTA1ZTRhOTliYTc2NjA3ODFhODBiZmZmOTIxNDQzMzNhNGM1ZDVhYTJmMzc4YWEiLCJpYXQiOjE3NjU3OTQyMzQsIm5iZiI6MTc2NTc5NDIzNCwiZXhwIjoxNzk3MzMwMjM0LCJzdWIiOiIyMjY4Iiwic2NvcGVzIjpbXX0.DjfP0UAiSvA7tYmgCjlqfxwxF27acEEHCYYLD8tFYfBKoU4EVye4-439pkuu6mfj3i3KbQi4WjAcs44WFEEpPHF_X3kVrqykhP87D8a_PnyFkYjgjINdY2a4dWKY_pdrX-O7B6L118Lm8I6VB_IVdlDthJJEjMGB4qVvgD-_J_y3eEZvjWvWEqobXu_uWPoB58sXrsCOiGFhXilDZrt8Gm66Wdj-jzh0X_4qUGK4oIIBH5_0WNCiAUJYbojBYaNXAab4BqohLA0IU-I7HZYrBss9sasT9kQfc-rZ8AoAvhlgEmNfNjS23oVNyUOgjuZ5L80vvW6gZgkSJvlHv8PzvqN0b_3XjFY09cCRfGFNtEao0hCQeQiH7FXhHWRdg1tLgyB1mauJ0b8DBcpbyYeOzIKti5KQzVv9o5Y-gTp4tFRASJ9DE8Zee5sjwhOoHBW5I17XCVHygEJdGnqr8RPrcHKxESBxgkluZdaqk5aoGFG0TMOAutu99ihx39SgRjHg9jH_doVZiRIC4GAZl5ERljpXJuAOjFbliLJx7SdHX-B9vmnTzIg_sTG-BHpt5rWqfg1aVcQ3LWcUqjc0oDMUhq95hlbEMknY0ZtXZX6f20_IsX_aPffTsCtMenKXpv3uGhXtnHkx9YRCJMMXlcMb84Ct0WnXqlNsQCCqNwjUXeQ"
)

LOGIN_ID = os.getenv("AEREA_LOGIN_ID", "222")


# =====================================================
# STATUS MAPS (✔ FIXED)
# =====================================================
STATUS_MAP = {
    0: "OPEN",
    1: "CLOSED",
    2: "ON SCHEDULE",
    3: "IN PROGRESS",
    4: "COMPLETED - PENDING RESIDENT UPDATE",
}

INSPECTION_STATUS_MAP = {
    0: "New",
    1: "Completed",
    2: "On Schedule",
    3: "In Progress",
}


# =====================================================
# INPUT MODEL
# =====================================================
class DefectSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    ticket: Optional[str] = None
    status: Optional[int] = None
    location: Optional[str] = None
    type: Optional[str] = None
    block_no: Optional[int] = None


# =====================================================
# OUTPUT MODELS
# =====================================================
class DefectRecord(BaseModel):
    ticket_no: str
    status: str
    block: Optional[str]
    unit_no: Optional[str]

    submitted_date: Optional[str]
    rectification_days: Optional[int]

    appointment_datetime: Optional[str]
    appointment_status: Optional[str]

    completion_date: Optional[str]
    reference_id: Optional[str]


class DefectSearchResponse(BaseModel):
    total: int
    records: List[DefectRecord]
    status_summary: Dict[str, int]


# =====================================================
# MCP TOOL
# =====================================================
@fastmcp.tool()
def search_defects(input: DefectSearchInput) -> DefectSearchResponse:
    payload = input.model_dump(exclude_none=True)
    payload["login_id"] = int(LOGIN_ID)

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API ERROR {response.status_code}: {response.text}")

    api_data = response.json()

    records: List[DefectRecord] = []
    status_summary: Dict[str, int] = {}

    for item in api_data.get("data", []):
        lists = item.get("lists", {})
        inspection = item.get("inspection")

        # ---- STATUS ----
        status_code = lists.get("status")
        status_text = STATUS_MAP.get(status_code, "UNKNOWN")
        status_summary[status_text] = status_summary.get(status_text, 0) + 1

        # ---- INSPECTION STATUS ----
        inspection_status_code = (
            inspection.get("status")
            if inspection
            else lists.get("inspection_status")
        )

        appointment_status = INSPECTION_STATUS_MAP.get(
            inspection_status_code,
            "UNKNOWN"
        )

        appointment_datetime = None
        if inspection and inspection.get("appt_date") and inspection.get("appt_time"):
            appointment_datetime = f"{inspection['appt_date']} {inspection['appt_time']}"

        records.append(
            DefectRecord(
                ticket_no=str(lists.get("ticket", "")),
                status=status_text,
                block=str(lists.get("block_no")) if lists.get("block_no") else None,
                unit_no=str(lists.get("unit_no")) if lists.get("unit_no") else None,
                submitted_date=lists.get("created_at"),
                rectification_days=lists.get("rectified_in_days"),
                appointment_datetime=appointment_datetime,
                appointment_status=appointment_status,
                completion_date=lists.get("updated_at"),
                reference_id=lists.get("ref_id"),
            )
        )

    return DefectSearchResponse(
        total=len(records),
        records=records,
        status_summary=status_summary,
    )


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    print("🚀 Starting Defect MCP Server")
    print("🔗 MCP URL: http://127.0.0.1:8002/mcp")
    print("🛠️ Tools: search_defects")

    uvicorn.run(
        fastmcp.streamable_http_app,
        host="127.0.0.1",
        port=8002,
    )
