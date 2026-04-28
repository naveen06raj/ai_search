"""
Defect MCP Server (Production Ready)
- No hardcoded token
- Multi-user support
- Cloud Run compatible
"""

import os
from typing import Optional, List, Dict

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP


# =====================================================
# MCP SERVER
# =====================================================
fastmcp = FastMCP("defect-mcp")


# =====================================================
# CONFIG
# =====================================================
API_URL = "https://aerea.panzerplayground.com/api/ops/v4/defectssearch"
FEEDBACK_API_URL = "https://aerea.panzerplayground.com/api/ops/v4/searchfeedback"
FACILITY_API_URL = "https://aerea.panzerplayground.com/api/ops/v4/searchfacility"
ANNOUNCEMENT_API_URL = "https://aerea.panzerplayground.com/api/ops/v4/searchannouncement"
ROLES_API_URL = "https://aerea.panzerplayground.com/api/ops/v4/roleslist"


# =====================================================
# STATUS MAPS
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
# INPUT MODEL (UPDATED)
# =====================================================
class DefectSearchInput(BaseModel):
    query: Optional[str] = None

    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    ticket: Optional[str] = None
    status: Optional[int] = None
    location: Optional[str] = None
    type: Optional[str] = None
    block_no: Optional[int] = None

    # 🔥 IMPORTANT (dynamic user data)
    login_id: int
    token: str


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

class FeedbackSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    ticket: Optional[str] = None
    status: Optional[int] = None
    category: Optional[int] = None
    building: Optional[str] = None
    filter: Optional[str] = None

    login_id: int
    token: str


class FacilitiesBookingSearchInput(BaseModel):
    fromdate: Optional[str] = None
    todate: Optional[str] = None
    unit: Optional[str] = None
    status: Optional[int] = None
    category: Optional[int] = None   # type_id
    building: Optional[str] = None

    login_id: int
    token: str

class AnnouncementSearchInput(BaseModel):
    startdate: Optional[str] = None
    enddate: Optional[str] = None
    roles: Optional[int] = None
    status: Optional[int] = None

    login_id: int
    token: str

# =====================================================
# MCP TOOL (UPDATED)
# =====================================================
@fastmcp.tool()
async def search_defects(input: DefectSearchInput) -> DefectSearchResponse:
    import httpx

    # 🔥 Extract dynamic user data
    token = input.token
    login_id = input.login_id

    if not token:
        raise ValueError("User token is required")

    # Build payload
    payload = input.model_dump(
        exclude_none=True,
        exclude={"token", "login_id", "query"}
    )

    payload["login_id"] = login_id

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Call backend API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(f"API ERROR {response.status_code}: {response.text}")

    api_data = response.json()

    records: List[DefectRecord] = []
    status_summary: Dict[str, int] = {}

    for item in api_data.get("data", []):
        lists = item.get("lists", {})
        inspection = item.get("inspection")

        # STATUS
        status_code = lists.get("status")
        status_text = STATUS_MAP.get(status_code, "UNKNOWN")
        status_summary[status_text] = status_summary.get(status_text, 0) + 1

        # INSPECTION STATUS
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

@fastmcp.tool()
async def search_feedback(input: FeedbackSearchInput):
    import httpx

    token = input.token

    if not token:
        raise ValueError("User token is required")

    # CLEAN PAYLOAD
    payload = input.model_dump(exclude_none=True)
    payload.pop("token", None)

    print(f"\n📤 [MCP] Feedback Payload: {payload}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            FEEDBACK_API_URL,
            data=payload,
            headers=headers,
            timeout=60,
        )

    print(f"📥 [MCP] Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ [MCP] Error Response: {response.text}")
        raise RuntimeError(f"API ERROR {response.status_code}: {response.text}")

    result = response.json()

    print(f"📦 [MCP] Response Count: {len(result.get('data', []))}")

    return result

@fastmcp.tool()
async def search_facilities_booking(input: FacilitiesBookingSearchInput):
    import httpx

    token = input.token

    if not token:
        raise ValueError("User token is required")

    # 🔥 CLEAN PAYLOAD (same as feedback)
    payload = input.model_dump(exclude_none=True)
    payload.pop("token", None)

    print(f"\n📤 [MCP] Facility Payload: {payload}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            FACILITY_API_URL,
            data=payload,   # ⚠️ IMPORTANT → form-data style
            headers=headers,
            timeout=60,
        )

    print(f"📥 [MCP] Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ [MCP] Error Response: {response.text}")
        raise RuntimeError(f"API ERROR {response.status_code}: {response.text}")

    result = response.json()

    print(f"📦 [MCP] Facility Records: {len(result.get('data', []))}")

    return result

@fastmcp.tool()
async def get_roles_list(token: str, login_id: int):
    import httpx

    payload = {
        "login_id": login_id
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            ROLES_API_URL,
            data=payload,
            headers=headers,
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(f"Roles API ERROR {response.status_code}: {response.text}")

    result = response.json()

    return result.get("roles", {})


@fastmcp.tool()
async def search_announcements(input: AnnouncementSearchInput):
    import httpx

    token = input.token

    if not token:
        raise ValueError("User token is required")

    # CLEAN PAYLOAD
    payload = input.model_dump(exclude_none=True)
    payload.pop("token", None)

    print(f"\n📤 [MCP] Announcement Payload: {payload}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            ANNOUNCEMENT_API_URL,
            data=payload,   # form-data style
            headers=headers,
            timeout=60,
        )

    print(f"📥 [MCP] Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ [MCP] Error Response: {response.text}")
        raise RuntimeError(f"API ERROR {response.status_code}: {response.text}")

    result = response.json()

    print(f"📦 [MCP] Announcement Records: {len(result.get('data', []))}")

    return result