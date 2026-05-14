"""
Defect MCP Server (Production Ready)
- No hardcoded token
- Multi-user support
- Cloud Run compatible
"""

import os
from typing import Optional, List, Dict, Any

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
# INPUT MODEL
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

    # IMPORTANT (dynamic user data)
    login_id: int
    token: str


# =====================================================
# OUTPUT MODELS
# =====================================================
class DefectSearchResponse(BaseModel):
    total: int
    records: List[Dict[str, Any]]
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
# MCP TOOL
# =====================================================
@fastmcp.tool()
async def search_defects(input: DefectSearchInput) -> DefectSearchResponse:
    import httpx

    token = input.token
    login_id = input.login_id

    if not token:
        raise ValueError("User token is required")

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

    records: List[Dict[str, Any]] = []
    status_summary: Dict[str, int] = {}

    for item in api_data.get("data", []):
        # Preserve the full nested record if present
        if isinstance(item, dict) and "lists" in item:
            lists = item.get("lists", {}) or {}
            inspection = item.get("inspection")

            status_code = lists.get("status")
            status_text = STATUS_MAP.get(status_code, "UNKNOWN")
            status_summary[status_text] = status_summary.get(status_text, 0) + 1

            # Keep the full object exactly as returned by API
            records.append({
                "lists": lists,
                "user_info": item.get("user_info"),
                "unit_info": item.get("unit_info"),
                "inspection": inspection,
            })

        else:
            # Fallback for flat rows, keep whatever exists
            lists = item if isinstance(item, dict) else {}
            status_code = lists.get("status")
            status_text = STATUS_MAP.get(status_code, "UNKNOWN")
            status_summary[status_text] = status_summary.get(status_text, 0) + 1

            records.append({
                "lists": lists,
                "user_info": None,
                "unit_info": None,
                "inspection": None,
            })

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
            data=payload,
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
            data=payload,
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


@fastmcp.tool()
async def defect_magic_map(
    extracted: List[Dict],
    token: str
):
    import httpx

    print(f"\n🤖 [AI MAGIC MAP] Extracted: {extracted}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    # STEP 1: GET LOCATIONS
    location_url = "https://aerea.panzerplayground.com/api/v8/getDefectslocation"

    async with httpx.AsyncClient() as client:
        loc_res = await client.post(
            location_url,
            data={"property": 1},
            headers=headers
        )

    if loc_res.status_code != 200:
        raise RuntimeError(f"Location API Error: {loc_res.text}")

    location_data = loc_res.json().get("data", [])

    location_map = {
        (item.get("defect_location") or "").lower(): {
            "id": item.get("id"),
            "name": item.get("defect_location")
        }
        for item in location_data
    }

    # HELPER MATCH
    def match(text, mapping):
        if not text:
            return None, None

        text = text.lower()

        for k, v in mapping.items():
            if k in text or text in k:
                return v["name"], v["id"]

        return None, None

    # PROCESS
    type_cache = {}
    results = []

    for item in extracted:
        loc_text = item.get("location")
        type_text = item.get("type")

        # LOCATION
        loc_name, loc_id = match(loc_text, location_map)

        if not loc_id:
            print(f"⚠️ Location not found: {loc_text}")
            continue

        # GET TYPES
        if loc_id not in type_cache:
            type_url = "https://aerea.panzerplayground.com/api/v8/getDefectstype"

            async with httpx.AsyncClient() as client:
                type_res = await client.post(
                    type_url,
                    data={
                        "property": 1,
                        "location": loc_id
                    },
                    headers=headers
                )

            if type_res.status_code != 200:
                raise RuntimeError(f"Type API Error: {type_res.text}")

            type_data = type_res.json().get("data", [])

            type_cache[loc_id] = {
                (t.get("defect_type") or "").lower(): {
                    "id": t.get("id"),
                    "name": t.get("defect_type")
                }
                for t in type_data
            }

        type_map = type_cache[loc_id]

        # TYPE MATCH
        selected_type = None
        selected_type_id = None

        user_type = (type_text or "").lower()

        for k, v in type_map.items():
            if k in user_type or user_type in k:
                selected_type = v["name"]
                selected_type_id = v["id"]
                break

        # fallback
        if not selected_type and type_map:
            first = next(iter(type_map.values()))
            selected_type = first["name"]
            selected_type_id = first["id"]

        # REMARK
        remark = item.get("remarks") or f"Issue related to {selected_type}"

        if not remark.endswith("."):
            remark += "."

        results.append({
            "location": loc_name,
            "location_id": loc_id,
            "type": selected_type,
            "type_id": selected_type_id,
            "remarks": remark
        })

    print(f"🧹 FINAL MAPPED: {results}")

    return results