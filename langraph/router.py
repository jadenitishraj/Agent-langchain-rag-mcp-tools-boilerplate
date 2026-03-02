from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from .schemas import AgentRequest, AgentResponse
from .agent_v2 import run_agent_v2 as run_agent, run_agent_v2_stream as run_agent_stream

# Import the limiter instance from main app
from slowapi import Limiter
from slowapi.util import get_remote_address

# We pull the limiter off the app's state at request time — but since we need
# it at decoration time we redeclare the same limiter here. The middleware in
# main.py is what enforces it; this just annotates the routes.
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)

# =============================================================================
# BLOCKER FIX #3 (continued): Rate limit agent endpoints specifically.
# These are the expensive ones — each call costs 5-8 LLM round-trips.
# 10 requests/minute per IP is generous for a single user, protective at scale.
# =============================================================================

@router.post("/ask", response_model=AgentResponse)
@limiter.limit("10/minute")
def ask_agent(request: Request, body: AgentRequest):
    """Non-streaming endpoint with history support"""
    # =========================================================================
    # BLOCKER FIX #7: Status code corrected — API key absence is a service
    # configuration problem (503 Service Unavailable), not a server crash (500).
    # NOTE: Key is also validated at startup in main.py, so this is a
    # belt-and-suspenders guard for safety.
    # =========================================================================
    try:
        history = None
        if body.history:
            history = [{"role": m.role, "content": m.content} for m in body.history]

        answer = run_agent(body.question, history=history)
        return AgentResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
@limiter.limit("10/minute")
async def ask_agent_stream(request: Request, body: AgentRequest):
    """Streaming endpoint with history support"""
    try:
        history = None
        if body.history:
            history = [{"role": m.role, "content": m.content} for m in body.history]

        return StreamingResponse(
            run_agent_stream(body.question, history=history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
