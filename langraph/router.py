from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from .schemas import AgentRequest, AgentResponse
from .agent_v2 import run_agent_v2 as run_agent, run_agent_v2_stream as run_agent_stream
import os

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)

# ============================================================================
# ROUTE DECORATOR
# - @router.post: Creates a POST endpoint (used for sending data to server)
# - "/ask": The URL path. Frontend calls: POST http://localhost:8000/ask
# - response_model=AgentResponse: FastAPI will validate that returned data
#   matches the AgentResponse Pydantic model. Also auto-generates API docs.
# ============================================================================
@router.post("/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest):
    # ========================================================================
    # FUNCTION DEFINITION
    # - request: AgentRequest → FastAPI automatically parses the incoming
    #   JSON body and converts it into an AgentRequest Pydantic model.
    # - If JSON is invalid or missing required fields, FastAPI returns 422 error.
    # ========================================================================
    """Non-streaming endpoint with history support"""
    
    # ========================================================================
    # TRY BLOCK: Wrap everything in try-except to catch any errors
    # and return proper HTTP error responses instead of crashing.
    # ========================================================================
    try:
        # ====================================================================
        # API KEY VALIDATION
        # - os.environ.get("OPENAI_API_KEY"): Reads the API key from .env
        # - If not set or empty, we can't call OpenAI, so fail early
        # - HTTPException: FastAPI's way to return HTTP error to frontend
        # - status_code=500: Internal Server Error
        # - detail: The error message sent to the frontend
        # ====================================================================
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="OPENAI_API_KEY environment variable not set."
            )
        
        # ====================================================================
        # HISTORY CONVERSION
        # - Frontend sends history as Pydantic models (Message objects)
        # - run_agent() expects plain Python dicts, not Pydantic models
        # - This list comprehension converts each Message to a dict:
        #   Message(role="user", content="Hi") → {"role": "user", "content": "Hi"}
        # ====================================================================
        history = None
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]
        
        # ====================================================================
        # THE MAIN AGENT CALL - WHERE ALL THE MAGIC HAPPENS!
        # - run_agent() is imported from agent.py
        # - It runs the full LangGraph workflow:
        #   1. Input Guardrails (safety checks)
        #   2. RAG Retrieval (search codebase knowledge)
        #   3. LLM Call with Tools (GPT-4o-mini)
        #   4. Tool Execution (if needed)
        #   5. Output Guardrails (PII redaction, etc.)
        # - Returns: A string containing the AI's response
        # ====================================================================
        answer = run_agent(request.question, history=history)
        
        # ====================================================================
        # RETURN RESPONSE
        # - Wrap the answer string in AgentResponse Pydantic model
        # - FastAPI automatically converts this to JSON: {"answer": "..."}
        # - Frontend receives this JSON response
        # ====================================================================
        return AgentResponse(answer=answer)
    
    # ========================================================================
    # ERROR HANDLING
    # - Catches ANY exception that occurs in the try block
    # - Converts Python exception to HTTP 500 error
    # - str(e): Converts exception to string for the error message
    # - Frontend receives: {"detail": "Error message here"}
    # ========================================================================
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/stream")
async def ask_agent_stream(request: AgentRequest):
    """Streaming endpoint with history support"""
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="OPENAI_API_KEY environment variable not set."
            )
        
        # Convert history to list of dicts
        history = None
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]
        
        return StreamingResponse(
            run_agent_stream(request.question, history=history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
