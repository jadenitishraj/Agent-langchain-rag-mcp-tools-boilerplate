"""
Multi-Agent Orchestrator System (Agent v2)
==========================================
This module implements a sophisticated multi-agent architecture to replace the single-node agent.
It features:
1.  **Input Guardrails**: Safety first.
2.  **Orchestrator**: The "Mastermind" that plans the execution.
3.  **Parallel Agents**:
    -   `HistorySummaryAgent`: Context from conversation.
    -   `RAGAgent`: Context from codebase (RAG v2).
    -   `MemoriesAgent`: Context from user memories (SQLite/MCP).
    -   `WebSearchAgent`: Context from the internet.
4.  **CombinerAgent**: Synthesizes all data into a coherent response.
5.  **VerifierAgent**: Quality assurance loop.
6.  **Output Guardrails**: Final safety check.

"The best way to predict the future is to create it." - Peter Drucker
"""

import os
import sys
import json
import asyncio
from typing import TypedDict, List, Optional, Any
import operator

# Try importing from venv if module not found, or assume environment is set up
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
except ImportError:
    # Adding typical venv paths just in case, though usually python path should handle it
    sys.path.append(os.path.join(os.getcwd(), "venv", "lib", "python3.9", "site-packages"))
    sys.path.append(os.path.join(os.getcwd(), "venv", "lib", "python3.10", "site-packages"))
    sys.path.append(os.path.join(os.getcwd(), "venv", "lib", "python3.11", "site-packages"))
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG
try:
    from rag_v2.query import get_context
    RAG_VERSION = "v2"
except ImportError:
    try:
        from rag.query import get_context
        RAG_VERSION = "v1"
    except ImportError:
        get_context = lambda q, k: "RAG not available"
        RAG_VERSION = "none"

# Import Guardrails
try:
    from guardrails import validate_input, validate_output, GuardrailConfig
except ImportError:
    # Mock if missing
    print("⚠️ Guardrails module missing, using mock.")
    class GuardrailConfig:
        def __init__(self, **kwargs): pass
    def validate_input(i, c): return type('obj', (object,), {'is_valid': True, 'sanitized_input': i, 'blocked_reason': ''})
    def validate_output(o, q, c, conf): return type('obj', (object,), {'modified': False, 'sanitized_output': o})
    guardrail_config = GuardrailConfig()

# Import Tools
from tools import ALL_TOOLS, get_all_memories, web_search, web_news
from tools.contact_tool import get_contact_info
from tools.memory_tool import save_memory, recall_memories, delete_memory

# Import SQLite MCP tools
try:
    from mcp_servers.sqlite_client import mcp_query_memories, mcp_search_memories
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp_query_memories = None
    mcp_search_memories = None

# =================================================================================================
# 1. State Definition
# =================================================================================================

class AgentState(TypedDict):
    """
    Global state for the multi-agent system.
    Tracks the flow of information from input to final output.
    """
    # Inputs
    input: str
    history: List[dict]
    
    # Flags & Status
    blocked: bool
    block_reason: str
    retry_count: int
    
    # Intermediate Artifacts (Parallel Outputs)
    sanitized_input: str
    history_summary: str      # From HistoryAgent
    rag_context: str          # From RAGAgent
    memories_context: str     # From MemoriesAgent
    web_context: str          # From WebSearchAgent
    
    # Synthesis
    draft_response: str       # From CombinerAgent
    critique: str             # From VerifierAgent
    is_valid: bool            # From VerifierAgent
    
    # Final Output
    output: str

def init_state(input_text: str, history: List[dict]) -> AgentState:
    """Initialize the default state."""
    return {
        "input": input_text,
        "history": history,
        "blocked": False,
        "block_reason": "",
        "retry_count": 0,
        "sanitized_input": input_text,
        "history_summary": "",
        "rag_context": "",
        "memories_context": "",
        "web_context": "",
        "draft_response": "",
        "critique": "",
        "is_valid": False,
        "output": ""
    }

# =================================================================================================
# 2. Configuration & Models
# =================================================================================================

# Guardrails config
# Re-init if needed
if 'guardrail_config' not in globals():
    guardrail_config = GuardrailConfig(
        enable_injection_detection=True,
        enable_toxicity_check=True,
        enable_topic_check=True,
        enable_pii_detection=True,
        enable_hallucination_check=True,
        max_input_length=1000,
        max_output_length=3000,
    )

# LLMs
llm_smart = ChatOpenAI(model="gpt-4o", temperature=0.2) 
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

print(f"🚀 Multi-Agent System V2 Initialized")

# =================================================================================================
# 3. Agent Functions (Not Nodes, but Functions called by Parallel Node)
# =================================================================================================

async def run_history_agent(state: AgentState) -> str:
    """Run History Agent logic."""
    history = state.get("history", [])
    if not history:
        return "No previous history."
    
    try:
        prompt = f"Summarize conversation history (last 5 messages): {json.dumps(history[-5:])}"
        response = await llm_fast.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error summarizing history: {e}"

async def run_rag_agent(state: AgentState) -> str:
    """Run RAG Agent logic."""
    query = state["sanitized_input"]
    try:
        # get_context is likely sync, so run in executor to not block async loop if heavy
        # Assuming get_context is fast enough or we just run it sync
        context = get_context(query, k=3)
        return context
    except Exception as e:
        return f"Error retrieving RAG context: {e}"

async def run_memories_agent(state: AgentState) -> str:
    """Run Memories Agent logic."""
    query = state["sanitized_input"]
    memories = []
    
    # 1. Try MCP Search
    if MCP_AVAILABLE and mcp_search_memories:
        try:
            # Check if mcp_search_memories is async or sync. Assuming sync.
            mcp_results = mcp_search_memories(query)
            memories.append(f"MCP Memories: {mcp_results}")
        except Exception as e:
            pass
            
    # 2. Try Standard Tool
    try:
        tool_memories = recall_memories(query)
        memories.append(f"Recursive Memories: {tool_memories}")
    except Exception as e:
        pass
        
    return "\n".join(memories) if memories else "No relevant memories found."

async def run_web_search_agent(state: AgentState) -> str:
    """Run Web Search Agent logic."""
    query = state["sanitized_input"]
    
    if not web_search:
        return "Web search tool not available."

    try:
        check_prompt = f"Does '{query}' require external web search (news/docs)? Reply YES or NO."
        decision = await llm_fast.ainvoke([HumanMessage(content=check_prompt)])
        
        if "YES" in decision.content.upper():
            # web_search might be sync, wrap if needed
            # Assuming it takes string input
            results = web_search.invoke(query) 
            return str(results)
        else:
            return "Web search skipped."
    except Exception as e:
        return f"Web search failed: {e}"

# =================================================================================================
# 4. Graph Nodes
# =================================================================================================

def input_guardrail_node(state: AgentState) -> AgentState:
    """🛡️ Input Guardrail Node"""
    print(f"\n[InputGuardrail] Checking input...")
    result = validate_input(state["input"], guardrail_config)
    
    if not result.is_valid:
        print(f"❌ [InputGuardrail] BLOCKED: {result.blocked_reason}")
        return {
            **state,
            "blocked": True,
            "block_reason": result.blocked_reason,
            "output": f"I cannot process this request. {result.blocked_reason}."
        }
    return {**state, "sanitized_input": result.sanitized_input}

def orchestrator_node(state: AgentState) -> AgentState:
    """🧠 Orchestrator Node"""
    # Just a pass-through planner for now
    print(f"🧠 [Orchestrator] Planning execution...")
    return state

async def parallel_agents_node(state: AgentState) -> AgentState:
    """
    ⚡ Parallel Agents Node
    Runs all retrieval agents concurrently.
    """
    print(f"⚡ [ParallelAgents] Running History, RAG, Memories, Web...")
    
    # Execute all 4 concurrently
    results = await asyncio.gather(
        run_history_agent(state),
        run_rag_agent(state),
        run_memories_agent(state),
        run_web_search_agent(state)
    )
    
    hist_res, rag_res, mem_res, web_res = results
    
    print(f"✅ [ParallelAgents] Finished.")
    print(f"   - History: {len(hist_res)} chars")
    print(f"   - RAG: {len(rag_res)} chars")
    print(f"   - Memories: {len(mem_res)} chars")
    print(f"   - Web: {len(web_res)} chars")
    
    return {
        **state,
        "history_summary": hist_res,
        "rag_context": rag_res,
        "memories_context": mem_res,
        "web_context": web_res
    }

async def combiner_agent_node(state: AgentState) -> AgentState:
    """🏗️ Combiner Agent Node"""
    print(f"🏗️ [CombinerAgent] Synthesizing...")
    
    prompt = f"""
    You are the Lead Developer Agent.
    
    USER QUESTION: {state['sanitized_input']}
    
    CONTEXT REPORTS:
    1. HISTORY: {state['history_summary']}
    2. MEMORIES: {state['memories_context']}
    3. RAG (CODEBASE): {state['rag_context']}
    4. WEB: {state['web_context']}
    
    INSTRUCTIONS:
    Synthesize these reports into an amazing, helpful response.
    """
    
    response = await llm_smart.ainvoke([HumanMessage(content=prompt)])
    return {**state, "draft_response": response.content}

async def verifier_agent_node(state: AgentState) -> AgentState:
    """🕵️ Verifier Agent Node"""
    print(f"🕵️ [VerifierAgent] Verifying...")
    
    prompt = f"""
    Review this response:
    QUESTION: {state['sanitized_input']}
    RESPONSE: {state['draft_response']}
    
    Return JSON: {{"is_valid": true/false, "critique": "..."}}
    """
    
    try:
        response = await llm_fast.ainvoke(
            [HumanMessage(content=prompt)],
            # method of returning json might differ by version, simple parse here
        )
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        is_valid = result.get("is_valid", True)
        critique = result.get("critique", "")
    except:
        is_valid = True
        critique = ""
        
    print(f"🕵️ [VerifierAgent] Valid: {is_valid}")
    return {
        **state,
        "is_valid": is_valid,
        "critique": critique,
        "retry_count": state["retry_count"] + 1
    }

def output_guardrail_node(state: AgentState) -> AgentState:
    """🛡️ Output Guardrail Node"""
    print(f"🛡️ [OutputGuardrail] Finalizing...")
    result = validate_output(
        output=state["draft_response"],
        question=state["sanitized_input"],
        context=state["rag_context"],
        config=guardrail_config
    )
    return {**state, "output": result.sanitized_output}

# =================================================================================================
# 5. Graph Construction
# =================================================================================================

workflow = StateGraph(AgentState)

workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("parallel_agents", parallel_agents_node)
workflow.add_node("combiner_agent", combiner_agent_node)
workflow.add_node("verifier_agent", verifier_agent_node)
workflow.add_node("output_guardrail", output_guardrail_node)

workflow.set_entry_point("input_guardrail")

def check_blocked(state):
    return "blocked" if state["blocked"] else "continue"

workflow.add_conditional_edges("input_guardrail", check_blocked, {"blocked": END, "continue": "orchestrator"})
workflow.add_edge("orchestrator", "parallel_agents")
workflow.add_edge("parallel_agents", "combiner_agent")
workflow.add_edge("combiner_agent", "verifier_agent")

def check_verification(state):
    if not state["is_valid"] and state["retry_count"] < 2:
        return "retry"
    return "approved"

workflow.add_conditional_edges("verifier_agent", check_verification, {"retry": "combiner_agent", "approved": "output_guardrail"})
workflow.add_edge("output_guardrail", END)

app = workflow.compile()

# =================================================================================================
# 6. Public Interface
# =================================================================================================

# Import Cache
try:
    from rag_v2.cache_manager import search_cache, save_to_cache
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False

async def run_agent_v2_async(question: str, history: Optional[List[dict]] = None) -> str:
    """Async run with Caching."""
    
    # 1. Check Cache
    if CACHE_ENABLED: # Check cache even with history to ensure consistency
        cached = search_cache(question)
        if cached:
            return cached

    state = init_state(question, history or [])
    result = await app.ainvoke(state)
    output = result["output"]
    
    # 2. Save to Cache
    if CACHE_ENABLED and output and not state.get("blocked", False):
        save_to_cache(question, output)
        
    return output

def run_agent_v2(question: str, history: Optional[List[dict]] = None) -> str:
    """Sync wrapper."""
    return asyncio.run(run_agent_v2_async(question, history))

async def run_agent_v2_stream(question: str, history: Optional[List[dict]] = None):
    """
    Stream generator (Manual Orchestration) with Caching.
    """
    # 1. Check Cache
    if CACHE_ENABLED:
        cached = search_cache(question)
        if cached:
            yield f"data: {json.dumps({'type': 'status', 'content': '⚡ Semantic Cache Hit!'})}\n\n"
            # Stream the cached response quickly
            chunk_size = 10
            for i in range(0, len(cached), chunk_size):
                chunk = cached[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01) # Simulate fast typing
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    # 1. Init State
    state = init_state(question, history or [])
    
    # 2. Input Guardrail
    yield f"data: {json.dumps({'type': 'status', 'content': 'Validating input...'})}\n\n"
    state = input_guardrail_node(state)
    if state["blocked"]:
        yield f"data: {json.dumps({'type': 'error', 'content': state['block_reason']})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # 3. Parallel Agents
    yield f"data: {json.dumps({'type': 'status', 'content': 'Running parallel agents (History, RAG, Memory, Web)...'})}\n\n"
    
    # We call the async functions directly
    results = await asyncio.gather(
        run_history_agent(state),
        run_rag_agent(state),
        run_memories_agent(state),
        run_web_search_agent(state)
    )
    hist_res, rag_res, mem_res, web_res = results
    
    # Update State
    state.update({
        "history_summary": hist_res,
        "rag_context": rag_res,
        "memories_context": mem_res,
        "web_context": web_res
    })
    
    # 4. Combiner (Streaming)
    yield f"data: {json.dumps({'type': 'status', 'content': 'Synthesizing response...'})}\n\n"
    
    prompt = f"""
    You are the Lead Developer Agent.
    
    USER QUESTION: {state['sanitized_input']}
    
    CONTEXT REPORTS:
    1. HISTORY: {state['history_summary']}
    2. MEMORIES: {state['memories_context']}
    3. RAG (CODEBASE): {state['rag_context']}
    4. WEB: {state['web_context']}
    
    INSTRUCTIONS:
    Synthesize these reports into an amazing, helpful response.
    """
    
    draft_response = ""
    async for chunk in llm_smart.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            draft_response += chunk.content
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
            
    state["draft_response"] = draft_response
    
    # 5. Verifier
    yield f"data: {json.dumps({'type': 'status', 'content': 'Verifying response...'})}\n\n"
    state = await verifier_agent_node(state)
    
    # 6. Retry Logic (One valid retry)
    if not state["is_valid"] and state["retry_count"] < 2:
         msg = json.dumps({'type': 'status', 'content': f"Critique: {state['critique']}. Refining..."})
         yield f"data: {msg}\n\n"
         
         retry_prompt = f"""
         Original Question: {state['sanitized_input']}
         Draft Response: {state['draft_response']}
         Critique: {state['critique']}
         
         Please rewrite the response to address the critique.
         """
         
         # Stream the retry
         draft_response = ""
         yield f"data: {json.dumps({'type': 'replace', 'content': ''})}\n\n" # Clear previous
         
         async for chunk in llm_smart.astream([HumanMessage(content=retry_prompt)]):
            if chunk.content:
                draft_response += chunk.content
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                
         state["draft_response"] = draft_response
    
    # 7. Output Guardrail
    yield f"data: {json.dumps({'type': 'status', 'content': 'Final safety check...'})}\n\n"
    state = output_guardrail_node(state)
    
    # Save to Cache (if successful)
    if CACHE_ENABLED and state["output"] and not state.get("blocked", False):
        save_to_cache(question, state["output"])
    
    if state["output"] != state["draft_response"]:
        # it was modified
        yield f"data: {json.dumps({'type': 'replace', 'content': state['output']})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

if __name__ == "__main__":
    # Test
    print(run_agent_v2("Hello world"))
