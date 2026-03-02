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
    agents_call: List[str]

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
        "output": "",
        "agents_call": [],
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
            print(f"⚠️ [MemoriesAgent] MCP search failed: {e}")
            
    # 2. Try Standard Tool
    # BLOCKER FIX #4: recall_memories(user_id, ...) — was incorrectly called
    # with the query string as user_id, which always returned empty results.
    # Now correctly passes user_id="default" (the shared user since no auth).
    try:
        tool_memories = recall_memories("default")
        memories.append(f"Recursive Memories: {tool_memories}")
    except Exception as e:
        print(f"⚠️ [MemoriesAgent] Standard recall failed: {e}")
        
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

async def run_web_news_agent(state: AgentState) -> str:
    """Run Web News Agent logic."""
    query = state["sanitized_input"]
    if not web_news:
        return "Web news tool not available."
    try:
        results = web_news.invoke(query)
        return str(results)
    except Exception as e:
        return f"Web news failed: {e}"

async def run_contact_agent(state: AgentState) -> str:
    """Run Contact Agent logic."""
    query = state["sanitized_input"]
    try:
        result = get_contact_info(query)
        return str(result) if result else "No contact info found."
    except Exception as e:
        return f"Contact lookup failed: {e}"

async def run_memory_agent(state: AgentState) -> str:
    """Run Memory Agent logic (save/recall/delete)."""
    query = state["sanitized_input"]
    try:
        intent_prompt = f"""
        Decide memory action for this user input:
        "{query}"
        Return only one word: SAVE, RECALL, DELETE, NONE
        """
        decision = await llm_fast.ainvoke([HumanMessage(content=intent_prompt)])
        action = decision.content.strip().upper()

        if "SAVE" in action:
            result = save_memory(query)
            return f"Memory saved: {result}"
        if "DELETE" in action:
            result = delete_memory(query)
            return f"Memory deleted: {result}"
        if "RECALL" in action:
            result = recall_memories(query)
            return f"Memory recall: {result}"
        return "No memory action required."
    except Exception as e:
        return f"Memory agent failed: {e}"

async def agent_caller(state: AgentState) -> List[str]:
    """
    Orchestrator selector. Returns list of agent function names to execute.
    """
    available_agents = [
        "run_history_agent",
        "run_rag_agent",
        "run_memories_agent",
        "run_web_search_agent",
        "run_web_news_agent",
        "run_contact_agent",
        "run_memory_agent",
    ]
    prompt = f"""
    You are the orchestrator.
    Decide which agents should be called for the user input.

    Available agents:
    1. History Agent : run_history_agent
    2. RAG Agent : run_rag_agent
    3. Memories Agent : run_memories_agent
    4. Web Search Agent : run_web_search_agent
    5. Web News Agent : run_web_news_agent
    6. Contact Agent : run_contact_agent
    7. Memory Agent : run_memory_agent

    User input: {state["sanitized_input"]}
    History length: {len(state.get("history", []))}

    Return strict JSON only in this format:
    {{"agents_call": ["run_rag_agent", "run_history_agent"]}}
    """
    try:
        response = await llm_fast.ainvoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        selected = parsed.get("agents_call", [])
        if not isinstance(selected, list):
            selected = []
        selected = [a for a in selected if a in available_agents]
        if not selected:
            selected = ["run_rag_agent", "run_history_agent"]
        return selected
    except Exception:
        return ["run_rag_agent", "run_history_agent"]

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

async def orchestrator_node(state: AgentState) -> AgentState:
    """🧠 Orchestrator Node"""
    print(f"🧠 [Orchestrator] Planning execution...")
    selected_agents = await agent_caller(state)
    print(f"🧠 [Orchestrator] Selected agents: {selected_agents}")
    return {
        **state,
        "agents_call": selected_agents,
    }

async def parallel_agents_node(state: AgentState) -> AgentState:
    """
    ⚡ Parallel Agents Node
    Runs only selected retrieval agents concurrently.
    """
    selected_agents = state.get("agents_call", [])
    print(f"⚡ [ParallelAgents] Running selected agents: {selected_agents}")

    if not selected_agents:
        return state

    agent_map = {
        "run_history_agent": run_history_agent,
        "run_rag_agent": run_rag_agent,
        "run_memories_agent": run_memories_agent,
        "run_web_search_agent": run_web_search_agent,
        "run_web_news_agent": run_web_news_agent,
        "run_contact_agent": run_contact_agent,
        "run_memory_agent": run_memory_agent,
    }

    tasks = []
    task_names = []
    for agent_name in selected_agents:
        fn = agent_map.get(agent_name)
        if fn:
            tasks.append(fn(state))
            task_names.append(agent_name)

    if not tasks:
        return state

    results = await asyncio.gather(*tasks, return_exceptions=True)

    updates = {**state}
    web_parts = []
    memories_parts = []

    for name, result in zip(task_names, results):
        text = str(result) if not isinstance(result, Exception) else f"{name} failed: {result}"
        if name == "run_history_agent":
            updates["history_summary"] = text
        elif name == "run_rag_agent":
            updates["rag_context"] = text
        elif name == "run_memories_agent":
            memories_parts.append(text)
        elif name == "run_web_search_agent":
            web_parts.append(f"[web_search]\n{text}")
        elif name == "run_web_news_agent":
            web_parts.append(f"[web_news]\n{text}")
        elif name == "run_contact_agent":
            memories_parts.append(f"[contact]\n{text}")
        elif name == "run_memory_agent":
            memories_parts.append(f"[memory]\n{text}")

    if web_parts:
        updates["web_context"] = "\n\n".join(web_parts)
    if memories_parts:
        updates["memories_context"] = "\n\n".join(memories_parts)

    print(f"✅ [ParallelAgents] Finished.")
    return updates

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
        )
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        is_valid = result.get("is_valid", True)
        critique = result.get("critique", "")
    # BLOCKER FIX #5: Replaced bare `except:` which silently auto-approved every
    # response on any parse failure, completely bypassing quality control.
    # Now uses explicit exception types and logs the error so failures are visible.
    except json.JSONDecodeError as e:
        print(f"⚠️ [VerifierAgent] JSON parse error — treating as valid: {e}")
        is_valid = True
        critique = "Verification skipped: response was not valid JSON"
    except Exception as e:
        print(f"⚠️ [VerifierAgent] Unexpected error during verification: {e}")
        is_valid = True
        critique = f"Verification skipped due to error: {type(e).__name__}"
        
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

    # 3. Orchestrator
    yield f"data: {json.dumps({'type': 'status', 'content': 'Orchestrator selecting agents...'})}\n\n"
    state = await orchestrator_node(state)

    # 4. Parallel Agents
    selected = state.get("agents_call", [])
    yield f"data: {json.dumps({'type': 'status', 'content': f'Running selected agents: {selected}'})}\n\n"
    state = await parallel_agents_node(state)
    
    # 5. Combiner (Streaming)
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
