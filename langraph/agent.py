"""
LangGraph Agent with RAG v2 + Guardrails + Streaming + Memory + Tools
"""
import os
import sys
import json
from typing import TypedDict, Generator, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG
try:
    from rag_v2.query import get_context
    RAG_VERSION = "v2"
except ImportError:
    from rag.query import get_context
    RAG_VERSION = "v1"

# Import Guardrails
from guardrails import validate_input, validate_output, GuardrailConfig

# Import Tools
from tools import ALL_TOOLS, get_all_memories, web_search, web_news
from tools.contact_tool import get_contact_info
from tools.memory_tool import save_memory, recall_memories, delete_memory

# Import SQLite MCP tools (Real MCP!)
try:
    from mcp_servers.sqlite_client import mcp_query_memories, mcp_memory_stats, mcp_search_memories
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp_query_memories = None
    mcp_memory_stats = None
    mcp_search_memories = None

print(f"🚀 Agent initialized with RAG {RAG_VERSION} + Guardrails + {len(ALL_TOOLS)} Tools")
if MCP_AVAILABLE:
    print("   └── SQLite MCP: ✅ Connected")

# Define the state
class AgentState(TypedDict):
    input: str
    output: str
    context: str
    history: List[dict]
    messages: List
    blocked: bool
    block_reason: str

# Initialize the OpenAI model with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(ALL_TOOLS)
llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

# Tool mapping for execution
TOOL_MAP = {
    "get_contact_info": get_contact_info,
    "save_memory": save_memory,
    "recall_memories": recall_memories,
    "delete_memory": delete_memory,
    "web_search": web_search,
    "web_news": web_news,
    # SQLite MCP tools
    "mcp_query_memories": mcp_query_memories,
    "mcp_memory_stats": mcp_memory_stats,
    "mcp_search_memories": mcp_search_memories,
}

# Guardrails configuration
guardrail_config = GuardrailConfig(
    enable_injection_detection=True,
    enable_toxicity_check=True,
    enable_topic_check=True,
    enable_pii_detection=True,
    enable_hallucination_check=True,
    max_input_length=1000,
    max_output_length=3000,
)

# System prompt
SYSTEM_PROMPT = """You are an AI assistant that helps developers understand this LangGraph + RAG + MCP boilerplate codebase.
You answer questions based on the provided context from the actual source code files.

This is an open-source boilerplate featuring:
- LangGraph for agent orchestration
- RAG (Retrieval Augmented Generation) for code understanding
- MCP (Model Context Protocol) for tool integration
- Guardrails for input/output validation
- Full-stack setup (FastAPI backend + React frontend)

You have access to the following tools:
1. get_contact_info: Use when users ask for contact information, email, phone number.
2. save_memory: Use when users say "remember", "save", "don't forget", "store this".
3. recall_memories: Use when users ask "what do you remember", "my saved memories".
4. delete_memory: Use when users want to forget/delete something from memory.
5. web_search: Use when RAG context is empty/not relevant OR user asks about current events, news, or external documentation.

WHEN TO USE web_search:
- User asks about current news, events, or recent information
- User asks about external libraries documentation
- RAG context is empty or irrelevant to the question
- User explicitly says "search online", "look up", "find on internet"
- Questions about technologies not covered in this codebase

IMPORTANT - When to use save_memory:
- User says "remember my name is X" → save_memory(memory="User's name is X", category="personal")
- User says "please save that I prefer Python" → save_memory(memory="User prefers Python", category="preference")
- User says "don't forget I'm working on the RAG module" → save_memory(memory="User is working on RAG module", category="context")

User's Saved Memories (from database):
{saved_memories}

Retrieved Code Context from the codebase:
{rag_context}

Guidelines:
- Answer in a helpful, clear, and educational tone like a senior developer
- When showing code, use proper markdown code blocks with syntax highlighting
- Explain WHY the code works, not just WHAT it does
- Reference specific file paths when discussing code
- Use saved memories to personalize responses
- When user asks to remember something, USE the save_memory tool
"""

def format_history(history: List[dict]) -> List:
    """Format conversation history as LangChain messages."""
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted.append(AIMessage(content=content))
    return formatted

def get_saved_memories_text(user_id: str = "default") -> str:
    """Get saved memories formatted for system prompt."""
    memories = get_all_memories(user_id)
    if not memories:
        return "No saved memories yet."
    return "\n".join([f"• {m}" for m in memories[:10]])  # Limit to 10 most recent

def validate_input_node(state: AgentState):
    """Validate user input through guardrails."""
    question = state["input"]
    
    print("\n" + "="*60)
    print("🛡️  INPUT GUARDRAILS")
    print("="*60)
    
    result = validate_input(question, guardrail_config)
    
    if not result.is_valid:
        print(f"❌ BLOCKED: {result.blocked_reason}")
        return {
            "blocked": True,
            "block_reason": result.blocked_reason,
            "output": f"I'm sorry, but I cannot process this request. {result.blocked_reason}."
        }
    
    print(f"✅ Input validated")
    print("="*60 + "\n")
    
    return {
        "input": result.sanitized_input,
        "blocked": False,
        "block_reason": ""
    }

def retrieve_context(state: AgentState):
    """Retrieve relevant context from RAG."""
    if state.get("blocked", False):
        return {"context": ""}
    
    question = state["input"]
    history = state.get("history", [])
    
    print("="*60)
    print(f"🔍 RAG {RAG_VERSION} RETRIEVAL")
    print("="*60)
    
    search_query = question
    if history:
        recent = history[-4:] if len(history) > 4 else history
        context_summary = " ".join([m.get("content", "")[:100] for m in recent])
        search_query = f"{context_summary} {question}"
    
    try:
        context = get_context(search_query, k=3)
        print(f"✅ Retrieved {len(context)} chars of context")
    except Exception as e:
        print(f"❌ RAG error: {e}")
        context = ""
    
    print("="*60 + "\n")
    return {"context": context}

def call_model_with_tools(state: AgentState):
    """Call LLM with tools support."""
    if state.get("blocked", False):
        return {}
    
    question = state["input"]
    rag_context = state.get("context", "")
    history = state.get("history", [])
    
    print("="*60)
    print("🤖 CALLING LLM (with tools)")
    print(f"   History: {len(history)} messages")
    print("="*60)
    
    # Get saved memories
    saved_memories = get_saved_memories_text()
    
    # Build system message
    system_message = SYSTEM_PROMPT.format(
        saved_memories=saved_memories,
        rag_context=rag_context if rag_context else "No specific context retrieved."
    )
    
    # Build messages
    messages = [SystemMessage(content=system_message)]
    
    if history:
        messages.extend(format_history(history))
    
    messages.append(HumanMessage(content=question))
    
    # Call LLM with tools
    response = llm_with_tools.invoke(messages)
    
    print(f"✅ LLM responded")
    
    # Check if tool was called
    if response.tool_calls:
        print(f"🔧 Tool called: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": messages + [response]}
    else:
        print(f"   No tools needed, direct response")
        return {"output": response.content, "messages": []}

def execute_tools(state: AgentState):
    """Execute any tool calls."""
    messages = state.get("messages", [])
    
    if not messages:
        return {}
    
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {}
    
    print("="*60)
    print("🔧 EXECUTING TOOLS")
    print("="*60)
    
    # Execute tools
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"   Running: {tool_name}({tool_args})")
        
        # Execute the tool
        if tool_name in TOOL_MAP:
            result = TOOL_MAP[tool_name].invoke(tool_args)
            tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            print(f"   Result: {str(result)[:100]}...")
    
    print("="*60 + "\n")
    
    # Get final response from LLM with tool results
    all_messages = messages + tool_results
    final_response = llm.invoke(all_messages)
    
    return {"output": final_response.content, "messages": []}

def should_use_tools(state: AgentState) -> str:
    """Check if tools were called."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "use_tools"
    return "skip_tools"

def validate_output_node(state: AgentState):
    """Validate LLM output through guardrails."""
    if state.get("blocked", False):
        return {}
    
    output = state.get("output", "")
    question = state.get("input", "")
    context = state.get("context", "")
    
    print("="*60)
    print("🛡️  OUTPUT GUARDRAILS")
    print("="*60)
    
    result = validate_output(
        output=output,
        question=question,
        context=context,
        config=guardrail_config
    )
    
    if result.modified:
        print(f"✏️  Output was modified")
    
    print("✅ Output validated")
    print("="*60 + "\n")
    
    return {"output": result.sanitized_output}

def should_continue(state: AgentState) -> str:
    """Conditional edge: skip if blocked."""
    if state.get("blocked", False):
        return "blocked"
    return "continue"

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("validate_input", validate_input_node)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("call_model", call_model_with_tools)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("validate_output", validate_output_node)

workflow.set_entry_point("validate_input")

workflow.add_conditional_edges(
    "validate_input",
    should_continue,
    {"continue": "retrieve", "blocked": END}
)

workflow.add_edge("retrieve", "call_model")

workflow.add_conditional_edges(
    "call_model",
    should_use_tools,
    {"use_tools": "execute_tools", "skip_tools": "validate_output"}
)

workflow.add_edge("execute_tools", "validate_output")
workflow.add_edge("validate_output", END)

agent_executor = workflow.compile()

def run_agent(question: str, history: Optional[List[dict]] = None) -> str:
    """Run agent with conversation history."""
    result = agent_executor.invoke({
        "input": question, 
        "context": "",
        "output": "",
        "history": history or [],
        "messages": [],
        "blocked": False,
        "block_reason": ""
    })
    return result["output"]

async def run_agent_stream(question: str, history: Optional[List[dict]] = None) -> Generator[str, None, None]:
    """Run agent with streaming output."""
    
    history = history or []
    
    # 1. Validate input
    input_result = validate_input(question, guardrail_config)
    
    if not input_result.is_valid:
        yield f"data: {json.dumps({'type': 'error', 'content': input_result.blocked_reason})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    
    sanitized_question = input_result.sanitized_input
    
    # 2. Retrieve context
    yield f"data: {json.dumps({'type': 'status', 'content': 'Searching knowledge base...'})}\n\n"
    
    search_query = sanitized_question
    if history:
        recent = history[-4:] if len(history) > 4 else history
        context_summary = " ".join([m.get("content", "")[:100] for m in recent])
        search_query = f"{context_summary} {sanitized_question}"
    
    try:
        rag_context = get_context(search_query, k=3)
    except:
        rag_context = ""
    
    # 3. Get saved memories
    saved_memories = get_saved_memories_text()
    
    # 4. Build messages
    yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
    
    system_message = SYSTEM_PROMPT.format(
        saved_memories=saved_memories,
        rag_context=rag_context if rag_context else "No specific context retrieved."
    )
    
    messages = [SystemMessage(content=system_message)]
    if history:
        messages.extend(format_history(history))
    messages.append(HumanMessage(content=sanitized_question))
    
    # 5. Check if tool is needed
    initial_response = llm_with_tools.invoke(messages)
    
    if initial_response.tool_calls:
        yield f"data: {json.dumps({'type': 'status', 'content': 'Processing your request...'})}\n\n"
        
        tool_results = []
        for tool_call in initial_response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name].invoke(tool_call["args"])
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        
        all_messages = messages + [initial_response] + tool_results
        
        full_response = ""
        for chunk in llm_streaming.stream(all_messages):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
    else:
        full_response = ""
        for chunk in llm_streaming.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
    
    # 6. Validate output
    output_result = validate_output(
        output=full_response,
        question=sanitized_question,
        context=rag_context,
        config=guardrail_config
    )
    
    if output_result.modified:
        yield f"data: {json.dumps({'type': 'replace', 'content': output_result.sanitized_output})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
