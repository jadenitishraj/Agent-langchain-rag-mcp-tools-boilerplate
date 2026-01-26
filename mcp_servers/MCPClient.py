# mcp_client.py

import asyncio
import os
from typing import Optional, List
from dataclasses import dataclass
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


@dataclass
class MCPResponse:
    success: bool
    content: str
    error: Optional[str] = None


class MCPClient:
    """
    Generic MCP client.
    Works with ANY MCP server.
    """

    def __init__(self, command: List[str]):
        self.command = command
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def _start(self):
        if self.session:
            return

        # Prepare parameters
        cmd = self.command[0]
        args = self.command[1:]
        
        # Pass environment to ensure PYTHONPATH is correct
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Use simple Stdio parameters
        server_params = StdioServerParameters(command=cmd, args=args, env=env)
        
        print(f"DEBUG: Starting stdio_client with {cmd} {args}")

        # Enter the stdio_client context
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        print("DEBUG: stdio_client started")
        
        self.session = ClientSession(read, write)
        print("DEBUG: Initializing session...")
        await self.session.initialize()
        print("DEBUG: Session initialized")

    async def call(self, tool: str, args: dict) -> MCPResponse:
        try:
            await self._start()
            result = await self.session.call_tool(tool, args)

            if result.content:
                # TextContent is likely the type, having 'text' attribute
                return MCPResponse(True, result.content[0].text)
            return MCPResponse(False, "", "Empty response")

        except Exception as e:
            return MCPResponse(False, "", str(e))

    async def list_tools(self):
        await self._start()
        return await self.session.list_tools()

    async def close(self):
        # Unwind the stack (closes stdio_client and session if it was context managed, 
        # but ClientSession might be just initialized. mcp typically closes via transport closure)
        if self.session:
             # If ClientSession has close, call it, but closing transport via exit stack should be enough usually
             # Checking if explicit close is needed... safe to let exit stack handle it if stdio_client handles it.
             pass

        await self.exit_stack.aclose()
        self.session = None


# Sugar API (this is the killer feature)
def connect(cmd: str) -> MCPClient:
    """
    Example:
        connect("python -m mcp_servers.sqlite_server")
    """
    return MCPClient(cmd.split())
