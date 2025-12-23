from fastmcp.mcp_config import StdioMCPServer
import os

server = StdioMCPServer(command="echo", args=["hello"])
print(f"Env is None? {server.env is None}")
print(f"Env value: {server.env}")

# Check what happen if we don't pass env
server2 = StdioMCPServer(command="echo", args=["hello"])
print(f"Server2 Env: {server2.env}")
