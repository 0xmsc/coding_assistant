import json
import os
import requests
import uuid

class MCPBridge:
    def __init__(self, mcp_url=None):
        self.mcp_url = mcp_url or os.environ.get("MCP_SERVER_URL")
        if not self.mcp_url:
            raise ValueError("MCP_SERVER_URL environment variable not set")
        self.session_id = None
        self._initialize_session()

    def _initialize_session(self):
        # Streamable-http initialization: GET to the URL
        response = requests.get(self.mcp_url, stream=True)
        # We don't actually need to read the stream for simple tool calls if the server supports it,
        # but let's assume standard behavior.
        self.session_id = str(uuid.uuid4())

    def call_tool(self, tool_name, arguments=None):
        payload = {
            "jsonrpc": "2.0",
            "method": f"tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
            "id": str(uuid.uuid4())
        }
        
        # In streamable-http, we POST to the endpoint.
        # Note: We might need to handle the long-running nature if it's truly streamable, 
        # but for simple tool calls, FastMCP usually returns a standard JSON-RPC response if possible.
        response = requests.post(self.mcp_url, json=payload)
        if response.status_code not in [200, 202]:
            raise RuntimeError(f"Tool call failed with status {response.status_code}: {response.text}")
        
        # If 202, it means the result will come through the stream. 
        # For simplicity in this bridge, we'll try to get the immediate result if 200.
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                raise RuntimeError(f"MCP Error: {result['error']}")
            return result.get("result")
        
        # Fallback if 202: we would need to listen to the SSE stream started in __init__
        # For now, let's hope for 200 or implement minimal polling/waiting if needed.
        raise RuntimeError("Streamable-HTTP 202 response handling not yet implemented in bridge")

def _generate_bridge():
    url = os.environ.get("MCP_SERVER_URL")
    if not url: return ""
    
    bridge = MCPBridge(url)
    try:
        # Discover tools
        payload = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "list"}
        resp = requests.post(url, json=payload)
        if resp.status_code != 200: return ""
        
        tools = resp.json().get("result", {}).get("tools", [])
        
        functions = []
        for tool in tools:
            name = tool["name"]
            func_name = f"mcp_{name}"
            # Create a wrapper function
            func_template = f"""
def {func_name}(**kwargs):
    bridge = MCPBridge("{url}")
    result = bridge.call_tool("{name}", kwargs)
    # Extract text content if it's standard TextContent
    if result and "content" in result:
        return "\\n".join(c.get("text", "") for c in result["content"] if c.get("type") == "text")
    return result
"""
            functions.append(func_template)
            
        return "\n".join(functions)
    except:
        return ""
