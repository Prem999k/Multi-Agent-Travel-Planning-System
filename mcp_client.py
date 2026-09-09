import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


# Load this project's .env file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Read MCP API keys.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Local MCP server locations.
AVIATIONSTACK_SERVER = os.path.join(BASE_DIR, "aviationstack-mcp")
WEATHER_SERVER = os.path.join(BASE_DIR, "weather_mcp_server.py")
AVIATIONSTACK_PYTHON = os.path.join(
    AVIATIONSTACK_SERVER,
    ".venv",
    "Scripts",
    "python.exe",
)


def _require_key(value: str | None, name: str) -> str:
    # Give a clear error before MCP receives an invalid None value.
    if not value:
        raise RuntimeError(f"{name} is missing from .env.")
    return value


def _aviationstack_command() -> tuple[str, list[str]]:
    # Prefer the dedicated AviationStack environment used by the server.
    if os.path.isfile(AVIATIONSTACK_PYTHON):
        return AVIATIONSTACK_PYTHON, [
            "-m",
            "aviationstack_mcp",
            "mcp",
            "run",
        ]

    # Fallback to uv when the nested MCP environment is unavailable.
    return "uv", [
        "--directory",
        AVIATIONSTACK_SERVER,
        "run",
        "-m",
        "aviationstack_mcp",
        "mcp",
        "run",
    ]


# Create a fresh MCP client per server call.
# This avoids stale asyncio/session objects across Streamlit reruns.
async def _call_server_tool(
    server_config: dict,
    tool_name: str,
    args: dict | None = None,
):
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()

    tool = next(
        (item for item in tools if item.name == tool_name),
        None,
    )

    if tool is None:
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found."
        )

    return await tool.ainvoke(args or {})


# Search airports through AviationStack MCP.
async def list_airports(
    search: str = "",
    limit: int = 10,
):
    key = _require_key(
        AVIATION_STACK_API_KEY,
        "AVIATION_STACK_API_KEY",
    )
    command, args = _aviationstack_command()

    return await _call_server_tool(
        {
            "aviationstack": {
                "transport": "stdio",
                "command": command,
                "args": args,
                "env": {
                    "AVIATION_STACK_API_KEY": key,
                },
            }
        },
        "list_airports",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


# Search airlines through AviationStack MCP.
async def list_airlines(
    search: str = "",
    limit: int = 10,
):
    key = _require_key(
        AVIATION_STACK_API_KEY,
        "AVIATION_STACK_API_KEY",
    )
    command, args = _aviationstack_command()

    return await _call_server_tool(
        {
            "aviationstack": {
                "transport": "stdio",
                "command": command,
                "args": args,
                "env": {
                    "AVIATION_STACK_API_KEY": key,
                },
            }
        },
        "list_airlines",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


# Search hotels and travel information through Tavily MCP.
async def tavily_search(query: str):
    key = _require_key(
        TAVILY_API_KEY,
        "TAVILY_API_KEY",
    )

    return await _call_server_tool(
        {
            "tavily": {
                "transport": "streamable_http",
                "url": (
                    "https://mcp.tavily.com/mcp/"
                    f"?tavilyApiKey={key}"
                ),
            }
        },
        "tavily_search",
        {
            "query": query,
        },
    )


# Get current weather through the local Weather MCP server.
async def current_weather(city: str):
    key = _require_key(
        OPENWEATHER_API_KEY,
        "OPENWEATHER_API_KEY",
    )

    return await _call_server_tool(
        {
            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [WEATHER_SERVER],
                "env": {
                    "OPENWEATHER_API_KEY": key,
                },
            }
        },
        "get_current_weather",
        {
            "city": city,
        },
    )


# Get forecast information through the local Weather MCP server.
async def forecast(city: str):
    key = _require_key(
        OPENWEATHER_API_KEY,
        "OPENWEATHER_API_KEY",
    )

    return await _call_server_tool(
        {
            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [WEATHER_SERVER],
                "env": {
                    "OPENWEATHER_API_KEY": key,
                },
            }
        },
        "get_forecast",
        {
            "city": city,
        },
    )
