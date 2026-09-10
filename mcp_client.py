import os
import shutil
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATIONSTACK_API_KEY")
    or os.getenv("AVIATION_STACK_API_KEY")
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"

# =========================================================
# Environment helpers
# =========================================================

def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=_require_env(
        "GROQ_API_KEY",
        GROQ_API_KEY,
    ),
)

# =========================================================
# MCP client configuration
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(WEATHER_SERVER_PATH),
            ],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)

# =========================================================
# Generic MCP tool loader
# =========================================================

async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATIONSTACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the "
                "terminal, activate the travel environment, "
                "and run `uvx --version`."
            )

    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )

        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                "Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    tools = await client.get_tools(
        server_name=server_name,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(
                    item.name
                    for item in tools
                )
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool

# =========================================================
# MCP diagnostic function
# =========================================================

async def get_all_tools() -> None:
    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )

            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )

            print(
                f"{server_name}: OK -> "
                f"{tool_names}"
            )

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}: {exc}"
            )

# =========================================================
# Tavily MCP
# =========================================================

search_tool = None


async def initialize_mcp():
    global search_tool

    if search_tool is not None:
        return

    _require_env(
        "TAVILY_API_KEY",
        TAVILY_API_KEY,
    )

    tools = await client.get_tools(
        server_name="tavily",
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    search_tool = tools_by_name.get(
        "tavily_search",
    )

    if search_tool is None:
        available_tools = ", ".join(
            sorted(tools_by_name.keys())
        )

        raise RuntimeError(
            "Tavily MCP connected, but the "
            "'tavily_search' tool was not found. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )


async def tavily_mcp_search(query: str):
    await initialize_mcp()

    return await search_tool.ainvoke(
        {
            "query": query,
        }
    )

# =========================================================
# AviationStack MCP
# =========================================================

aviation_tools: dict[str, Any] = {}


async def initialize_aviation_tools():
    global aviation_tools

    if aviation_tools:
        return

    _require_env(
        "AVIATIONSTACK_API_KEY",
        AVIATION_STACK_API_KEY,
    )

    if shutil.which("uvx") is None:
        raise RuntimeError(
            "uvx was not found. Install uv and make sure "
            "`uvx --version` works in the active environment."
        )

    tools = await client.get_tools(
        server_name="aviationstack",
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
    }

    if not aviation_tools:
        raise RuntimeError(
            "AviationStack MCP connected but "
            "returned no tools."
        )


async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    await initialize_aviation_tools()

    tool = aviation_tools.get(tool_name)

    if tool is None:
        available_tools = ", ".join(
            sorted(aviation_tools.keys())
        )

        raise ValueError(
            f"AviationStack tool '{tool_name}' "
            "was not found. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )

    return await tool.ainvoke(
        tool_args or {}
    )

# =========================================================
# Weather MCP
# =========================================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool
    global forecast_tool

    if (
        weather_tool is not None
        and forecast_tool is not None
    ):
        return

    _require_env(
        "OPENWEATHER_API_KEY",
        OPENWEATHER_API_KEY,
    )

    if not WEATHER_SERVER_PATH.is_file():
        raise FileNotFoundError(
            "Weather MCP server file was not found: "
            f"{WEATHER_SERVER_PATH}"
        )

    tools = await client.get_tools(
        server_name="weather",
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    weather_tool = tools_by_name.get(
        "get_current_weather",
    )

    forecast_tool = tools_by_name.get(
        "get_forecast",
    )

    missing_tools = []

    if weather_tool is None:
        missing_tools.append(
            "get_current_weather"
        )

    if forecast_tool is None:
        missing_tools.append(
            "get_forecast"
        )

    if missing_tools:
        available_tools = ", ".join(
            sorted(tools_by_name.keys())
        )

        raise RuntimeError(
            "Missing Weather MCP tools: "
            f"{', '.join(missing_tools)}. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )


async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city,
        }
    )


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )

# =========================================================
# Destination extractor
# =========================================================

def extract_destination(query: str) -> str:
    prompt = f"""
Extract only the destination city or country
from the user's travel request for TravelMind AI.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination