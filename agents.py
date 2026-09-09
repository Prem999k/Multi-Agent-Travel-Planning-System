import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import (
    current_weather,
    forecast,
    list_airlines,
    list_airports,
    tavily_search,
)
from state import TravelState


# Create the shared LLM used by all agents.
llm = get_llm()


# Send a prompt to the selected LLM and return text.
def _llm_text(system: str, prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


# Safely extract JSON from an LLM response.
def _json_from_llm(text: str) -> dict[str, Any]:
    text = text.strip()

    # Remove markdown code fences when the model adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Find the JSON object inside possible extra text.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain valid JSON.")

    return json.loads(text[start : end + 1])


# Keep large MCP/LLM outputs within reasonable prompt limits.
def _clip(value: Any, limit: int = 2500) -> str:
    text = str(value)

    if len(text) <= limit:
        return text

    return text[:limit] + "\n[Content truncated]"


# Validate the request and decide which specialist agents are needed.
def supervisor_agent(state: TravelState):
    query = state["user_query"]

    # Input guardrail checks whether the request is travel-related.
    guardrail_prompt = f"""
Determine whether the following request is a valid travel planning request.

Allow requests related to:
- travel planning
- destinations
- flights
- hotels
- weather for travel
- budgets
- itineraries
- transportation
- sightseeing
- travel preferences

Reject requests that are clearly unrelated to travel planning.

Return only JSON in this format:

{{
    "allowed": true,
    "reason": ""
}}

User request:
{query}
"""

    guardrail_raw = _llm_text(
        "You are a travel request validation guardrail. Return strict JSON only.",
        guardrail_prompt,
    )

    guardrail_result = _json_from_llm(guardrail_raw)

    # Stop the graph when the request fails the guardrail.
    if not guardrail_result.get("allowed", False):
        reason = guardrail_result.get(
            "reason",
            "Request rejected by input guardrail.",
        )

        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(content=reason),
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Supervisor decides which specialist agents are required.
    prompt = f"""
You are the supervisor of a real-world multi-agent travel planning system.

Decide which specialist agents are needed for this user request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare guidance
- hotel_agent: hotels, stays, neighborhoods, accommodation
- weather_agent: weather, climate, season, packing, forecast
- budget_agent: budget, affordability, costs, price constraints
- itinerary_agent: required for the final travel plan

Return only JSON with this schema:

{{
    "selected_agents": [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent"
    ],
    "trip_constraints": {{
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }},
    "reasoning": ""
}}

User request:
{query}
"""

    raw = _llm_text(
        "You route work to travel specialist agents. Return strict JSON only.",
        prompt,
    )

    parsed = _json_from_llm(raw)

    selected = parsed.get("selected_agents", [])

    # Keep only valid agent names.
    valid_agents = {
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    }

    selected = [
        agent
        for agent in selected
        if agent in valid_agents
    ]

    # Itinerary is required for a usable travel plan.
    if "itinerary_agent" not in selected:
        selected.append("itinerary_agent")

    return {
        "selected_agents": selected,
        "trip_constraints": parsed.get(
            "trip_constraints",
            {},
        ),
        "supervisor_reasoning": parsed.get(
            "reasoning",
            "",
        ),
        "messages": [
            AIMessage(
                content="Supervisor created the travel planning workflow."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Collect flight information from AviationStack MCP.
def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    destination = constraints.get("destination", "")

    try:
        airports = asyncio.run(
            list_airports(
                destination,
                limit=10,
            )
        )
    except Exception as exc:
        airports = {
            "error": f"Airport MCP unavailable: {exc}"
        }

    try:
        airlines = asyncio.run(
            list_airlines(
                "",
                limit=10,
            )
        )
    except Exception as exc:
        airlines = {
            "error": f"Airline MCP unavailable: {exc}"
        }

    prompt = f"""
Create concise flight guidance for this trip.

User request:
{_clip(query, 2000)}

Trip constraints:
{_clip(constraints, 1500)}

Airport MCP data:
{_clip(airports, 2200)}

Airline MCP data:
{_clip(airlines, 2200)}

Include:
- likely departure and arrival airports
- relevant airlines
- estimated flight duration
- realistic fare guidance
- peak-season considerations
- practical booking advice

Clearly distinguish estimated information from live MCP data.
"""

    result = _llm_text(
        "You are a practical flight planning specialist.",
        prompt,
    )

    return {
        "flight_results": result,
        "messages": [
            AIMessage(content="Flight planning completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Search for hotels and recommended areas through Tavily MCP.
def hotel_agent(state: TravelState):
    query = (
        "Find useful hotel options, recommended areas to stay, "
        "and accommodation guidance for this trip: "
        f"{state['user_query']}"
    )

    try:
        result = asyncio.run(
            tavily_search(query)
        )
    except Exception as exc:
        result = {
            "error": f"Hotel MCP unavailable: {exc}"
        }

    # Keep raw MCP information bounded for downstream agents.
    hotel_text = _clip(result, 3500)

    return {
        "hotel_results": hotel_text,
        "messages": [
            AIMessage(content="Accommodation research completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Get current weather and forecast through the Weather MCP server.
def weather_agent(state: TravelState):
    constraints = state.get("trip_constraints", {})
    city = constraints.get("destination", "")

    try:
        weather_data = asyncio.run(
            current_weather(city)
        )
    except Exception as exc:
        weather_data = {
            "error": f"Current weather unavailable: {exc}"
        }

    try:
        forecast_data = asyncio.run(
            forecast(city)
        )
    except Exception as exc:
        forecast_data = {
            "error": f"Forecast unavailable: {exc}"
        }

    result = f"""
Current weather:
{_clip(weather_data, 1200)}

Forecast:
{_clip(forecast_data, 1800)}
"""

    return {
        "weather_results": result,
        "messages": [
            AIMessage(content="Weather research completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Analyze estimated trip costs and compare them with the budget.
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User request:
{_clip(state['user_query'], 2200)}

Trip constraints:
{_clip(state.get('trip_constraints', {}), 1500)}

Flight results:
{_clip(state.get('flight_results', ''), 2200)}

Hotel results:
{_clip(state.get('hotel_results', ''), 2200)}

Weather results:
{_clip(state.get('weather_results', ''), 1600)}

Return a concise budget assessment containing:

1. Estimated cost categories
2. Major cost drivers
3. Budget risk areas
4. Money-saving suggestions
5. Whether the plan appears feasible
6. A practical estimated total range

Do not invent precise live prices.
Clearly mark estimates as estimates.
"""

    result = _llm_text(
        "You are a practical travel budget analyst.",
        prompt,
    )

    return {
        "budget_results": result,
        "messages": [
            AIMessage(content="Budget analysis completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Build a practical day-by-day itinerary using specialist outputs.
def itinerary_agent(state: TravelState):
    human_feedback = str(
        state.get("human_feedback", "") or ""
    ).strip()

    # Add HITL feedback only when a previous draft was rejected.
    revision_context = ""
    if human_feedback:
        revision_context = f"""
Previous human review feedback:
{_clip(human_feedback, 2200)}

This is a revision request. Update the itinerary specifically
according to the feedback while preserving useful parts of the
previous plan. Do not ignore the requested changes.
"""

    prompt = f"""
Create a practical draft travel itinerary.

User request:
{_clip(state['user_query'], 2200)}

Trip constraints:
{_clip(state.get('trip_constraints', {}), 1500)}

Flight information:
{_clip(state.get('flight_results', ''), 2200)}

Accommodation information:
{_clip(state.get('hotel_results', ''), 2200)}

Weather information:
{_clip(state.get('weather_results', ''), 1800)}

Budget analysis:
{_clip(state.get('budget_results', ''), 2200)}

{revision_context}

Requirements:

- Create a day-by-day itinerary.
- Use realistic travel times.
- Group nearby attractions together.
- Include morning, afternoon, and evening activities.
- Include transportation guidance.
- Consider weather and season.
- Keep the plan within the stated budget where possible.
- Mention important assumptions.
- Avoid claiming unavailable live information as confirmed.
- Make the result easy for a human to review.
- When revising, clearly apply the human's requested changes.

Keep the response focused and practical.
"""

    result = _llm_text(
        "You are an expert itinerary planner who carefully follows human review feedback.",
        prompt,
    )

    approval_request = f"""
Please review the following travel plan.

{result}

Approve it when it is suitable.

You can also provide feedback such as:
- change the budget
- change hotels
- remove an activity
- add an attraction
- change travel pace
- change transportation
- change the number of days
- change the destination focus

Reply with approval or specific revision feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Pause execution and collect human approval or revision feedback.
def human_approval_agent(state: TravelState):
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),
            "approval_request": state.get(
                "approval_request",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "",
            },
        }
    )

    # Safely read the human response.
    if not isinstance(feedback, dict):
        feedback = {
            "approved": False,
            "feedback": str(feedback),
        }

    approved = bool(
        feedback.get(
            "approved",
            False,
        )
    )

    human_feedback = str(
        feedback.get(
            "feedback",
            "",
        )
    )

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# Create the final polished response shown to the user.
def final_response_agent(state: TravelState):
    approved = state.get(
        "approved",
        False,
    )

    if approved:
        prompt = f"""
The human approved this travel plan.

Produce the final user-ready travel plan.

Original request:
{_clip(state['user_query'], 2200)}

Draft itinerary:
{_clip(state.get('itinerary', ''), 5000)}

Budget analysis:
{_clip(state.get('budget_results', ''), 2500)}

Create a polished answer with:

- Trip overview
- Transportation / flights
- Accommodation
- Weather and packing
- Day-by-day itinerary
- Budget summary
- Important travel notes

Use clear headings and practical details.
Do not mention internal agents, MCP, LangGraph, prompts, or implementation details.
"""
    else:
        prompt = f"""
The human did not approve the draft travel plan.

Create a revised user-ready response based on the requested feedback.

Original user request:
{_clip(state['user_query'], 2200)}

Draft itinerary:
{_clip(state.get('itinerary', ''), 4000)}

Human feedback:
{_clip(state.get('human_feedback', ''), 2200)}

Budget analysis:
{_clip(state.get('budget_results', ''), 2200)}

Revise the plan according to the feedback.

Produce:
- revised trip overview
- transportation
- accommodation
- weather guidance
- day-by-day itinerary
- budget
- important notes

Do not mention internal agents, MCP, LangGraph, prompts, or implementation details.
"""

    result = _llm_text(
        "You produce polished, practical, user-ready travel plans.",
        prompt,
    )

    return {
        "final_response": result,
        "messages": [
            AIMessage(content=result)
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }