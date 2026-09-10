import os
import certifi
from dotenv import load_dotenv
from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)

# =========================
# Environment
# =========================

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# =========================
# PostgreSQL
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

# =========================
# Groq API
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )

# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
)

# =========================
# State
# =========================

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    # HITL
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int

# =========================
# Shared agent configuration
# =========================

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

def _llm_text(
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)

def _json_from_llm(
    text: str,
) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "The model did not return a JSON object."
        )

    return json.loads(
        text[start:end + 1]
    )

def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }

# =========================
# Supervisor Agent + Guardrail
# =========================

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information.

Valid requests can include:
- destinations
- flights
- hotels
- weather
- budgets
- transportation
- sightseeing
- food
- packing
- visas
- itineraries

Block clearly unrelated requests and requests asking for harmful or illegal
instructions.

Do not block a valid travel request merely because details are missing.

Return strict JSON only:

{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    try:
        guardrail_raw = _llm_text(
            (
                "You are the input guardrail for TravelMind AI. "
                "Return strict JSON only."
            ),
            guardrail_prompt,
        )

        guardrail_result = _json_from_llm(
            guardrail_raw
        )

        allowed = bool(
            guardrail_result.get(
                "allowed",
                True,
            )
        )

        guardrail_reason = str(
            guardrail_result.get(
                "reason",
                "",
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:
        print(
            f"Guardrail fallback used: {exc}",
            flush=True,
        )

        allowed = True
        guardrail_reason = (
            "Guardrail validation fallback allowed the request."
        )

    if not allowed:
        reason = guardrail_reason or (
            "TravelMind AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, "
            "budget, transportation, sightseeing, packing, or itinerary."
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=reason
                )
            ],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the Supervisor Agent for TravelMind AI.

Choose only the specialist agents that are needed for the user's request.

Available agents:

- flight_agent:
  flights, airports, airlines, routes, airfare, booking advice

- hotel_agent:
  hotels, accommodation, neighborhoods, places to stay

- weather_agent:
  weather, climate, season, forecast, packing advice

- budget_agent:
  cost, affordability, budget limits, feasibility

- itinerary_agent:
  creates the integrated travel blueprint and must always be included

Return strict JSON only:

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

    try:
        supervisor_raw = _llm_text(
            (
                "You are the Supervisor Agent of a multi-agent "
                "travel-planning system. Return strict JSON only."
            ),
            supervisor_prompt,
        )

        parsed = _json_from_llm(
            supervisor_raw
        )

        requested_agents = parsed.get(
            "selected_agents",
            [],
        )

        selected_agents = [
            agent
            for agent in AGENT_ORDER
            if agent in requested_agents
            and agent in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append(
                "itinerary_agent"
            )

        constraints = _empty_constraints()

        parsed_constraints = parsed.get(
            "trip_constraints",
            {},
        )

        if isinstance(
            parsed_constraints,
            dict,
        ):
            constraints.update(
                parsed_constraints
            )

        reasoning = str(
            parsed.get(
                "reasoning",
                "",
            )
        ).strip()

        llm_calls += 1

    except Exception as exc:
        print(
            f"Supervisor fallback used: {exc}",
            flush=True,
        )

        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()

        reasoning = (
            "Supervisor parsing failed, so the original full "
            "travel workflow was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": llm_calls,
    }

# =========================
# Guardrail blocked response
# =========================

def guardrail_blocked_agent(
    state: TravelState,
):
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the TravelMind AI input guardrail."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(
                content=reason
            )
        ],
    }

# =========================
# Flight Agent
# =========================

FLIGHT_AGENT_PROMPT = """
You are the flight planning specialist for TravelMind AI.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Provide practical flight guidance covering:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving the route
4. Typical flight duration
5. Estimated airfare range when possible
6. Peak-season pricing considerations
7. Practical booking advice

Do not claim that a flight or ticket is confirmed unless the supplied data explicitly confirms it.

Prioritize realistic, concise and useful information for a real traveler.
"""

def flight_agent(
    state: TravelState,
):
    print(
        "\nINSIDE FLIGHT AGENT\n",
        flush=True,
    )

    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        print(
            "\nAIRPORTS:",
            airports,
            flush=True,
        )

        print(
            "\nAIRLINES:",
            airlines,
            flush=True,
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(
                airports
            )[:3000],
            airline_data=str(
                airlines
            )[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an expert travel flight "
                        "planner for TravelMind AI."
                    )
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

        flight_data = response.content

    except Exception as exc:
        flight_data = (
            f"Flight information unavailable: {exc}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated."
            )
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Hotel Agent
# =========================

def hotel_agent(
    state: TravelState,
):
    query = (
        "Best hotels and practical accommodation "
        f"options for {state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(
                query
            )
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Weather Agent
# =========================

def weather_agent(
    state: TravelState,
):
    city = extract_destination(
        state["user_query"]
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_search(
                city
            )
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(
                city
            )
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Budget Agent
# =========================

def budget_agent(
    state: TravelState,
):
    prompt = f"""
Analyze whether this trip is realistic for the user's TOTAL group budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Important rules:

- Treat the stated budget as the total budget for the whole group.
- Do not treat it as per-person unless the user explicitly says so.
- Use INR (₹).
- Consider traveler count.
- Consider the default accommodation rule:
  1 room for 4 or fewer travelers.
  2 rooms for more than 4 travelers.
- Respect explicit room requests.
- Consider practical transport.
- Clearly identify budget risk areas.
- Use approximate estimates when exact live prices are unavailable.
- Do not invent confirmed prices.

Return:

1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a practical travel "
                    "budget analyst for TravelMind AI."
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [
            AIMessage(
                content="Budget assessment generated."
            )
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Itinerary Agent
# =========================

def itinerary_agent(
    state: TravelState,
):
    prompt = f"""
Create a practical TravelMind AI travel blueprint.

The user needs a real travel plan, not a generic travel article.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

IMPORTANT TRIP RULES:

- Read the user's destination, origin, duration, number of travelers, budget, and all explicit requirements.
- Treat the stated budget as the TOTAL budget for the entire group.
- Do not treat the budget as per-person unless explicitly stated.
- All costs must be presented in INR (₹).
- Adjust financial estimates according to the actual number of travelers.
- Follow explicit user requirements over default rules.

ACCOMMODATION RULE:

- Use 1 room when there are 4 or fewer travelers.
- Use 2 rooms when there are more than 4 travelers.
- If the user explicitly asks for a different number of rooms or accommodation arrangement, follow that request.
- Do not add unnecessary rooms.
- Reflect the chosen room count in accommodation estimates.

TRANSPORT RULES FOR INDIA:

- Prefer train travel from Hyderabad when practical and affordable.
- If train travel is not practical, consider buses.
- For local travel, prefer practical low-cost options such as local buses, metro, auto, Rapido, shared transport, or other suitable local transport.
- Use domestic flights when necessary, clearly requested, or materially more practical.
- Choose transport based on practicality, travel time, and the user's TOTAL group budget.

TRANSPORT RULES FOR INTERNATIONAL TRIPS:

- Start with the practical international flight journey from Hyderabad to the destination.
- Clearly mention a connecting flight or major hub when applicable.
- After reaching the destination, use available practical local or intercity transport such as trains, metro, buses, domestic flights, taxis, or other suitable options.
- Do not assume every city-to-city journey requires a flight.
- Choose transport based on practicality, travel time, and the user's TOTAL group budget.

BUDGET RULES:

- The stated budget is the TOTAL budget for all travelers.
- Do not convert the user's stated budget into a per-person target.
- Adjust accommodation, transport, food, activities and total projected cost based on the number of travelers.
- Shared rooms and shared transport can be used where practical.
- If the requested budget is unrealistic, clearly state that the budget may be exceeded.
- Do not force unrealistic prices simply to remain under budget.
- Use realistic estimates when exact prices are unavailable.
- Do not claim confirmed bookings or live prices unless supported by supplied data.

ITINERARY RULES:

- Keep the requested destination.
- Keep the requested number of days.
- Keep the overall route and sightseeing sequence consistent.
- Do not replace the established itinerary simply because traveler count or budget changes.
- Traveler count and budget should mainly affect accommodation, transport choices and financial estimates.
- Respect all other explicit user requirements.

Create a concise and realistic itinerary that:

- Respects the user's requested number of days.
- Uses the supplied flight, hotel and weather research.
- Uses the budget analysis where available.
- Is practical and easy to follow.
- Is budget-aware.
- Includes useful transport decisions.
- Includes accommodation guidance.
- Includes weather considerations when relevant.
- Clearly identifies estimates.
- Does not claim confirmed bookings or reservations.
- Does not invent precise live prices.
- Prioritizes concrete places and useful travel decisions.

Use this structure exactly:

# [Destination / Trip Name]

## [X]-Day Travel Blueprint

### Trip Summary

Provide:
- Route
- Duration
- Travelers
- Travel style
- Budget considerations

### Financial Overview

Use a compact Markdown table:

| Expense Category | Estimated Cost (INR) |
| --- | --- |
| Flights / Intercity Travel | Estimate |
| Accommodation | Estimate |
| Local Transport | Estimate |
| Food & Miscellaneous | Estimate |
| Activities / Sightseeing | Estimate |
| Total Projected Cost | Estimate |

Clearly state when the requested total group budget may be exceeded.

### Day-by-Day Itinerary

Day 1: [Clear descriptive title]

Provide 3–5 clear, practical points.

Each point should be a complete sentence that tells the traveler what to do.

When travel is involved, explicitly state the journey.

Examples:
- Flight from Hyderabad to Moscow via the available connecting hub.
- Take the recommended train from Hyderabad to Tirupati.
- Transfer from the airport to the hotel using the recommended local transport.
- Check in to the hotel and keep the rest of the evening for nearby sightseeing.

Include where useful:
- Main places or activities
- Transport between places
- Logical sequence of activities
- Accommodation / overnight location
- Useful practical or budget information

Continue through every requested day.

IMPORTANT DAY-BY-DAY RULES:

- Keep the daily flow practical and chronological.
- Make the sentences clear enough for a real traveler to understand exactly what to do.
- Do not make the day sections too short.
- Do not make them excessively long.
- Do not split every day into Morning / Afternoon / Evening subsections.
- Do not create a separate long report for every day.
- Keep the same requested number of days.
- Keep the route and sightseeing sequence consistent.
- Include actual travel actions on arrival, transfer and departure days when applicable.

### Essential Money-Saving Tips

Provide concise practical recommendations about:
- Group-size savings
- Local food
- Advance booking
- Transport strategy
- Activity savings

### Things to Remember

Provide a short list of trip-specific reminders such as:
- Important documents or tickets
- Booking or reservation reminders
- Destination-specific practical considerations
- Weather-related precautions
- Payment, cash or connectivity considerations when relevant
- Important timing or transfer reminders

### Packing & Practical Notes

Provide concise destination-specific packing guidance based on:
- Weather
- Trip duration
- Main activities
- Local conditions
- Any practical destination requirements

Do not make this section excessively long.

### Final Recommendations

Provide a short final set of practical recommendations that help the traveler execute the trip smoothly.

Do not add unrelated sections.

Do not create separate long reports for flights, hotels or weather.

The final result should read like a professionally prepared travel blueprint produced by TravelMind AI.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional travel planner "
                    "for TravelMind AI. Create concise, practical "
                    "and budget-aware travel blueprints designed "
                    "for real-world use."
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. "
        "Approve it to create the final polished plan, "
        "or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Human-in-the-Loop
# =========================

def human_approval_agent(
    state: TravelState,
):
    review = interrupt(
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
            "selected_agents": state.get(
                "selected_agents",
                [],
            ),
            "supervisor_reasoning": state.get(
                "supervisor_reasoning",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(
        review.get(
            "approved",
            False,
        )
    )

    human_feedback = str(
        review.get(
            "feedback",
            "",
        )
    ).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }

# =========================
# Final Response Agent
# =========================

def final_agent(
    state: TravelState,
):
    if state.get(
        "approved",
        False,
    ):
        review_instruction = (
            "The user approved the draft. "
            "Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision.

Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Prepare the final TravelMind AI travel blueprint.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

STRICT OUTPUT FORMAT:

# [Destination / Trip Name]

## [X]-Day Travel Blueprint

### Trip Summary

### Financial Overview

Use:

| Expense Category | Estimated Cost (INR) |
| --- | --- |
| Flights / Intercity Travel | Estimate |
| Accommodation | Estimate |
| Local Transport | Estimate |
| Food & Miscellaneous | Estimate |
| Activities / Sightseeing | Estimate |
| Total Projected Cost | Estimate |

### Day-by-Day Itinerary

Day 1: [Clear descriptive title]

Provide 3–5 clear, practical points.

Every point must be a complete sentence.

When travel is involved, explicitly state the journey.

Examples:
- Flight from Hyderabad to Moscow via a connecting hub.
- Take the recommended train from Hyderabad to Tirupati.
- Transfer from the airport to the hotel using the recommended local transport.
- Check in to the hotel and continue with nearby sightseeing.

Continue for every requested day.

IMPORTANT DAY-BY-DAY RULES:

- Keep the same requested number of days.
- Keep the overall route and sightseeing sequence consistent.
- Keep the daily flow chronological and practical.
- Give medium detail.
- Do not make sections too short.
- Do not make them excessively long.
- Do not use Morning / Afternoon / Evening subsections.
- Do not replace travel actions with vague wording such as only "Land at the airport", "Transfer", "Hotel", or "Evening".

### Essential Money-Saving Tips

Include concise practical tips covering:
- Group-size savings
- Local food
- Advance booking
- Transport strategy
- Activity savings

### Things to Remember

Include concise trip-specific reminders covering:
- Documents
- Tickets or reservations
- Timing and transfer reminders
- Weather considerations
- Payment, cash or connectivity considerations
- Destination-specific practical points

### Packing & Practical Notes

Include concise destination-specific packing and preparation guidance based on:
- Weather
- Duration
- Activities
- Local conditions
- Practical destination needs

### Final Recommendations

Provide a short practical final recommendation section.

IMPORTANT:

- Treat the user's stated budget as the TOTAL group budget.
- Do not call it a per-person budget unless explicitly requested.
- Respect the actual traveler count.
- Use the correct accommodation room assumption.
- Respect the requested duration.
- Preserve the established itinerary flow.
- Apply human feedback when revision was requested.
- Clearly label estimates.
- Do not claim confirmed bookings or reservations unless explicitly supported.
- Do not invent precise live prices.
- Mention when live flight data does not provide ticket pricing.
- Include useful weather-based travel advice.
- Keep the response concise and decision-useful.
- Do not mention internal agents, MCP, LangGraph, prompts, tools, APIs or implementation details.
- Do not add unrelated sections.
- Return Markdown only.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional AI travel planning "
                    "assistant for TravelMind AI. Your output must "
                    "be practical, realistic, budget-aware and useful "
                    "for a real traveler."
                )
            ),
            HumanMessage(
                content=final_prompt
            ),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [
            response
        ],
        "llm_calls": (
            state.get(
                "llm_calls",
                0,
            ) + 1
        ),
    }

# =========================
# Dynamic Supervisor Routing
# =========================

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}

def _selected_agents(
    state: TravelState,
) -> list[str]:
    selected = state.get(
        "selected_agents",
        []
    )

    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]

def route_from_supervisor(
    state: TravelState,
) -> str:
    if not state.get(
        "guardrail_allowed",
        True,
    ):
        return "guardrail_blocked"

    selected = _selected_agents(
        state
    )

    return (
        selected[0]
        if selected
        else "itinerary_agent"
    )

def route_after_agent(
    current_agent: str,
):
    def route(
        state: TravelState,
    ) -> str:
        selected = _selected_agents(
            state
        )

        current_index = AGENT_ORDER.index(
            current_agent
        )

        for next_agent in AGENT_ORDER[
            current_index + 1:
        ]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route

# =========================
# Build Graph
# =========================

graph = StateGraph(
    TravelState
)

graph.add_node(
    "supervisor",
    supervisor_agent,
)

graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent,
)

graph.add_node(
    "flight_agent",
    flight_agent,
)

graph.add_node(
    "hotel_agent",
    hotel_agent,
)

graph.add_node(
    "weather_agent",
    weather_agent,
)

graph.add_node(
    "budget_agent",
    budget_agent,
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent,
)

graph.add_node(
    "human_approval",
    human_approval_agent,
)

graph.add_node(
    "final_agent",
    final_agent,
)

graph.add_edge(
    START,
    "supervisor",
)

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "flight_agent",
    route_after_agent(
        "flight_agent"
    ),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "hotel_agent",
    route_after_agent(
        "hotel_agent"
    ),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "weather_agent",
    route_after_agent(
        "weather_agent"
    ),
    ROUTE_MAP,
)

graph.add_conditional_edges(
    "budget_agent",
    route_after_agent(
        "budget_agent"
    ),
    ROUTE_MAP,
)

graph.add_edge(
    "itinerary_agent",
    "human_approval",
)

graph.add_edge(
    "human_approval",
    "final_agent",
)

graph.add_edge(
    "final_agent",
    END,
)

graph.add_edge(
    "guardrail_blocked",
    END,
)

# =========================
# PostgreSQL Checkpointer
# =========================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(_conn)

checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer
)

# =========================
# HITL result helpers
# =========================

def _interrupt_payload(
    result: dict[str, Any],
) -> dict[str, Any] | None:
    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        return None

    first_interrupt = interrupts[0]

    payload = getattr(
        first_interrupt,
        "value",
        first_interrupt,
    )

    return (
        payload
        if isinstance(
            payload,
            dict,
        )
        else {"value": payload}
    )

def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get(
        "messages",
        []
    )

    last_message = (
        messages[-1].content
        if messages
        else ""
    )

    answer = (
        result.get(
            "final_response"
        )
        or last_message
    )

    interrupt_payload = _interrupt_payload(
        result
    )

    if interrupt_payload:
        answer = (
            interrupt_payload.get(
                "draft_itinerary"
            )
            or result.get(
                "itinerary",
                "",
            )
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": (
            interrupt_payload is not None
        ),
        "approval_request": (
            interrupt_payload.get(
                "approval_request",
                "",
            )
            if interrupt_payload
            else result.get(
                "approval_request",
                "",
            )
        ),
        "flight_results": result.get(
            "flight_results",
            "",
        ),
        "hotel_results": result.get(
            "hotel_results",
            "",
        ),
        "weather_results": result.get(
            "weather_results",
            "",
        ),
        "budget_results": result.get(
            "budget_results",
            "",
        ),
        "itinerary": (
            interrupt_payload.get(
                "draft_itinerary",
                "",
            )
            if interrupt_payload
            else result.get(
                "itinerary",
                "",
            )
        ),
        "selected_agents": result.get(
            "selected_agents",
            [],
        ),
        "trip_constraints": result.get(
            "trip_constraints",
            {},
        ),
        "supervisor_reasoning": result.get(
            "supervisor_reasoning",
            "",
        ),
        "guardrail_allowed": result.get(
            "guardrail_allowed",
            True,
        ),
        "guardrail_reason": result.get(
            "guardrail_reason",
            "",
        ),
        "approved": result.get(
            "approved"
        ),
        "human_feedback": result.get(
            "human_feedback",
            "",
        ),
        "llm_calls": result.get(
            "llm_calls",
            0,
        ),
    }

# =========================
# Start TravelMind Agent
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
):
    if not thread_id:
        thread_id = (
            f"user_{uuid.uuid4().hex}"
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(
        result,
        thread_id,
    )

# =========================
# Resume after HITL
# =========================

def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    if not thread_id:
        raise ValueError(
            "thread_id is required to resume a travel plan."
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(
        result,
        thread_id,
    )