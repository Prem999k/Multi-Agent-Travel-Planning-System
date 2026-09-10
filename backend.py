import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid
import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
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
# PostgreSQL
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
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
    api_key=GROQ_API_KEY
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


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


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

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

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner for TravelMind AI."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels and practical accommodation options for {state['user_query']}"

    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Weather Agent
# =========================

def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
""",
        "messages": [
            AIMessage(
                content="Weather information fetched."
            )
        ]
    }


# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a practical TravelMind AI travel blueprint.

The user needs a real travel plan, not a generic travel article.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

IMPORTANT TRIP RULES:

- Read the user's destination, origin, duration, number of travelers, budget, and any other explicit requirements.
- Treat the stated budget as the TOTAL budget for the entire group.
- Do not treat the budget as per-person unless the user explicitly says "per person".
- All costs must be presented in INR (₹).
- Adjust financial estimates according to the actual number of travelers.
- Follow any explicit user requirement over the default rules below.

ACCOMMODATION RULE:

- Use 1 room when there are 4 or fewer travelers.
- Use 2 rooms when there are more than 4 travelers.
- If the user explicitly asks for a different number of rooms or accommodation arrangement, follow the user's request.
- Do not add unnecessary rooms.
- Reflect the chosen number of rooms in accommodation estimates.

TRANSPORT RULES FOR INDIA:

- Prefer train travel from Hyderabad when practical and affordable.
- If train travel is not practical, consider buses.
- For local travel, prefer practical low-cost options such as local buses, metro, auto, Rapido, shared transport, or other suitable local transport.
- Use domestic flights when necessary, clearly requested, or materially more practical for the route and budget.
- Choose transport based on practicality, travel time, and the user's TOTAL group budget.

TRANSPORT RULES FOR INTERNATIONAL TRIPS:

- Start with the practical international flight journey from Hyderabad to the destination.
- Clearly mention a connecting flight or major hub when applicable.
- After reaching the destination, use available practical local or intercity transport such as trains, metro, buses, domestic flights, taxis, or other suitable options.
- Do not assume every city-to-city journey requires a flight.
- Choose transport based on practicality, travel time, and the user's TOTAL group budget.

BUDGET RULES:

- The stated budget is the TOTAL budget for all travelers.
- Do not convert the user's stated budget into a per-person target in the final output.
- Adjust accommodation, transport, food, activities and total projected cost based on the number of travelers.
- Shared rooms and shared transport can be used where practical.
- If the requested budget is unrealistic, clearly state that the budget may be exceeded.
- Do not force unrealistic prices simply to remain under the user's budget.
- Use realistic estimates when exact prices are unavailable.
- Do not claim confirmed bookings or live prices unless supported by supplied data.

ITINERARY RULES:

- Keep the requested destination.
- Keep the requested number of days.
- Keep the overall route and sightseeing sequence consistent.
- Do not replace the established itinerary simply because the traveler count or budget changes.
- Traveler count and budget should mainly affect accommodation, transport choices and financial estimates.
- Other explicit user requirements should also be respected without changing the overall output structure.

Create a concise and realistic itinerary that:

- Respects the user's requested number of days.
- Uses the supplied flight, hotel and weather research.
- Is practical and easy to follow.
- Is budget-aware.
- Includes useful transport decisions.
- Includes accommodation guidance.
- Includes weather considerations when relevant.
- Clearly identifies estimates.
- Does not claim confirmed bookings or reservations.
- Does not invent precise live prices.
- Prioritizes concrete places and useful travel decisions.

Use this structure:

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

Use a compact Markdown table covering relevant categories such as:

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

Day 2: [Clear descriptive title]

Provide 3–5 clear practical points with the same level of detail.

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

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a professional travel planner for TravelMind AI. "
                "Create concise, practical and budget-aware travel blueprints "
                "designed for real-world use."
            )
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Prepare the final TravelMind AI travel blueprint for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Itinerary:
{state['itinerary']}

STRICT OUTPUT RULE:

- Preserve the established TravelMind AI presentation structure.
- Read the user's destination, duration, number of travelers, total budget, and other explicit requirements.
- Treat the stated budget as the TOTAL budget for the entire group.
- Do not call the user's stated budget a per-person budget unless the user explicitly requested that.
- Recalculate financial estimates according to the actual number of travelers.
- All financial amounts must be in INR (₹).

ACCOMMODATION RULE:

- Default: 1 room for 4 or fewer travelers.
- Default: 2 rooms for more than 4 travelers.
- If the user explicitly requests another room count or accommodation arrangement, follow the user's request.
- Do not add unnecessary rooms.
- Show the room assumption in the accommodation cost strategy.

TRANSPORT RULES:

For trips within India:
- Prefer trains from Hyderabad when practical and affordable.
- If train travel is not practical, consider buses.
- For local movement, prefer practical affordable options such as local buses, metro, auto, Rapido, shared transport or other available local transport.
- Use domestic flights only when necessary, explicitly requested, or materially more practical.

For international trips:
- Start with the practical international flight from Hyderabad to the destination.
- Clearly state connecting flights or major hubs where applicable.
- After arrival, use practical available local/intercity transport such as trains, metro, buses, domestic flights, taxis or other suitable options.
- Choose the transport combination according to practicality and the TOTAL group budget.

ITINERARY PRESERVATION:

- Do not change the destination.
- Do not change the requested duration.
- Do not change the overall route.
- Do not change the sightseeing sequence simply because the traveler count or budget changes.
- Keep the supplied itinerary as the basis for the final day-by-day plan.
- Financial changes and practical transport changes must not destroy the established itinerary flow.
- Follow any explicit user requirement over default assumptions.

Return a polished, practical travel document.

Use the following presentation structure:

# [Destination / Trip Name]

## [X]-Day Itinerary & Financial Blueprint

### Trip at a Glance

Include:
- Origin
- Destination
- Duration
- Travelers
- Budget
- Travel style

### Financial Breakdown

Use a Markdown table:

| Expense Category | Cost Estimation Strategy | Estimated Total (INR) |
| --- | --- | --- |
| Flights / Intercity Travel | Practical transport strategy | Estimate |
| Accommodation | Budget / mid-range strategy | Estimate |
| Local Transport | Shared / public / intercity strategy | Estimate |
| Food & Miscellaneous | Local food / entry fees / buffer | Estimate |
| Activities / Sightseeing | Relevant activity guidance | Estimate |
| Total Projected Cost | Budget feasibility summary | Estimate |

IMPORTANT FINANCIAL RULES:

- These totals represent the entire group.
- Use the actual traveler count from the user request.
- Use the correct room assumption based on traveler count.
- Do not change the user's requested total budget into a per-person budget.
- Clearly state when the requested budget is likely to be exceeded.
- Use estimates and do not invent precise live prices.

### Day-by-Day Itinerary

Day 1: [Clear descriptive title]

Provide 3–5 clear, information-dense points.

Each point should be a complete sentence.

When the day involves travel, explicitly state the journey.

For example:
- Flight from Hyderabad to Moscow via a connecting hub.
- Transfer from the airport to the hotel using the recommended local transport.
- Check in to the hotel and freshen up.
- Visit the main nearby attraction in the logical order.
- Return to the hotel after dinner.

Do not replace a travel action with vague wording such as only:
- "Land at the airport."
- "Transfer."
- "Hotel."
- "Evening."

Day 2: [Clear descriptive title]

Provide 3–5 clear, information-dense points with the same level of detail.

Continue through every requested day.

IMPORTANT DAY-BY-DAY RULES:

- Keep the same requested number of days.
- Keep the overall route and sightseeing sequence consistent with the supplied itinerary.
- Make the sentences clear and useful.
- Give medium detail: enough for the traveler to know what to do, but not a long report.
- Do not split every day into Morning / Afternoon / Evening subsections.
- Do not create unnecessary sub-sections inside each day.
- Do not change the day-by-day plan simply because the number of travelers or budget changes.

### Essential Money-Saving Tips

Include practical tips such as:

- Leverage Group Size
- Eat Local
- Advance Booking
- Transport Strategy
- Activity Savings

### Things to Remember

Provide concise trip-specific reminders covering only the most useful items, such as:
- Documents, tickets or reservations
- Timing and transfer reminders
- Weather or local-condition considerations
- Payment/cash/connectivity requirements when relevant
- Important destination-specific practical points

### Packing & Practical Notes

Provide a concise destination-specific packing and preparation list based on:
- Expected weather
- Duration
- Activities
- Local conditions
- Destination-specific practical needs

### Final Recommendations

Provide a short final recommendation section summarizing the most important actions before and during the trip.

IMPORTANT RULES:

- Be clear and practical.
- Respect the user's number of days, travelers and TOTAL budget.
- Preserve useful information from the supplied research.
- Clearly label estimates.
- Do not claim bookings, tickets or reservations are confirmed unless explicitly supported by the research.
- Do not invent precise live prices.
- Mention when live flight data does not provide ticket pricing.
- Include useful weather-based travel advice when relevant.
- Keep the output concise and decision-useful.
- Do not mention internal agents, MCP, LangGraph, prompts, tools, APIs or implementation details.
- Do not add unrelated sections such as methodology, AI notes or conclusion.
- Do not use Morning / Afternoon / Evening subsections for every day.
- Return Markdown only.

The document should feel like a professionally prepared TravelMind AI travel blueprint.
"""

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a professional AI travel planning assistant for TravelMind AI. "
                "Your output must be practical, realistic, budget-aware and useful for a real traveler."
            )
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }