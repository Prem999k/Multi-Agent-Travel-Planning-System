import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState


# Fixed specialist execution order.
AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


# Return only valid supervisor-selected agents.
def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents") or []
    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]


# Route after supervisor validation.
def route_from_supervisor(state: TravelState) -> str:
    selected = _selected_agents(state)

    if not selected:
        return END

    return selected[0]


# Route to the next selected specialist.
def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# Route HITL approval to finalization or itinerary revision.
def route_after_human_approval(state: TravelState) -> str:
    if state.get("approved", False):
        return "final_response"

    return "itinerary_agent"


# Build the LangGraph application.
def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            **ROUTE_MAP,
            END: END,
        },
    )

    graph.add_conditional_edges(
        "flight_agent",
        route_after_agent("flight_agent"),
        ROUTE_MAP,
    )
    graph.add_conditional_edges(
        "hotel_agent",
        route_after_agent("hotel_agent"),
        ROUTE_MAP,
    )
    graph.add_conditional_edges(
        "weather_agent",
        route_after_agent("weather_agent"),
        ROUTE_MAP,
    )
    graph.add_conditional_edges(
        "budget_agent",
        route_after_agent("budget_agent"),
        ROUTE_MAP,
    )

    graph.add_edge(
        "itinerary_agent",
        "human_approval",
    )

    graph.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {
            "itinerary_agent": "itinerary_agent",
            "final_response": "final_response",
        },
    )

    graph.add_edge(
        "final_response",
        END,
    )

    # PostgreSQL provides persistent LangGraph checkpoints.
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing from .env."
        )

    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
    )

    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    return graph.compile(
        checkpointer=checkpointer,
    )


# Compiled app used by Streamlit.
app = build_graph()
