from typing import Annotated, Any, TypedDict
import operator

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):
    # Conversation and user identity.
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    user_query: str

    # Planning decisions.
    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str

    # Specialist outputs.
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    # Human-in-the-loop state.
    approval_request: str
    human_feedback: str
    approved: bool

    # Final user-facing answer.
    final_response: str

    # Basic observability.
    llm_calls: int
