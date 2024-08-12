from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from langchain_community.tools.tavily_search import TavilySearchResults
from tools.lookup_policy import lookup_policy
from tools.car_rental import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental,
)  # 4
from tools.hotels import book_hotel, cancel_hotel, search_hotels, update_hotel  # 4
from tools.excursions import (
    book_excursion,
    cancel_excursion,
    update_excursion,
    search_trip_recommendations,
)  # 4
from tools.flights import (
    search_flights,
    fetch_user_flight_information,
    update_ticket_to_new_flight,
    cancel_ticket,
)  # 4


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def all_tools():
    """
    In part 1 and part 2 those all tools will be utilized
    Totally 18 tools.
    """
    tool_list = (
        [lookup_policy]
        + [TavilySearchResults(max_results=1)]
        + [book_car_rental, cancel_car_rental, search_car_rentals, update_car_rental]
        + [book_hotel, cancel_hotel, search_hotels, update_hotel]
        + [
            book_excursion,
            cancel_excursion,
            update_excursion,
            search_trip_recommendations,
        ]
        + [
            search_flights,
            fetch_user_flight_information,
            update_ticket_to_new_flight,
            cancel_ticket,
        ]
    )
    return tool_list


def part_3_safe_tools():
    tool_list = [
        TavilySearchResults(max_results=1),
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        search_car_rentals,
        search_hotels,
        search_trip_recommendations,
    ]
    return tool_list


def part_3_sensitive_tools():
    tool_list = [
        update_ticket_to_new_flight,
        cancel_ticket,
        book_car_rental,
        update_car_rental,
        cancel_car_rental,
        book_hotel,
        update_hotel,
        cancel_hotel,
        book_excursion,
        update_excursion,
        cancel_excursion,
    ]
    return tool_list


part_4_update_flight_safe_tools = [search_flights]
part_4_update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
part_4_book_hotel_safe_tools = [search_hotels]
part_4_book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
part_4_book_excursion_safe_tools = [search_trip_recommendations]
part_4_book_excursion_sensitive_tools = [
    book_excursion,
    update_excursion,
    cancel_excursion,
]
part_4_book_car_rental_safe_tools = [search_car_rentals]
part_4_book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
