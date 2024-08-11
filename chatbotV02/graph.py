from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from chatbotV02.state import State
from chatbotV02.agent import (
    Assistant,
    create_runnable,
)
from tools.helpers import create_tool_node_with_fallback, all_tools
from tools.flights import fetch_user_flight_information
from IPython.display import Image, display


# Explicitly populate the user state within the first node
# so the assistant doesn't have to use a tool just to learn about the user.
def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


def build_graph():

    # instantiate objects
    builder = StateGraph(State)
    assistant_runnable = create_runnable()
    assistant = Assistant(runnable=assistant_runnable)
    tools_ = all_tools()

    # additional step to explicitly extract user information
    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    builder.add_node(
        node="assistant",
        action=assistant,
    )

    builder.add_node(node="tools", action=create_tool_node_with_fallback(tools=tools_))

    builder.add_edge(
        start_key="fetch_user_info",
        end_key="assistant",
    )

    # this will decide: which tool to go
    builder.add_conditional_edges(source="assistant", path=tools_condition)

    builder.add_edge(start_key="tools", end_key="assistant")

    # the checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
    return graph


def draw_graph(graph):

    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
