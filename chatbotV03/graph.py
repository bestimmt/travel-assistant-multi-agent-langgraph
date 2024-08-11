from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from chatbotV03.state import State
from chatbotV03.agent import (
    Assistant,
    create_runnable,
)
from tools.helpers import (
    create_tool_node_with_fallback,
    part_3_safe_tools,
    part_3_sensitive_tools,
)
from tools.flights import fetch_user_flight_information
from IPython.display import Image, display
from typing import Literal


# Explicitly populate the user state within the first node
# so the assistant doesn't have to use a tool just to learn about the user.
def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


def build_graph():

    # instantiate objects
    builder = StateGraph(State)
    assistant_runnable = create_runnable()
    assistant = Assistant(runnable=assistant_runnable)
    safe_tools = part_3_safe_tools()
    sensitive_tools = part_3_sensitive_tools()
    sensitive_tool_names = {t.name for t in sensitive_tools}

    # additional step to explicitly extract user information
    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    builder.add_node(
        node="assistant",
        action=assistant,
    )

    builder.add_node("safe_tools", create_tool_node_with_fallback(tools=safe_tools))
    builder.add_node(
        "sensitive_tools", create_tool_node_with_fallback(tools=safe_tools)
    )

    builder.add_edge(
        start_key="fetch_user_info",
        end_key="assistant",
    )

    def route_tools(
        state: State,
    ) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call = ai_message.tool_calls[0]
        if first_tool_call["name"] in sensitive_tool_names:
            return "sensitive_tools"
        return "safe_tools"

    builder.add_conditional_edges(
        "assistant",
        route_tools,
    )
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, interrupt_before=["sensitive_tools"])
    return graph


def draw_graph(graph):

    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
