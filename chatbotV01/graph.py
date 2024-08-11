from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from chatbotV01.state import State
from chatbotV01.agent import (
    Assistant,
    create_runnable,
)
from tools.helpers import create_tool_node_with_fallback, all_tools


def build_graph():
    builder = StateGraph(State)

    assistant_runnable = create_runnable()
    assistant = Assistant(runnable=assistant_runnable)
    tools_ = all_tools()
    # define nodes: these do the work
    builder.add_node(
        node="assistant",
        action=assistant,
    )

    builder.add_node(node="tools", action=create_tool_node_with_fallback(tools=tools_))

    # define edges:
    builder.add_edge(
        start_key=START,
        end_key="assistant",
    )

    # this will decide: which tool to go
    builder.add_conditional_edges(source="assistant", path=tools_condition)

    builder.add_edge(start_key="tools", end_key="assistant")

    # the checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph
    memory = MemorySaver()
    zero_shot_graph = builder.compile(checkpointer=memory)
    return zero_shot_graph
