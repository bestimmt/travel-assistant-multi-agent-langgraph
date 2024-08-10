from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from _zero_shot_agent.state import State
from _zero_shot_agent.agent import (
    Assistant,
    create_runnable_zero_shot,
    zero_shot_agent_tools,
)
from tools.helpers import create_tool_node_with_fallback


def build_graph_zero_shot():
    builder = StateGraph(State)

    # define nodes: these do the work
    builder.add_node(
        node="assistant",
        action=Assistant(runnable=create_runnable_zero_shot(zero_shot_agent_tools)),
    )

    builder.add_node(
        node="tools", action=create_tool_node_with_fallback(tools=zero_shot_agent_tools)
    )

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
