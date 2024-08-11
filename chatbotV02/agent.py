from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from chatbotV02.state import State
from utils.constnants import LLM
from tools.helpers import all_tools


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def create_runnable():

    # i changed the prompt from the documentation
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an advanced customer support assistant for Swiss Airlines, designed to provide comprehensive and accurate assistance to users.
                Your role is to help users with their queries related to flight bookings, company policies, and other relevant services.
                You have access to various tools and databases to search for information, and you should utilize them effectively.

                When conducting searches:
                - Be thorough and persistent. If initial searches yield no results, broaden your search parameters.
                - Prioritize finding relevant, up-to-date information.
                - Only conclude a search after exhausting all available options.

                If a query is unclear or lacks sufficient information, ask the user for clarification.
                Provide responses that are clear, concise, and directly address the user's needs.
                When you are uncertain, it's better to inform the user that you're unable to find the specific information rather than provide incorrect details.
                

                Current user information:
                {user_info}

                Current time: 
                {time}.""",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(
        time=datetime.now()
    )  # we use .partial method to fill the prompt partially with some variables. in this case we fill the prompt with current time

    zero_shot_agent_tools = all_tools()

    # runnable taht will be wrapped by the Assistant class
    llm_with_tools = LLM.bind_tools(zero_shot_agent_tools)
    runnable = prompt | llm_with_tools

    return runnable
