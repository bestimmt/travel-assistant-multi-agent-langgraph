from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from chatbotV01.state import State
from utils.constnants import LLM
from tools.helpers import all_tools


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
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

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant for Swiss Airlines. "
                " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                " If a search comes up empty, expand your search before giving up."
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(
        time=datetime.now()
    )  # we use .partial method to fill the prompt partially with some variables. in this case we fill the prompt with current time

    zero_shot_agent_tools = all_tools()

    # runnable taht will be wrapped by the Assistant class
    runnable = primary_assistant_prompt | LLM.bind_tools(zero_shot_agent_tools)

    return runnable
