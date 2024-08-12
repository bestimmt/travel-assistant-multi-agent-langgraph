# Digital `Travel Assistant` 🤖 with LangGraph

This repository contains a modular and compact implementation of an end-to-end multi-agent architecture for a travel assistant chatbot, inspired by the [LangGraph's Travel Assistant project](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/). The original notebook has been transformed into a fully-fledged chatbot application using Object-Oriented Programming (OOP) principles such as abstraction and encapsulation.

## Chatbot Versions

- **Version 1**: A simpler version equipped with 18 tools.
- **Version 2**: Introduces Human-in-the-loop functionality and adds a data validation step in the `GraphState`.
- **Version 3**: Tools are divided into two groups—sensitive and safe. The flow is routed accordingly, with human approval required for sensitive tasks.
- **Version 4**: A multi-agent architecture featuring separate specialized agents for different tasks and an orchestrator to manage the workflow.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fatih-ml/travel-assistant-multi-agent-langgraph.git
   ```
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your environment**:
   - Prepare your API keys in a `.env` file.
   - Configure your file paths and LLM settings in `.env` and `utils/constnants.py`.
   - you can use PYTHONPATH for easing imports
4. **Set up the database**:
   ```bash
   python utils/prepare_database.py
   ```
   This will initialize the SQL database and the Chroma VectorDatabase.
5. **Run any chatbot version**:
   ```bash
   python chatbotV04_multiagents/chatbot.py
   ```
6. **Experiment and Evaluate**:
   - Ask different questions in conversations.
   - Experiment with different LLMs and temperature settings.
   - Try out different vector databases and retrieval methods.
   - Modify and test different prompts.

## Dependencies: API Keys

Ensure that the following API keys are stored in your `.env` file:

- **OpenAI API KEY**
- **Tavily API KEY**
- **LangChain API KEY** (for tracing with LangSmith) [optional]

## Contributing

Feel free to contribute by opening issues, submitting pull requests, or asking questions. Your feedback is invaluable!

Thank you for visiting and exploring the project!
