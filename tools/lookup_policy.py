from utils.prepare_database import DataPreparer
from langchain_core.tools import tool


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain operations are permitted. \
    Use this before making any flight changes performing other 'write' events."""

    retriever = DataPreparer().start_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])
