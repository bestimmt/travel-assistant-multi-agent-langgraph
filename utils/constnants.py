import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PYTHONPATH = os.getenv("PYTHONPATH")  # /filepath/to/PROJECT_ROOT_DIRECTORY
DB = PYTHONPATH + "database/travel2.sqlite"
BACKUP = PYTHONPATH + "database/travel2.backup.sqlite"
VECTOR_DB = PYTHONPATH + "database/chroma_langchain_db"
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=1)
DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
FAQ_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
