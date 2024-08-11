from langchain_openai import ChatOpenAI


LLM = ChatOpenAI(model="gpt-4o-mini", temperature=1)
DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
FAQ_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
DB = r"C:\Users\fcali\OneDrive\Masaüstü\DATA_SCIENCE\LLMs\TRAVEL_ASSISTANT_MULTI_AGENT\database\travel2.sqlite"
BACKUP = r"C:\Users\fcali\OneDrive\Masaüstü\DATA_SCIENCE\LLMs\TRAVEL_ASSISTANT_MULTI_AGENT\database\travel2.backup.sqlite"
VECTOR_DB = r"C:\Users\fcali\OneDrive\Masaüstü\DATA_SCIENCE\LLMs\TRAVEL_ASSISTANT_MULTI_AGENT\database\chroma_langchain_db"
