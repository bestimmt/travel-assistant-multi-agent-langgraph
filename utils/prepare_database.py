import os
import shutil
import sqlite3
import pandas as pd
import requests
import re
from typing import Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils.constnants import DB_URL, FAQ_URL, DB, BACKUP, VECTOR_DB


class DataPreparer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, message: str) -> None:
        """Helper method for logging if verbose is True."""
        if self.verbose:
            print(message)

    def download_databases(
        self,
        local_file: str = DB,
        backup_file: str = BACKUP,
        db_url: str = DB_URL,
        overwrite: bool = False,
    ) -> None:
        if overwrite or not os.path.exists(local_file):
            self.log("Trying to request and download database from URL...")
            response = requests.get(db_url)
            response.raise_for_status()  # Ensure the request was successful
            with open(local_file, "wb") as f:
                f.write(response.content)
            self.log(f"DB saved to {local_file}")
            # Backup - we will use this to "reset" our DB in each section
            shutil.copy(local_file, backup_file)
            self.log(f"DB copied to {backup_file}")
        else:
            self.log(f"DB already exists at {local_file}. Skipping download...")

    def update_timestamps(self, local_file: str = DB) -> None:
        conn = sqlite3.connect(local_file)
        self.log(f"DB connection established to {local_file}")
        cursor = conn.cursor()

        sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
        tables = pd.read_sql(sql_query, conn)["name"].to_list()

        table_dataframes = {}
        for table in tables:
            table_dataframes[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        latest_flight_time = pd.to_datetime(
            table_dataframes["flights"]["actual_departure"].replace("\\N", pd.NaT)
        ).max()

        current_time = pd.to_datetime("now").tz_localize(latest_flight_time.tz)

        time_diff = current_time - latest_flight_time

        table_dataframes["bookings"]["book_date"] = (
            pd.to_datetime(
                table_dataframes["bookings"]["book_date"].replace("\\N", pd.NaT),
                utc=True,
            )
            + time_diff
        )

        self.log(f"Bookings table, timestamps shifted by {time_diff} days")

        datetime_columns = [
            "scheduled_departure",
            "scheduled_arrival",
            "actual_departure",
            "actual_arrival",
        ]
        for column in datetime_columns:
            table_dataframes["flights"][column] = (
                pd.to_datetime(
                    table_dataframes["flights"][column].replace("\\N", pd.NaT)
                )
                + time_diff
            )
        self.log(f"Flights table, timestamps shifted by {time_diff} days")

        for table_name, table_df in table_dataframes.items():
            table_df.to_sql(table_name, conn, if_exists="replace", index=False)

        self.log(f"All timestamp changes overwritten to DB: {local_file}")
        conn.commit()
        conn.close()

    def create_faq_documents(self, faq_doc_url: str = FAQ_URL) -> list[Document]:
        """Custom text splitter for the FAQ document."""
        response = requests.get(faq_doc_url)
        response.raise_for_status()
        faq_text = response.text

        docs = [
            Document(page_content=txt.strip())
            for txt in re.split(r"(?=\n##)", faq_text)
        ]
        return docs

    def create_vectorstore(
        self,
        faq_doc_url: str = FAQ_URL,
        directory: str = VECTOR_DB,
        overwrite: bool = False,
    ) -> Chroma:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        collection_name = "faq_vectors"

        if os.path.exists(directory) and not overwrite:
            # Load the existing vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=directory,
            )
            self.log("Existing vectorstore loaded successfully with type: VectorStore")
        else:
            # Create a new vector store from documents and persist it
            vectorstore = Chroma.from_documents(
                documents=self.create_faq_documents(faq_doc_url),
                persist_directory=directory,
                embedding=embedding_model,
                collection_name=collection_name,
            )
            self.log("Vector database created from the documents successfully!")

        return vectorstore

    def start_retriever(
        self,
        faq_doc_url: str = FAQ_URL,
        directory: str = VECTOR_DB,
        overwrite: bool = False,
        k: int = 2,
    ):
        vectorstore = self.create_vectorstore(faq_doc_url, directory, overwrite)
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        self.log(f"Retriever instantiated = 'similarity: {k}'")
        return retriever

    def prepare_all(
        self,
        db_url: str = DB_URL,
        local_file: str = DB,
        backup_file: str = BACKUP,
        faq_doc_url: str = FAQ_URL,
        v_db_file: str = VECTOR_DB,
    ) -> None:
        self.download_databases(local_file, backup_file, db_url, overwrite=False)
        self.update_timestamps(local_file)
        retriever = self.start_retriever(faq_doc_url, v_db_file, overwrite=False)
        self.log("All preparation steps completed successfully.")


# Example usage:
if __name__ == "__main__":
    preparer = DataPreparer(verbose=True)
    preparer.prepare_all()
