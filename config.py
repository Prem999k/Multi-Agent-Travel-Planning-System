import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load the project's .env file.
load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


# Create the shared Groq model.
def get_llm():
    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )
    )
