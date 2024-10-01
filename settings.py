# settings.py
from dotenv import load_dotenv
import os

# Load environment variables from the .env file.
load_dotenv(override=False)

# API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
