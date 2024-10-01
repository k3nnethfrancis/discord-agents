# settings.py
from dotenv import load_dotenv
import os

# Load environment variables from the .env file.
load_dotenv(override=False)

# API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")
DISCORD_PERMISSION_INTEGER = os.getenv("DISCORD_PERMISSION_INTEGER")