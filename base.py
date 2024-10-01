# base.py
from openai import OpenAI, AsyncOpenAI
import settings
from typing import Optional


class BaseConfig:
    def __init__(self, name, log_file=None):
        # Set up logger
        # self.logger = self.setup_logger(name)

        # Load API keys
        self.OPENAI_API_KEY = settings.OPENAI_API_KEY

    def get_openai_client(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        if base_url is not None:
            return OpenAI(
                base_url=base_url,
                api_key=api_key or settings.OPENAI_API_KEY
            )
        else:
            return OpenAI(
                api_key=api_key or settings.OPENAI_API_KEY
            )  # Return the asynchronous client

    def get_async_openai_client(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        if base_url is not None:
            return AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or settings.OPENAI_API_KEY
            )
        else:
            return AsyncOpenAI(
                api_key=api_key or settings.OPENAI_API_KEY
            )  # Return the asynchronous client