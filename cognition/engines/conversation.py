from cognition.models.chat_models import ChatMessage, ChatHistory
from typing import Union
import settings

class Conversation:
    def __init__(self):
        self.conversation_history = ChatHistory(messages=[])

    def add_message(self, message: Union[ChatMessage, dict]):
        if isinstance(message, dict):
            message = ChatMessage(**message)
        self.conversation_history.messages.append(message)

    def process_json(self, messages):
        for index, message in enumerate(messages):
            if index == 0:
                system_message = ChatMessage(
                    role="system",
                    content=settings.ari_sys_text
                )
                self.conversation_history.messages.append(system_message)
            self.add_message(message)  # Use the updated add_message method

    async def async_add_message(self, message: Union[ChatMessage, dict]):
        if isinstance(message, dict):
            message = ChatMessage(**message)
        self.conversation_history.messages.append(message)