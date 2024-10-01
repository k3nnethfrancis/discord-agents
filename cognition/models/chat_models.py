from pydantic import BaseModel
from typing import List, Optional, Union, Dict

# create a chat message object that can have content as either a string or a dictionary
class ChatMessage(BaseModel):
    role: str
    content: Union[str, Dict]

# create a chat history object that is a list of chat messages
class ChatHistory(BaseModel):
    messages: List[ChatMessage]