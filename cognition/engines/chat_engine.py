import settings
import json
# from pydantic import BaseModel
# from typing import List, Dict, Optional
from cognition.models.chat_models import ChatMessage, ChatHistory

# # create a chat message object that is a dictionary of role and content and tools
# class ChatMessage(BaseModel):
#     role: str
#     content: str
#     name: Optional[str] = None

# # create a chat history object that is a list of chat messages (which are dictionaries of role, content, and tools)
# class ChatHistory(BaseModel):
#     messages: List[ChatMessage]

class Conversation:
    def __init__(self):
        self.conversation_history = ChatHistory(messages=[])

    def add_message(self, message: ChatMessage):
        self.conversation_history.messages.append(message)

    def process_json(self, messages):
        for index, message in enumerate(messages):
            if index == 0:
                system_message = ChatMessage(
                    role="system",
                    content=settings.ari_sys_text
                )
                self.conversation_history.messages.append(system_message)
            self.conversation_history.messages.append(message)

        # print("conversation_history:")
        # print(self.conversation_history)

    async def async_add_message(self, message_json):
        self.conversation_history.messages.append(message_json)
    
    def add_message(self, message_json):
        self.conversation_history.messages.append(message_json)


# create a chat engine class that is initialized with an llm and a conversration object
class ChatEngine:
    def __init__(self, llm, conversation: Conversation):
        self.llm = llm
        self.conversation = conversation

    def chat(self, message: str):
        self.conversation.add_message(ChatMessage(role="user", content=message))
        response = self.llm.generate(self.conversation.conversation_history.messages)
        self.conversation.add_message(ChatMessage(role="assistant", content=response))
        return response

    async def _arun(self):
        accumulated_arguments = ""  # Accumulate JSON string parts
        current_tool_call_id = None
        current_tool_call_name = None
        non_tool_call_accumulated_response = ""

        # print('Starting sequence...')
        async for chunk in self.llm._arun(self.conversation.conversation_history.messages):
            # print(f'Processing chunk: {chunk} \n---')
            choice = chunk.choices[0]
            delta = choice.delta

            # Handle tool call initialization and continuation
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    current_tool_call_id = tool_call.id or current_tool_call_id
                    current_tool_call_name = tool_call.function.name or current_tool_call_name
                    accumulated_arguments += tool_call.function.arguments

                    # Try parsing the accumulated JSON arguments
                    try:
                        json_args = json.loads(accumulated_arguments)
                        # print(f'Complete JSON arguments received: {json_args}')
                        function_result = await self.llm._afunction_call(current_tool_call_name, json_args)
                        # print(f'Function result: {function_result}')

                        # add the tool details per opeAI API requirements
                        await self.conversation.async_add_message(
                            ChatMessage(
                                role="assistant",
                                content={
                                    "tool_calls": [{
                                        "id": current_tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": current_tool_call_name,
                                            "arguments": str(json_args)
                                        }
                                    }]
                                }
                            )
                        )
                        # add the tool results
                        await self.conversation.async_add_message(
                            ChatMessage(
                                role="tool",
                                name=current_tool_call_name,
                                content=str(function_result)
                            )
                        )

                        # generate the response now that we have context
                        non_tool_call_accumulated_response = ""
                        async for new_chunk in self.llm._arun(self.conversation.conversation_history.messages):
                            # print(f'Processing new chunk after tool call: {new_chunk}')
                            # yield new_chunk
                            new_choice = new_chunk.choices[0]
                            new_delta = new_choice.delta
                            content = new_delta.content

                            if content is not None:
                                yield new_delta.content
                                non_tool_call_accumulated_response += new_chunk.choices[0].delta.content

                    except json.JSONDecodeError:
                        print('Incomplete JSON string, waiting for more data...')

            else:
                # Non-tool call content handling
                if delta.content:
                    non_tool_call_accumulated_response += delta.content or ''
                    #print('Yielding content for non-tool call:', delta.content)
                    yield delta.content

        # Final addition to the conversation if there is any accumulated content from non-tool calls
        if non_tool_call_accumulated_response:
            await self.conversation.async_add_message(
                ChatMessage(
                    role="assistant",
                    content=str(non_tool_call_accumulated_response)
                )
            )