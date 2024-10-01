import settings
import json
from cognition.engines.conversation import Conversation
from cognition.models.chat_models import ChatMessage

class ChatEngine:
    def __init__(self, llm, conversation: Conversation):
        self.llm = llm
        self.conversation = conversation

    def chat(self, message: str):
        self.conversation.add_message(ChatMessage(role="user", content=message))
        response = self.llm.generate(self.conversation.conversation_history.messages)
        # Ensure response is a string
        response_content = response if isinstance(response, str) else response['content']
        self.conversation.add_message(ChatMessage(role="assistant", content=response_content))
        return response_content

    async def _arun(self):
        accumulated_arguments = ""  # Accumulate JSON string parts
        current_tool_call_id = None
        current_tool_call_name = None
        non_tool_call_accumulated_response = ""

        async for chunk in self.llm._arun(self.conversation.conversation_history.messages):
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    current_tool_call_id = tool_call.id or current_tool_call_id
                    current_tool_call_name = tool_call.function.name or current_tool_call_name
                    accumulated_arguments += tool_call.function.arguments

                    try:
                        json_args = json.loads(accumulated_arguments)
                        function_result = await self.llm._afunction_call(current_tool_call_name, json_args)

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
                        await self.conversation.async_add_message(
                            ChatMessage(
                                role="tool",
                                name=current_tool_call_name,
                                content=str(function_result)
                            )
                        )

                        non_tool_call_accumulated_response = ""
                        async for new_chunk in self.llm._arun(self.conversation.conversation_history.messages):
                            new_choice = new_chunk.choices[0]
                            new_delta = new_choice.delta
                            content = new_delta.content

                            if content is not None:
                                yield new_delta.content
                                non_tool_call_accumulated_response += new_chunk.choices[0].delta.content

                    except json.JSONDecodeError:
                        print('Incomplete JSON string, waiting for more data...')

            else:
                if delta.content:
                    non_tool_call_accumulated_response += delta.content or ''
                    yield delta.content

        if non_tool_call_accumulated_response:
            await self.conversation.async_add_message(
                ChatMessage(
                    role="assistant",
                    content=str(non_tool_call_accumulated_response)
                )
            )