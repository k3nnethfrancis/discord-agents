"""
Module for testing the chat engine
"""

import asyncio
from cognition.engines.chat_engine import ChatEngine, Conversation
from cognition.llms.openai import OpenAIModel
from cognition.llms.huggingface import LocalModel

system_prompt = "You are a large language model"

conversation = Conversation()
conversation.add_message({"role": "system", "content": system_prompt})

llm = OpenAIModel()
# llm = LocalModel()

chat_engine = ChatEngine(llm=llm, conversation=conversation)

# Test prompt
test_prompt = "What is the capital of France?"

print(chat_engine.chat(test_prompt))


# conversation.add_message({"role": "user", "content": test_prompt})

# async def test_chat_engine():
#     async for response in chat_engine._arun():
#         print(response)
#         return response  # Return the first response

# async def main():
#     response = await test_chat_engine()
#     print(f"Response: {response}")

# if __name__ == "__main__":
#     asyncio.run(main())