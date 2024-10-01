"""
Module for testing the chat engine
"""

import asyncio
from cognition.engines.chat_engine import ChatEngine, Conversation
# from cognition.llms.openai import OpenAIModel
# from cognition.llms.huggingface import LocalModel
from cognition.llms.ollama import OllamaModel

system_prompt = "You are a large language model"

conversation = Conversation()
conversation.add_message({"role": "system", "content": system_prompt})

# llm = OpenAIModel()
# llm = LocalModel()
llm = OllamaModel()

chat_engine = ChatEngine(llm=llm, conversation=conversation)

# Test prompts
test_prompts = [
    "What is the capital of France?",
    "What is the population of Paris?",
    "Can you tell me a fun fact about the Eiffel Tower?",
    "What is the first question I asked you?"
]

for prompt in test_prompts:
    print(f"User: {prompt}")
    response = chat_engine.chat(prompt)
    print(f"Assistant: {response}\n\n")

# Uncomment the following lines to test async functionality
# async def test_chat_engine():
#     async for response in chat_engine._arun():
#         print(response)
#         return response  # Return the first response

# async def main():
#     response = await test_chat_engine()
#     print(f"Response: {response}")

# if __name__ == "__main__":
#     asyncio.run(main())