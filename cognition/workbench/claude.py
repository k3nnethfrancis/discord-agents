import os
from dotenv import load_dotenv
import anthropic
from tools.bing_search import BingSearchAPI
import json
import argparse

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bing_search = BingSearchAPI()
MODEL_NAME = "claude-3-5-sonnet-20240620"

tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        }
    }
]

def perform_bing_search(query, count=3):
    results = bing_search.search(query, count)
    formatted_results = []
    for result in results:
        formatted_result = f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content']}\n"
        formatted_results.append(formatted_result)
    return "\n".join(formatted_results)

def process_tool_call(tool_name, tool_input):
    if tool_name == "web_search":
        return perform_bing_search(tool_input["query"])

def chatbot_interaction(user_message, debug=False):
    if debug:
        print(f"\n{'='*50}\nUser Message: {user_message}\n{'='*50}")

    messages = [
        {"role": "user", "content": user_message}
    ]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        temperature=0.7,
        tools=tools,
        messages=messages
    )

    if debug:
        print(f"\nInitial Response:")
        print(f"Stop Reason: {response.stop_reason}")
        print(f"Content: {response.content}")

    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        if debug:
            print(f"\nTool Used: {tool_name}")
            print(f"Tool Input:")
            print(json.dumps(tool_input, indent=2))

        tool_result = process_tool_call(tool_name, tool_input)

        if debug:
            print(f"\nTool Result:")
            print(json.dumps(tool_result, indent=2))

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result),
                    }
                ],
            },
        ]

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            temperature=0.7,
            tools=tools,
            messages=messages
        )

        if debug:
            print(f"\nResponse:")
            print(f"Stop Reason: {response.stop_reason}")
            print(f"Content: {response.content}")

    final_response = next(
        (block.text for block in response.content if block.type == "text"),
        None,
    )

    if debug:
        print(f"\nFinal Response: {final_response}")
        print(f"\nCurrent Conversation History:")
        print(json.dumps(messages, indent=2))

    return final_response

def main(debug=False):
    print("Chat with Claude (type 'exit' to end the conversation)")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot_interaction(user_input, debug)
        if not debug:
            print(f"Claude: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Claude")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    main(debug=args.debug)
