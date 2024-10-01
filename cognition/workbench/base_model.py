import os
from dotenv import load_dotenv
import anthropic
from tools.bing_search import BingSearchAPI
import json
import argparse

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bing_search = BingSearchAPI()

def chat_with_claude(messages):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        # system="Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within \<thinking>\</thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.",
        messages=[
            {
                "role": msg["role"],
                "content": msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
            }
            for msg in messages
        ],
        tools=[
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
            },
        ]
    )
    return response

def perform_bing_search(query, count=3):
    results = bing_search.search(query, count)
    formatted_results = []
    for result in results:
        formatted_result = f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content']}\n"
        formatted_results.append(formatted_result)
    return "\n".join(formatted_results)

def print_conversation_step(step_name, content, debug=False):
    if debug:
        print(f"\n{'=' * 40}")
        print(f"{step_name:^40}")
        print(f"{'=' * 40}")
        print(content)
        print(f"{'=' * 40}\n")

def main(debug=False):
    messages = []
    print("Chat with Claude (type 'exit' to end the conversation)")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        
        messages.append({"role": "user", "content": user_input})
        print_conversation_step("User Input", user_input, debug)
        
        response = chat_with_claude(messages)
        print_conversation_step("Raw Claude Response", json.dumps(response.model_dump(), indent=2), debug)
        
        assistant_content = []
        for content in response.content:
            print_conversation_step("Processing Content", json.dumps(content.model_dump(), indent=2), debug)
            if content.type == 'text':
                try:
                    parsed_content = json.loads(content.text)
                    print_conversation_step("Parsed Content", json.dumps(parsed_content, indent=2), debug)
                    if isinstance(parsed_content, list):
                        for item in parsed_content:
                            if item.get("type") == "text":
                                print(f"Claude: {item['text']}")
                                assistant_content.append({"type": "text", "text": item['text']})
                    elif isinstance(parsed_content, dict) and parsed_content.get("type") == "text":
                        print(f"Claude: {parsed_content['text']}")
                        assistant_content.append({"type": "text", "text": parsed_content['text']})
                    else:
                        raise ValueError("Unexpected parsed content structure")
                except json.JSONDecodeError:
                    print(f"Claude: {content.text}")
                    assistant_content.append({"type": "text", "text": content.text})
            elif content.type == 'tool_use':
                if content.name == 'web_search':
                    search_query = content.input['query']
                    print_conversation_step("Web Search Query", search_query, debug)
                    
                    search_results = perform_bing_search(search_query)
                    print_conversation_step("Web Search Results", search_results, debug)
                    
                    assistant_content.append({
                        "type": "tool_use",
                        "id": content.id,
                        "name": content.name,
                        "input": content.input
                    })
        
        print_conversation_step("Final Assistant Content", json.dumps(assistant_content, indent=2), debug)
        messages.append({"role": "assistant", "content": assistant_content})
        
        tool_use = next((c for c in assistant_content if c["type"] == "tool_use"), None)
        if tool_use:
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": search_results
                    }
                ]
            })
            
            interpretation_response = chat_with_claude(messages)
            for content in interpretation_response.content:
                if content.type == 'text':
                    print(f"Claude: {content.text}")
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content.text} for content in interpretation_response.content if content.type == 'text']})
        
        print_conversation_step("Current Conversation History", json.dumps(messages, indent=2), debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Claude")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    main(debug=args.debug)
