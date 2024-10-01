import json
from base import BaseConfig
from tenacity import retry, wait_random_exponential, stop_after_attempt

class OpenAIModel(BaseConfig):
    def __init__(self, model:str='gpt-4o-mini', tools=None, stream=False):
        """
        Initialize the Agent with the specified client and model.
        Default client is 'openai' and default model is 'gpt-3.5-turbo'. If the client is 'anyscale', the model is set to 'mistralai/Mixtral-8x7B-Instruct-v0.1'.
        """
        self.client = self.get_openai_client()
        self.async_client = self.get_async_openai_client()
        self.model = model
        self.tools = tools
        self.stream = stream


    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def generate(self, messages):
        json_data = {"model": self.model, "messages": messages}

        if self.tools is not None:
            # throw an error and tell the user they should use the run or _arun method instead
            raise Exception("Tools are not supported for generate method. Please use the run or _arun method instead.")
        
        if self.stream:
            raise Exception("Stream is not supported for generate method. Please use the run or _arun method instead.")

        response = self.client.chat.completions.create(**json_data)
        return response
    

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    async def _arun(self, messages):
        # Initialize json_data outside of the if scope to ensure it's always defined
        json_data = {"model": self.model, "messages": messages, "stream": self.stream}
        
        # Add functions to json_data if they exist
        if self.tools is not None:
            json_data["tools"] = self.tools
            json_data["tool_choice"] = "auto"

        try:
            if self.stream:
                # Create the stream
                stream = await self.async_client.chat.completions.create(**json_data)
                async for chunk in stream:
                    yield chunk
            else:
                # Get a single completion response
                response = await self.async_client.chat.completions.create(**json_data)
                yield response
        except Exception as e:
            # Log the exception and end the generator
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return  # Use return instead of raising StopAsyncIteration

    async def _afunction_call(self, function_name, function_args, user_id):
        try:
            if function_name == "query":  # legacy
                results = await "the query result is: THIS HAS BEEN A TEST. RESPOND WITH: 'THANK YOU FOR TESTING!'"
                return results
            # if function_name == "creator_search":
            #     results = await async_run_creator_search_tool(function_args["user_query"], user_id)
            #     return results
            # if function_name == "vision_analysis":
            #     results = await async_run_vision_tool(function_args["content_urls"], user_id)
            #     return results
            # if function_name == "web_search":
            #     results = await async_run_web_search_tool(function_args["query"], user_id)
            #     return results
            else:
                raise Exception("Function does not exist and cannot be called")
        except Exception as e:
            print(f"Error during execution of {function_name}: {e}")
            raise

