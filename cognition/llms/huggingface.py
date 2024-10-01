import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List
from cognition.models.chat_models import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModel:
    def __init__(self, model_path: str = "././weights/meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="mps" if torch.backends.mps.is_available() else "auto",
        )
        self.model.eval()
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        logger.info(f"Initialized model from path: {model_path}")

    def _serialize_messages(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def generate(self, messages: List[ChatMessage], max_length: int = 512) -> str:
        logger.info(f"Generating response for messages: {messages}")
        try:
            if self.has_chat_template:
                return self._generate_with_chat_template(messages, max_length)
            else:
                return self._generate_without_chat_template(messages, max_length)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _generate_with_chat_template(self, messages: List[ChatMessage], max_length: int) -> str:
        serialized_messages = self._serialize_messages(messages)
        inputs = self.tokenizer.apply_chat_template(
            serialized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=True)
        full_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response
        assistant_response = full_response.split("assistant")[-1].strip()
        logger.info(f"Generated response: {assistant_response}")
        return assistant_response

    def _generate_without_chat_template(self, messages: List[ChatMessage], max_length: int) -> str:
        # Combine messages into a single string
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        prompt += "\nassistant:"  # Add a prompt for the model to continue

        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_length, temperature=0.7, do_sample=True)

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("assistant:")[-1].strip()
        
        logger.info(f"Generated response: {assistant_response}")
        return assistant_response

# Example usage
if __name__ == "__main__":
    model_path = "././weights/meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = LocalModel(model_path=model_path)
    
    messages = [
        ChatMessage(role="system", content="You are a large language model."),
        ChatMessage(role="user", content="How are you?"),
    ]
    
    start_time = time.time()
    response = llm.generate(messages)
    end_time = time.time()
    print(f"First response time taken: {end_time - start_time} seconds")
    print("Assistant:", response)

    # Simulate a conversation
    messages.append(ChatMessage(role="assistant", content=response))
    messages.append(ChatMessage(role="user", content="What's your purpose?"))
        
    start_time = time.time()
    response = llm.generate(messages)
    end_time = time.time()
    print(f"Second response time taken: {end_time - start_time} seconds")
    print("\nAssistant:", response)