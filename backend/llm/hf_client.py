from huggingface_hub import InferenceClient
from .base import LLMClient, LLMResponse
from config import settings, validate_hf_token
from typing import Optional

class HFClient(LLMClient):
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_id = model_id
        
        # Validate token at initialization
        token = validate_hf_token()
        
        # The SDK reads the token and handles headers internally
        self.client = InferenceClient(model=model_id, token=token)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Using the OpenAI-compatible chat interface
        completion = self.client.chat.completions.create(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            stream=False
        )

        content = completion.choices[0].message.content
        
        return LLMResponse(
            content=content,
            raw_response=completion,
            model_name=self.model_id,
            usage={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        )