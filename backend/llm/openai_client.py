from openai import AsyncOpenAI
from .base import LLMClient, LLMResponse
from config import settings, validate_openai_key
from typing import Optional

class OpenAIClient(LLMClient):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Validate API key at initialization
        api_key = validate_openai_key()
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            stream=False
        )
        
        content = completion.choices[0].message.content
        
        return LLMResponse(
            content=content,
            raw_response=completion,
            model_name=self.model_name,
            usage={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        )
