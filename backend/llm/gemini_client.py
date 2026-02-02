import google.genai as genai
from .base import LLMClient, LLMResponse
from config import settings, validate_gemini_key
from typing import Optional

class GeminiClient(LLMClient):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        # Validate API key at initialization
        api_key = validate_gemini_key()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = await self.model.generate_content_async(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
                temperature=kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            )
        )
        
        return LLMResponse(
            content=response.text,
            raw_response=str(response),
            model_name=self.model_name
        )