import google.generativeai as genai
from .base import LLMClient, LLMResponse
from config import settings

class GeminiClient(LLMClient):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = await self.model.generate_content_async(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
            )
        )
        
        return LLMResponse(
            content=response.text,
            raw_response=str(response),
            model_name=self.model_name
        )