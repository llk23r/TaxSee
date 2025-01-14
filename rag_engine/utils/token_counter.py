import os
from typing import Union, List, Dict, Any
from pathlib import Path
import logging

import google.generativeai as genai
from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = GenerativeModel(model_name)
        self._setup_model_limits()

    def _setup_model_limits(self) -> None:
        try:
            model_info = genai.get_model(f"models/{self.model.model_name}")
            self.input_token_limit = model_info.input_token_limit
            self.output_token_limit = model_info.output_token_limit
            self.context_window = self.input_token_limit + self.output_token_limit
            logger.info(
                f"Model limits - Input: {self.input_token_limit}, "
                f"Output: {self.output_token_limit}, "
                f"Context Window: {self.context_window}"
            )
        except Exception as e:
            logger.warning(f"Failed to get model limits: {e}")
            self.input_token_limit = 30720
            self.output_token_limit = 2048
            self.context_window = 32768

    def count_tokens(
        self, 
        content: Union[str, List[Union[str, Path, Dict[str, Any]]]],
        system_prompt: str = None,
        tools: List[Any] = None
    ) -> Dict[str, int]:
        try:
            content_tokens = self.model.count_tokens(content).total_tokens
            
            system_tokens = 0
            if system_prompt:
                system_tokens = self.model.count_tokens(system_prompt).total_tokens
            
            tools_tokens = 0
            if tools:
                temp_model = GenerativeModel(
                    self.model.model_name,
                    tools=tools
                )
                tools_tokens = (
                    temp_model.count_tokens("test").total_tokens - 
                    self.model.count_tokens("test").total_tokens
                )
            
            total_tokens = content_tokens + system_tokens + tools_tokens
            available_tokens = self.input_token_limit - total_tokens
            
            return {
                'content_tokens': content_tokens,
                'system_tokens': system_tokens,
                'tools_tokens': tools_tokens,
                'total_tokens': total_tokens,
                'input_token_limit': self.input_token_limit,
                'available_tokens': available_tokens
            }
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise

    def will_fit_in_context(
        self,
        content: Union[str, List[Union[str, Path, Dict[str, Any]]]],
        system_prompt: str = None,
        tools: List[Any] = None,
        buffer_tokens: int = 100
    ) -> bool:
        try:
            counts = self.count_tokens(content, system_prompt, tools)
            return counts['total_tokens'] + buffer_tokens <= self.input_token_limit
        except Exception as e:
            logger.error(f"Error checking context fit: {e}")
            return False

    def estimate_output_tokens(self, input_tokens: int, ratio: float = 1.5) -> int:
        estimated = int(input_tokens * ratio)
        return min(estimated, self.output_token_limit) 