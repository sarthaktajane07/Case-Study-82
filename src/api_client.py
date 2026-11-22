import time
import json
import logging
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NimbusApiClient")

class NimbusApiClient:
    """
    Client for interacting with LLM APIs (Ollama OpenAI-compatible or direct Ollama API).
    Implements robust retry with exponential backoff, timeout handling, and structured telemetry.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        default_model: str = "gemma2:2b",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def generate_bulletin(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 160,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calls chat completions with exponential backoff and returns structured execution metadata.
        """
        model_to_use = model or self.default_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload_params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload_params["stop"] = stop
        if seed is not None:
            payload_params["seed"] = seed
        if extra_body:
            payload_params["extra_body"] = extra_body

        attempt = 0
        start_time = time.time()
        last_exception = None

        while attempt < self.max_retries:
            attempt += 1
            call_start = time.time()
            try:
                response = self.client.chat.completions.create(**payload_params)
                latency_ms = round((time.time() - call_start) * 1000, 2)

                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason

                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return {
                    "success": True,
                    "attempts": attempt,
                    "latency_ms": latency_ms,
                    "model": model_to_use,
                    "request_parameters": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "seed": seed,
                    },
                    "prompt": user_prompt,
                    "response_text": content,
                    "finish_reason": finish_reason,
                    "token_usage": usage_dict,
                    "error": None,
                }

            except Exception as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {e}. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        total_latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "success": False,
            "attempts": attempt,
            "latency_ms": total_latency_ms,
            "model": model_to_use,
            "request_parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop,
                "seed": seed,
            },
            "prompt": user_prompt,
            "response_text": "",
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": str(last_exception),
        }
