import json
import typing as typ
from pathlib import Path

import requests
from clients.base import BaseClient
from clients.vllm.models import VllmChatRequest
from components.models import ChatMessage, CompletionConfig
from transformers import AutoTokenizer


def query_vllm(prompt: str, url: str, completion_config: CompletionConfig) -> str:
    """Call the generate function.."""
    m = VllmChatRequest(prompt=prompt, **completion_config.model_dump())
    response = requests.post(url=f"{url}/generate", json=m.model_dump(exclude_none=True), timeout=600)
    response.raise_for_status()
    data = json.loads(response.content)
    return data["text"]


class VllmAPI(BaseClient):
    """Generic client for Vllm supported models."""

    checkpoint: str
    endpoint: str
    fn: typ.Callable = query_vllm
    cache_dir: str | Path = "/scratch/s183568/rosetta-cache"
    cache_reset: bool = False

    def __init__(
        self,
        checkpoint: str,
        endpoint: str = "localhost:8001",
    ) -> None:
        """Initialize client."""
        self.checkpoint = checkpoint
        self.template = AutoTokenizer.from_pretrained(checkpoint)
        self.endpoint = f"http://{endpoint}"
        self.fn = query_vllm

        if "codellama" in self.checkpoint:
            self.template = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def __call__(self, messages: str | list[ChatMessage], guidance_str: str = "", completion_config: CompletionConfig = CompletionConfig()) -> str:  # type: ignore  # noqa: B008
        """Generate a response with a vllm served model."""
        if isinstance(messages, str):
            messages_str = messages
        else:
            if self.template.chat_template is None:  # if foundation model
                messages_str: str = "\n".join([m.content for m in messages if m.role != "system"])
            elif "mistralai" in self.checkpoint:
                # move system message to user message:
                system_message = messages.pop(0)
                messages[0].content = system_message.content + "\n\n" + messages[0].content
                messages_str: str = self.template.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True  # type: ignore
                )

            else:
                messages_str: str = self.template.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True  # type: ignore
                )

        # print(messages_str)

        if guidance_str:
            messages_str = messages_str + guidance_str

        # guided_str = messages_str + ' functions.QuestionAnswer({"explanation": '
        response: str = self.fn(prompt=messages_str, url=self.endpoint, completion_config=completion_config)

        # select first element of response (we are only sampling one response)
        response = response[0]

        # print("response", response)

        # remove prompt (message_str) from generated response
        response = response.replace(messages_str, "").strip()

        return guidance_str + response
