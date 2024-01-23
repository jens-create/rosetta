from components.models import CompletionConfig
import pydantic


class VllmChatRequest(CompletionConfig):
    """Vllm request."""

    prompt: str = pydantic.Field(..., description="The prompt to send to the model.")
