import typing as typ

import pydantic


class ResponseFormat(pydantic.BaseModel):
    """Format of the model output."""

    type: str | pydantic.Json = pydantic.Field(
        default='{ "type": "json_object" }', description="Must be one of 'text' or 'json_object'."
    )


class Function(pydantic.BaseModel):
    """Function called by tool."""

    name: str = pydantic.Field(..., description="Name of the function.")
    parameters: pydantic.Json = pydantic.Field(..., description="Parameters for the function.")
    description: str | None = pydantic.Field(default=None, description="Description of the function.")


class Tool(pydantic.BaseModel):
    """Tool to call."""

    type: typ.Literal["function"] | None = pydantic.Field(default="function", description="Must be 'function'.")
    function: Function = pydantic.Field(..., description="Function to call.")


class ToolChoice(pydantic.BaseModel):
    """Tool choice."""

    type: typ.Literal["function"] | None = pydantic.Field(default="function", description="Must be 'function'.")
    function: list[Tool] = pydantic.Field(
        ...,
        description='Specifying a particular function via {"type: "function", "function": {"name": "my_function"}} forces the model to call that function.',
    )


class OpenAIChatRequest(pydantic.BaseModel):
    """OpenAI chat request."""

    messages: list[str] = pydantic.Field(..., description="A list of messages comprising the conversation so far.")
    model: str = pydantic.Field(..., description="ID of the model to use.")
    frequency_penalty: float | None = pydantic.Field(
        default=0, description="Penalize new tokens based on their frequency, between -2.0 and 2.0."
    )
    logit_bias: dict[int, float] | None = pydantic.Field(default=None, description="Map of token ID to bias value.")
    max_tokens: int | None = pydantic.Field(default=None, description="Maximum number of tokens to generate.")
    n: int | None = pydantic.Field(default=1, description="Number of chat completion choices to generate.")
    presence_penalty: None | float = pydantic.Field(
        default=0, description="Penalize new tokens based on their presence, between -2.0 and 2.0."
    )
    response_format: ResponseFormat | None = pydantic.Field(default=None, description="Format of the model output.")
    seed: int | None = pydantic.Field(default=None, description="Seed for deterministic results (Beta).")
    stop: typ.Union[str, list[str]] | None = pydantic.Field(
        default=None, description="Sequences where the API will stop generating tokens."
    )
    stream: bool | None = pydantic.Field(default=False, description="If true, sends partial message deltas.")
    temperature: float | None = pydantic.Field(default=1, description="Sampling temperature between 0 and 2.")
    top_p: float | None = pydantic.Field(default=1, description="Nucleus sampling with top_p probability mass.")
    tools: list[Tool] | None = pydantic.Field(default=None, description="List of tools the model may call.")
    tool_choice: typ.Union[typ.Literal["none", "auto"], ToolChoice] | None = pydantic.Field(
        default=None, description="Controls which function is called by the model."
    )
    user: str | None = pydantic.Field(default=None, description="Unique identifier for the end-user.")
