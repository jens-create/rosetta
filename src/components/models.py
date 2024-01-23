import os
import pathlib
import re
import typing as typ
from pathlib import Path

import pydantic
import yaml

ROSETTA_CACHE_DIR = str(pathlib.Path(os.environ.get("ROSETTA_CACHE_DIR", "~/.cache/rosetta")).expanduser())


class YamlConfig(pydantic.BaseModel):
    """Static model."""

    class Config:
        """Model configuration."""

        arbitrary_types_allowed = True
        # frozen = True
        # extra = "forbid"

    path: str | None = pydantic.Field(default=None, description="Path to yaml config file.")

    @pydantic.model_validator(mode="before")
    @classmethod
    def load_and_validate_yaml(cls, values: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Load and validate configuration from a YAML file if the path is provided."""
        config_path = values.get("path")
        if config_path is None:
            return values

        path = Path(config_path)
        if not path.is_file() and path.suffix != ".yaml":
            raise ValueError("Path must be a valid .yaml file")

        with open(path, "r") as file:
            return yaml.safe_load(file)


class LoaderConfig(YamlConfig):
    """Dataset loader."""

    name_or_path: str = pydantic.Field(..., description="dataset path")
    split: str = pydantic.Field(..., description="dataset split")
    subset: str | None = pydantic.Field(default=None, description="dataset subset")
    num_samples: int | None = pydantic.Field(default=None, description="dataset size")
    seed: int | None = pydantic.Field(default=42, description="dataset seed")
    batch_size: int | None = pydantic.Field(default=10, description="dataset batch size")
    num_proc: int | None = pydantic.Field(default=4, description="dataset number of processes")
    cache_dir: str | Path = pydantic.Field(default=ROSETTA_CACHE_DIR, description="dataset cache directory")


class EvaluatorConfig(YamlConfig):
    """Evaluation configuration."""

    metrics: list[str] = pydantic.Field(..., description="metrics to supported by HuggingFace evaluate.")
    prediction_field: str = pydantic.Field(..., description="list of predictions to evaluate.")
    reference_field: str | None = pydantic.Field(
        None,
        description="list of references for each prediction or a list of several references per prediction.",
    )
    pipe: str | None = pydantic.Field(
        default=None, description="pipeline to use for transforming data before evaluation."
    )


class ModelConfig(YamlConfig):
    """Class representing a model."""

    name: str = pydantic.Field(..., description="The name of the model.")
    endpoint: str = pydantic.Field(default="localhost", description="The endpoint of the model.")
    reset_cache: bool = pydantic.Field(default=False, description="Whether to reset the cache of the model.")

    @pydantic.field_validator("endpoint")
    def validate_endpoint(cls, v: str) -> str:
        """Validate the endpoint."""
        if not re.match(r"^[a-zA-Z0-9.-]+:\d+$", v):
            raise ValueError("Endpoint must be in the format 'host:port'")
        return v


class ChatMessage(pydantic.BaseModel):
    """Chat message."""

    role: typ.Literal["user", "assistant", "system"] = pydantic.Field(..., description="message role")
    content: str = pydantic.Field(..., description="message content")


class PromptConfig(YamlConfig):
    """Prompt configuration."""

    messages: list[ChatMessage] = pydantic.Field(..., description="messages to send to the model.")

    def format(self) -> str:
        """Format the prompt."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


class System(pydantic.BaseModel):
    """System configuration."""

    cache_dir: str = pydantic.Field(default=ROSETTA_CACHE_DIR, description="cache directory")
    batch_size: int = pydantic.Field(default=10, description="dataset batch size")
    num_proc: int = pydantic.Field(default=4, description="dataset number of processes")


class CompletionConfig(pydantic.BaseModel):
    """Completion configuration."""

    n: int = pydantic.Field(1, description="number of samples to generate")
    temperature: float = pydantic.Field(0.0, description="temperature")
    max_tokens: int = pydantic.Field(512, description="maximum number of tokens to generate")
    top_p: float = pydantic.Field(1, description="top p")
    presence_penalty: float = pydantic.Field(0.0, description="presence penalty")
    frequency_penalty: float = pydantic.Field(0.0, description="frequency penalty")
    stream: bool = pydantic.Field(False, description="stream")
    stop: list[str] = pydantic.Field([], description="stop tokens")
    skip_special_tokens: bool = pydantic.Field(True, description="skip special tokens")
