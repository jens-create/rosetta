import hashlib
import time
import typing as typ
from pathlib import Path

import pydantic
from clients.base import BaseClient
from clients.vllm.client import VllmAPI
from components.models import LoaderConfig

from experiments.medicalqa.agents import Agent, AnswerQuestionStructuredAgent, ReActAgent
from experiments.medicalqa.agentsTools import (  # , WikipediaSummarizeAgent
    BaseAgentTools,
    WikipediaAgent,
    WikipediaCoTAgent,
    WikipediaDirectAgent,
)
from experiments.medicalqa.metrics import Accuracy, Id, Metric, ReActEvaluation, validFunctionCalling, validity
from experiments.medicalqa.responses import McQuestionAnswering


class System(pydantic.BaseModel):
    """System configuration."""

    cache_dir: str | Path
    result_dir: str | Path


class Local(System):
    """Local configuration"""

    cache_dir: str | Path = "./scratch/rosetta-cache"
    result_dir: str | Path = "./scratch/rosetta-results"


class Titans(System):
    """Titans configuration."""

    cache_dir: str | Path = "/scratch/s183568/rosetta-cache"
    result_dir: str | Path = "/scratch/s183568/rosetta-results"


class Meluxina(System):
    """Meluxina configuration."""

    cache_dir: str | Path = "/project/scratch/p200149/rosetta-cache"
    result_dir: str | Path = "/project/scratch/p200149/rosetta-results"


class MedMcQaLoader(LoaderConfig):
    """Multilingu QG loader configuration."""

    name_or_path: str = "medmcqa"
    split: str = "validation"
    num_samples: int = 1000
    seed: int = 42
    cache_dir: str | Path = "./"
    batch_size: int = 10
    num_proc: int = 128


class USMLELoader(LoaderConfig):
    """Multilingu QG loader configuration."""

    name_or_path: str = "GBaker/MedQA-USMLE-4-options"
    split: str = "test"
    # num_samples: int = 1272
    seed: int = 42
    cache_dir: str | Path = "./"
    batch_size: int = 10
    num_proc: int = 128


class Experiment(pydantic.BaseModel):
    """Experiment configuration."""

    model: str
    system: System
    dataset: LoaderConfig
    client: VllmAPI
    agent: typ.Type[Agent]
    metric: typ.Optional[Metric]
    prompt_type: int
    summarize: bool = False
    sumfilter: bool = False
    two_step: bool = False

    class Config:
        """Model configuration."""

        arbitrary_types_allowed = True

    @classmethod
    def from_config(
        cls,
        model: str,
        system: System,
        dataset: LoaderConfig,
        client: VllmAPI,
        agent: typ.Type[Agent],
        prompt_type: int,
        summarize: bool = False,
        sumfilter: bool = False,
        two_step: bool = False,
    ) -> "Experiment":
        metrics = [Id(), Accuracy()]

        if agent in [ReActAgent]:
            metrics.append(ReActEvaluation())  # type: ignore

        if agent in [AnswerQuestionStructuredAgent, BaseAgentTools]:
            metrics.append(validFunctionCalling())
        if agent in [
            WikipediaAgent,
            WikipediaCoTAgent,
            WikipediaDirectAgent,
        ]:
            metrics.append(validity())

        metric = Metric.from_config(
            metrics=metrics, output_model=McQuestionAnswering, result_dir=system.result_dir  # type: ignore
        )

        return cls(
            model=model,
            system=system,
            dataset=dataset,
            client=client,
            agent=agent,
            metric=metric,
            prompt_type=prompt_type,
            summarize=summarize,
            sumfilter=sumfilter,
            two_step=two_step,
        )

    @pydantic.model_validator(mode="after")
    def setup_dirs(self) -> "Experiment":
        """Fix and create output and cache directories.
        Result directory is for results.
        Cache directory (./datasets) is for datasets.
        Cache directory (./{{experiment_identifier}}) is for generated completions.
        """
        # Result directory
        if self.dataset.name_or_path == "medmcqa":
            self.system.result_dir = setup_output_dir(Path(self.system.result_dir) / "medmcqa")
        if self.dataset.name_or_path == "GBaker/MedQA-USMLE-4-options":
            self.system.result_dir = setup_output_dir(Path(self.system.result_dir) / "USMLE")

        # dataset cache directory
        self.dataset.cache_dir = setup_output_dir(Path(self.system.cache_dir) / "datasets")

        # experiment cache directory. This is where the generated completions are stored.
        # Cache directory is a function of a subset of the experiment configuration.
        # This ensures that the cache directory is unique for each experiment.
        identifier = hashlib.sha256(str(self.client.checkpoint).encode()).hexdigest()
        self.client.cache_dir = setup_output_dir(
            Path(self.system.cache_dir) / "completion_cache" / self.dataset.name_or_path / identifier
        )

        # init callable for client with updated cache_dir
        BaseClient.__init__(
            self=self.client,
            fn=self.client.fn,
            cache_dir=self.client.cache_dir,
            cache_reset=self.client.cache_reset,
        )

        identifier_with_messages = hashlib.sha256(
            (str(self.client.checkpoint) + str(self.agent.__name__)).encode()
        ).hexdigest()
        print(f"Experiment identifier: {identifier_with_messages}")
        if self.metric:
            self.metric.result_dir = setup_output_dir(
                self.system.result_dir  # type: ignore
                / self.model
                / self.agent.__name__
                / f"Prompt {self.prompt_type}"
                / f"Summarize {self.summarize}"
                / f"Two-step {self.two_step}"
                / f"Sumfilter {self.sumfilter}"
                / time.strftime("%d-%m-%y")
                / identifier_with_messages
            )
            # save the experiment configuration
            saveable_configs = [
                self.dataset.model_dump(exclude={"cache_dir"}),
                {"agent": str(self.agent.__name__)},
                {"checkpoint": self.client.checkpoint},
                {"prompt_type": self.prompt_type},
            ]
            # saveable_configs += [m.model_dump() for m in self.messages]

            self.metric.save_config(saveable_configs)

        return self


def setup_output_dir(dir_: Path) -> Path:
    """Setup the specified directory."""
    dir_ = dir_.resolve()
    # Create the directory if it doesn't exist
    dir_.mkdir(parents=True, exist_ok=True)

    return dir_
