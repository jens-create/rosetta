import typing as typ

from clients.vllm.client import VllmAPI
from components.models import LoaderConfig

from experiments.medicalqa.agents import (
    Agent,
    AnswerQuestionStructuredAgent,
    CoTAgent,
    DirectAgent,
    FewShotCoTAgent,
    FewShotDirectAgent,
    FindZebraAgent,
    RavenAgent,
)
from experiments.medicalqa.metrics import Accuracy, Metric, ReActEvaluation, ToolEvaluation, validFunctionCalling
from experiments.medicalqa.models import Experiment, System
from experiments.medicalqa.prompt import PromptConfig, QuestionAnswer, Shot, get_templates
from experiments.medicalqa.responses import McQuestionAnswering
from experiments.medicalqa.shots import get_fixed_shots


class ReAct(Experiment):
    """ReAct method"""

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client: VllmAPI, agent: typ.Type[Agent]) -> "ReAct":
        """Create an experiment from a configuration."""
        # instantiate metric class
        metric = Metric.from_config([Accuracy(), ReActEvaluation()], McQuestionAnswering, sys.result_dir)  # type: ignore

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=agent,
            metric=metric,
        )


class Direct(Experiment):
    """ReAct method"""

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "Direct":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "mistralai/Mistral-7B-v0.1",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy()], McQuestionAnswering, sys.result_dir)  # type: ignore

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=DirectAgent,
            metric=metric,
        )


class CoT(Experiment):
    """ReAct method"""

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "CoT":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "mistralai/Mistral-7B-v0.1",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy()], McQuestionAnswering, sys.result_dir)  # type: ignore

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=CoTAgent,
            metric=metric,
        )


class FewShotCoT(Experiment):
    """ReAct method"""

    # {'accuracy': 0.495}

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "FewShotCoT":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "mistralai/Mistral-7B-v0.1",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy()], McQuestionAnswering, sys.result_dir)  # type: ignore

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=FewShotCoTAgent,
            metric=metric,
        )


class FewShotDirect(Experiment):
    """ReAct method"""

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "FewShotDirect":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "mistralai/Mistral-7B-v0.1",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy()], McQuestionAnswering, sys.result_dir)  # type: ignore

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=FewShotDirectAgent,
            metric=metric,
        )


class LLama7FoundationZeroShotExperiment(Experiment):
    """An experiment for the Llama2-7b foundation model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama7FoundationZeroShotExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("meta-llama/Llama-2-7b-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLama7ChatZeroShotExperiment(Experiment):
    """An experiment for the Llama2-7b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    {'accuracy': 0.32, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.89, 'validQuestionAnswer': 0.89}} @100 samples.
    --> {'accuracy': 0.346, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.903, 'validQuestionAnswer': 0.887}}

    prompt function = {'accuracy': 0.243, 'validFunctionCalling': {'validFunction': 0.712, 'validJSON': 0.667, 'validQuestionAnswer': 0.656}}
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama7ChatZeroShotExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("meta-llama/Llama-2-7b-chat-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLama7ChatZeroShotFindZebraExperiment(Experiment):
    """An experiment for the Llama2-7b chat model.
    This experiment uses the FindZebraAgent.
    This experiment is zeroshot.
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama7ChatZeroShotFindZebraExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "meta-llama/Llama-2-7b-chat-hf",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=FindZebraAgent,
            metric=metric,
        )


class LLama13ChatZeroShotExperiment(Experiment):
    """An experiment for the Llama2-13b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    Old: approach... {'accuracy': 0.372, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.98, 'validQuestionAnswer': 0.975}}.
    {'accuracy': 0.35, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.98, 'validQuestionAnswer': 0.97}} @100 samples.
    --> {'accuracy': 0.377, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.974, 'validQuestionAnswer': 0.955}}.

    prompt function = 0,0 ,0? weird but true
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama13ChatZeroShotExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("meta-llama/Llama-2-13b-chat-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLama70ChatZeroShotExperiment(Experiment):
    """An experiment for the Llama2-13b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    --> {'accuracy': 0.346, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.786, 'validQuestionAnswer': 0.782}}
    Can not produce valid json...
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama70ChatZeroShotExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("meta-llama/Llama-2-70b-chat-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLamaCode7InstructZeroShotExperiment(Experiment):
    """An experiment for the LlamaCode 7b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    --> {'accuracy': 0.327, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.982, 'validQuestionAnswer': 0.98}}.
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]):
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("codellama/CodeLlama-7b-Instruct-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLamaCode13InstructZeroShotExperiment(Experiment):
    """An experiment for the LlamaCode 13b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    --> {'accuracy': 0.33, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.975, 'validQuestionAnswer': 0.973}}.
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]):
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("codellama/CodeLlama-13b-Instruct-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLamaCode34InstructZeroShotExperiment(Experiment):
    """An experiment for the LlamaCode 34b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is zeroshot.
    -->
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]):
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("codellama/CodeLlama-34b-Instruct-hf")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class Mistral7bInstruct(Experiment):
    """This experiment is zeroshot.
    #{'accuracy': 0.274, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.701, 'validQuestionAnswer': 0.682}}.
    prompt1 functions.Quesiton... {'accuracy': 0.291, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.718, 'validQuestionAnswer': 0.697}}.
    prompt2 funciton = {..} {'accuracy': 0.394, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.971, 'validQuestionAnswer': 0.947}}
    prompt2 updated {'accuracy': 0.419, 'validFunctionCalling': {'validFunction': 1.0, 'validJSON': 0.979, 'validQuestionAnswer': 0.959}}.
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]):
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("mistralai/Mistral-7B-Instruct-v0.1")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class Mixtral8x7bInstruct(Experiment):
    """This experiment is zeroshot."""

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]):
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("mistralai/Mixtral-8x7B-Instruct-v0.1")

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLama13ChatFewShotExperiment(Experiment):
    """An experiment for the Llama2-13b chat model.
    This experiment uses the FunctionCallingAgent.
    This experiment is fewshot.
    {'accuracy': 0.435, 'validFunctionCalling': {'validFunction': 0.999, 'validJSON': 0.98, 'validQuestionAnswer': 0.976}}.
    962e2fd0795608725319916a73797c7f3da2a7f86b34852192b68d756ce00f45: {'accuracy': 0.437, 'validFunctionCalling': {'validFunction': 0.999, 'validJSON': 0.98, 'validQuestionAnswer': 0.975}}.
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama13ChatFewShotExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type("meta-llama/Llama-2-13b-chat-hf")

        # prompt_config
        system_message = """You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.
        You must respond with "function = " followed with valid JSON that follows the provided schema.
        """
        templates = get_templates("medmcqa")
        shots = [Shot(**s) for s in get_fixed_shots("medmcqa")]
        tools = None

        extract_function = QuestionAnswer
        prompt_config = PromptConfig(
            system_message=system_message,
            templates=templates,
            shots=shots,
            extract_function=extract_function,
            tools=tools,
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=AnswerQuestionStructuredAgent,
            metric=metric,
        )


class LLama13ChatZeroShotFindZebraExperiment(Experiment):
    """An experiment for the Llama2-13b chat model.
    This experiment uses the FindZebraAgent.
    This experiment is zeroshot.
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama13ChatZeroShotFindZebraExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "meta-llama/Llama-2-13b-chat-hf",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=FindZebraAgent,
            metric=metric,
        )


class LLama70ChatZeroShotFindZebraExperiment(Experiment):
    """An experiment for the Llama2-70b chat model.
    This experiment uses the FindZebraAgent.
    This experiment is zeroshot.
    """

    @classmethod
    def from_partial(
        cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]
    ) -> "LLama70ChatZeroShotFindZebraExperiment":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "meta-llama/Llama-2-70b-chat-hf",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), validFunctionCalling()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=FindZebraAgent,
            metric=metric,
        )


class Raven(Experiment):
    """An experiment for the Raven model.
    This experiment uses the FindZebraAgent.
    This experiment is zeroshot.
    {'accuracy': 0.332, 'tool_usage': {'reasoning': 0.972, 'answer': 0.972, 'medical_database_query': 0.235}}.
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "Raven":
        """Create an experiment from a configuration."""
        # client
        client: VllmAPI = client_type(
            "Nexusflow/NexusRaven-V2-13B",
        )

        # instantiate metric class
        metric = Metric.from_config([Accuracy(), ToolEvaluation()], McQuestionAnswering, sys.result_dir)  # type: ignore
        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=RavenAgent,
            metric=metric,
        )


class TwoStep(Experiment):
    """This experiment firstly asks whether the functions should be called.
    Afterwards the result of the functions are parsed to the QuestionAnswer phase.
    """

    @classmethod
    def from_partial(cls, sys: System, ds: LoaderConfig, client_type: typ.Type[VllmAPI]) -> "TwoStep":
        # client
        client: VllmAPI = client_type(
            "meta-llama/Llama-2-13b-chat-hf",
        )

        metric = None

        return cls(
            system=sys,
            dataset=ds,
            client=client,
            agent=RavenAgent,
            metric=metric,
        )
