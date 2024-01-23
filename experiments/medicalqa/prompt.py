from enum import Enum
from typing import Type

import jinja2
import pydantic
from components.models import ChatMessage
from instructor import OpenAISchema

from experiments.medicalqa.functionary import Function, generate_schema_from_functions
from experiments.medicalqa.functions.findzebra import FindZebraAPI
from experiments.medicalqa.shots import get_fixed_shots

SYMBOLS = ["A", "B", "C", "D"]


class Shot(pydantic.BaseModel):
    """A shot is a question with four answer options and an explanation."""

    question: str = pydantic.Field(..., description="question")
    opa: str = pydantic.Field(..., description="answer option A")
    opb: str = pydantic.Field(..., description="answer option B")
    opc: str = pydantic.Field(..., description="answer option C")
    opd: str = pydantic.Field(..., description="answer option D")
    choices: list[str] = pydantic.Field(..., description="answer choices")
    target: int = pydantic.Field(..., description="target answer")
    explanation: str = pydantic.Field(..., description="explanation")


class ExtractFunction(OpenAISchema):
    """Base class for extracting functions from shots."""

    @classmethod
    def from_shot(cls, shot: Shot) -> "ExtractFunction":
        """Extract the function from the shot."""
        raise NotImplementedError


class Options(str, Enum):
    """Answer options."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class QuestionAnswerDirect(ExtractFunction):
    answer: Options = pydantic.Field(..., description="Therefore, among A through D, the answer is")

    @pydantic.validator("answer", pre=True)
    def validate_answer(cls, v) -> Options:
        """Validate the answer. If e.g. A) <opa> is chosen, then the answer is A."""
        if isinstance(v, str):
            # Extract the option part if the input is a string
            option_part = v.split(")")[0]
            if option_part in Options.__members__:
                return Options(option_part)

            # check if v is lower
            if v.upper() in Options.__members__:
                return Options(v.upper())

            raise ValueError(f"Answer option {v} not in {Options}")
        elif v not in Options:  # noqa: RET506
            raise ValueError(f"Answer {v} not in {Options}")
        return v

    @staticmethod
    def typescript() -> str:
        """Generate the typescript for the answer schema."""
        lines = [
            # "// Supported function definitions that must be used.",
            "namespace functions {",
            "",
            "// Answer the multiple choice question with the given options. One of the given options is the correct answer to the question.",
            "type QuestionAnswer = (_: {",
            "// Therefore, among A through D, the answer is",
            'answer_option: ("A" | "B" | "C" | "D"),',
            "}) => any;",
            "",
            "} // namespace functions",
        ]
        return "\n".join(lines)


class QuestionAnswer(ExtractFunction):
    """Answer the multiple choice question with the given options."""

    explanation: str = pydantic.Field(
        ...,
        description="Explanation of the answer option chosen.",
    )
    answer: Options = pydantic.Field(..., description="Therefore, among A through D, the answer is")

    @pydantic.validator("answer", pre=True)
    def validate_answer(cls, v) -> Options:
        """Validate the answer. If e.g. A) <opa> is chosen, then the answer is A."""
        if isinstance(v, str):
            # Extract the option part if the input is a string
            option_part = v.split(")")[0]
            if option_part in Options.__members__:
                return Options(option_part)

            # check if v is lower
            if v.upper() in Options.__members__:
                return Options(v.upper())

            raise ValueError(f"Answer option {v} not in {Options}")
        elif v not in Options:  # noqa: RET506
            raise ValueError(f"Answer {v} not in {Options}")
        return v

    @staticmethod
    def typescript() -> str:
        """Generate the typescript for the answer schema."""
        lines = [
            # "// Supported function definitions that must be used.",
            "namespace functions {",
            "",
            "// Answer the multiple choice question with the given options. One of the given options is the correct answer to the question.",
            "type QuestionAnswer = (_: {",
            "// Explanation of the answer option chosen.",
            "explanation: string,",
            "// Therefore, among A through D, the answer is",
            'answer: ("A" | "B" | "C" | "D"),',
            "}) => any;",
            "",
            "} // namespace functions",
        ]
        return "\n".join(lines)

    @staticmethod
    def typescript2() -> str:
        """Generate the typescript for the answer schema."""
        lines = [
            # "// Supported function definitions that must be used.",
            "// Answer the multiple choice question with the given options. One of the given options is the correct answer to the question.",
            "type QuestionAnswer = (_: {",
            "// Explanation of the answer option chosen.",
            "explanation: string,",
            "// Therefore, among A through D, the answer is",
            'answer: ("A" | "B" | "C" | "D"),',
            "}) => any;",
        ]
        return "\n".join(lines)

    @staticmethod
    def typescript3() -> str:
        """Generate the typescript for the answer schema."""
        lines = [
            # "// Supported function definitions that must be used.",
            "// Answer the multiple choice question with the given options. One of the given options is the correct answer to the question.",
            "type AnswerQuestion = (_: {",
            "// Explanation of the answer option chosen.",
            "explanation: string,",
            "// Therefore, among A through D, the answer is",
            'answer: ("A" | "B" | "C" | "D"),',
            "}) => any;",
        ]
        return "\n".join(lines)

    @classmethod
    def from_shot(cls, shot: Shot) -> "QuestionAnswer":
        """Extract the function from the shot."""
        return cls(explanation=shot.explanation, answer=Options(SYMBOLS[shot.target]))


def get_templates(dataset: str) -> dict[str, str]:
    """Get the templates from the dataset.
    Three templates needed; system, input and output.
    The input and output are tied together with shots.
    The system template is dataset specific and will be tied together with the functions.
    """
    if dataset == "medmcqa":
        return {
            "input": "Question: {{ question }}\nAnswer options:\nA) {{opa}}\nB) {{opb}}\n C) {{opc}}\n D) {{opd}}\n",
            "output": "Explanation: Let's think step by step: {{ explanation }}.\nTherefore, among A through D, the correct answer is ({{ ['A', 'B', 'C', 'D'][target] }})",
        }
    if dataset == "GBaker/MedQA-USMLE-4-options":
        return {
            "input": "Question: {{ question }}\nAnswer options:\nA) {{options['A']}}\nB) {{options['B']}}\n C) {{options['C']}}\n D) {{options['D']}}\n",
        }

    raise ValueError(f"Dataset {dataset} not supported.")


class PromptConfig(pydantic.BaseModel):
    """INPUT: templates, shots, functions.
    OUTPUT: messages (vllm) eller messages, functions, function_call (openai).
    """

    system_message: str = pydantic.Field(..., description="system message")
    templates: dict[str, str] = pydantic.Field(..., description="input template and output template for each dataset")
    shots: list[Shot] = pydantic.Field(..., description="list of shots")
    extract_function: Type[ExtractFunction] | None = pydantic.Field(None, description="output format function")
    tools: FindZebraAPI | None = pydantic.Field(None, description="list of functions")

    class Config:
        """Config for pydantic."""

        arbitrary_types_allowed = True

    @property
    def messages(self) -> list[ChatMessage]:
        """Build the messages from the system message, templates, shots and functions."""
        messages = []
        # This is heavily influenced by the way functionary does it!
        functions = []
        if self.extract_function:
            functions.append(Function(**self.extract_function.openai_schema))
            extract_function_schema = generate_schema_from_functions(functions=functions, namespace="functions")
            messages.append(ChatMessage(role="system", content=self.system_message + "\n" + extract_function_schema))

        if self.tools:
            messages.append(ChatMessage(role="system", content=self.system_message + "\n" + self.tools.typescript))

        # add shots
        for shot in self.shots:
            messages.append(
                ChatMessage(
                    role="user",
                    content=jinja2.Template(self.templates["input"]).render(
                        question=shot.question, opa=shot.opa, opb=shot.opb, opc=shot.opc, opd=shot.opd
                    ),
                )
            )
            # add shots as a function call
            if self.extract_function:
                extract_function = self.extract_function.from_shot(shot)
                content = f"function = {{\"name\": \"{extract_function.openai_schema['name']}\", \"arguments\": {extract_function.from_shot(shot).model_dump_json(indent=0)}}}"
                messages.append(ChatMessage(role="assistant", content=content))
            # or add shots as "free-text"
            else:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=jinja2.Template(self.templates["output"]).render(
                            explanation=shot.explanation, target=shot.target
                        ),
                    )
                )

        # add question
        messages.append(ChatMessage(role="user", content=self.templates["input"]))
        return messages


# For calling the function that the model has chosen with the right arguments..
# def get_output_model(function_name: str) -> pydantic.BaseModel:
#     """Get the output model from the class name."""
#     mod = import_module("path.to.functions")
#     return getattr(mod, function_name)

if __name__ == "__main__":
    system_message = "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct."
    templates = get_templates("medmcqa")
    shots = [Shot(**s) for s in get_fixed_shots("medmcqa")]
    tools = None

    extract_function = QuestionAnswer

    prompt_config = PromptConfig(
        system_message=system_message, templates=templates, shots=shots, extract_function=extract_function, tools=tools
    )

    messages = prompt_config.messages

    for message in messages:
        print(message.role)
        print(message.content)
        print("\n")

    pass
