import re
import typing as typ

import pydantic

# from clients.models import ChatCompletion

SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Create a generic variable that can be 'OutputModel', or any subclass.
T = typ.TypeVar("T", bound="OutputModel")


class ChatCompletion(pydantic.BaseModel):
    ...


class McQueryModel(pydantic.BaseModel):
    ...


class OutputModel(pydantic.BaseModel):
    """A generic class for model outputs."""

    query: str
    completion: str

    @classmethod
    def from_response(cls: typ.Type[T], completion: ChatCompletion, request: dict[str, typ.Any]) -> "OutputModel":
        """Create a chat completion from the response."""
        ...


class McQuestionAnswering(OutputModel):
    """MultipleChoice Question Answering class for model outputs."""

    answers: list[str]
    target: int
    prediction: int

    @classmethod
    def from_response(
        cls, completion: ChatCompletion, request: dict[str, typ.Any]  # noqa: ANN102
    ) -> "McQuestionAnswering":
        """Create a chat completion from the response."""
        m = McQueryModel(**request)

        # if function call is present, extract the answer from the function call
        # if completion.choices[0].message.function_call:
        #     # print("Function_call...")
        #     args = completion.choices[0].message.function_call.arguments
        #     explanation = args.get("explanation", "")
        #     answer = args.get("answer", "")
        #     try:
        #
        #         prediction = int(answer)  # SYMBOLS.index(answer)
        #     except:
        #         prediction = extract_multiple_choice_prediction(
        #             completion=str(answer) + str(explanation), choices=m.answers
        #         )
        #     return cls(**m.dict(), completion=str(explanation), prediction=prediction)
        # # if extract the answer from the completion message
        # if completion.choices[0].message.content:
        #     completion_str = completion.choices[0].message.content
        #     prediction = extract_multiple_choice_prediction(completion=completion_str, choices=m.answers)
        #
        #     return cls(**m.dict(), completion=completion_str, prediction=prediction)
        return cls(**m.dict(), completion="", prediction=0)

        # raise ValueError("Neither message or function call found in the completion message.")


def extract_answer_idx(answer_str: str, options: list[str]) -> int:
    """Extracts the index of the selected answer option from a given answer string.
    :param answer_str: The answer string to extract the answer index from.
    :param options: The list of answer options to match against.
    :return: The index of the selected answer option, or -1 if the answer index couldn't be inferred.
    """
    symbols_pattern = r"(?:^|\()([A-J])(?:[\s.,:\)]|$)"
    exact_answers = "|".join([re.escape(option) for option in options])
    answers = "|".join([re.escape(option.lower()) for option in options])

    matches_symbol = re.findall(symbols_pattern, answer_str, re.MULTILINE)
    matches_exact = re.findall(exact_answers, answer_str, re.MULTILINE)
    matches_answer = re.findall(answers, answer_str, re.MULTILINE | re.IGNORECASE)

    if len(set(matches_exact)) == 1:
        predicted_idx = options.index(matches_exact[0])
    elif len(set(matches_answer)) == 1:
        lowered_options = [option.lower() for option in options]
        predicted_idx = lowered_options.index(matches_answer[0].lower())
    elif len(set(matches_symbol)) == 1:
        predicted_idx = SYMBOLS.index(matches_symbol[0])
    else:
        predicted_idx = -1
    return predicted_idx


def extract_multiple_choice_prediction(completion: str, choices: list[str]) -> int:
    """Parses the generated string into a dict containing reasoning path and prediction.
    :param completion: The generated string to be parsed.
    :param eg: A dictionary containing the query data.
    :return: A dict containing the parsed data.
    """
    # sample = format_sample(eg["uid"], eg["question"], [e.strip() for e in eg["choices"]], eg["target"])
    first_letter = r"(^[A-J])"
    last_letter = r"[A-J]$"
    simple_pattern = r"answer([^.]*)\."
    option_pattern = "|".join([re.escape(option) for option in choices])
    symbol_pattern = r"\(?[A-J](?![A-Za-z0-9])[\)\.\:]?"
    first_letter_match = re.search(first_letter, completion, re.MULTILINE)
    last_letter_match = re.search(last_letter, completion, re.MULTILINE)
    answer_matches = re.search(simple_pattern, completion, re.MULTILINE | re.IGNORECASE)
    option_matches = re.findall(option_pattern, completion, re.MULTILINE | re.IGNORECASE)
    symbol_matches = re.findall(symbol_pattern, completion, re.MULTILINE | re.IGNORECASE)

    answer_str = completion
    if answer_matches:
        answer_str = answer_matches.group(1)
        predicted_idx = extract_answer_idx(answer_str, choices)
    elif len(set(option_matches)) == 1:
        lowered_options = [option.lower() for option in choices]
        predicted_idx = lowered_options.index(option_matches[0].lower())
    elif len(set(symbol_matches)) == 1:
        answer_str = symbol_matches[0]
        predicted_idx = extract_answer_idx(answer_str, choices)
    elif first_letter_match:
        predicted_idx = SYMBOLS.index(first_letter_match.group())
    elif last_letter_match:
        predicted_idx = SYMBOLS.index(last_letter_match.group())
    else:
        sep = "\n\n" if "\n\n" in completion else "."
        answer_str = completion.split(sep)[0].strip()
        first_sentence_check = extract_answer_idx(answer_str, choices)
        whole_sentence_check = extract_answer_idx(completion, choices)
        predicted_idx = max(first_sentence_check, whole_sentence_check)

    return predicted_idx


def extract_prediction(text, pattern=r"Therefore, among A through D, the answer is \(([ABCD])\)"):
    """Extracts the prediction (A, B, C, or D) from a given string.

    The function searches for the pattern 'Therefore, among A through D, the answer is (X)'
    where X is A, B, C, or D and extracts the prediction.

    Args:
    text (str): The text containing the prediction.

    Returns:
    str: The extracted prediction (A, B, C, or D).
    """
    text = text.replace("\n", " ")
    match = re.search(pattern, text)
    return match.group(1) if match else "NA"
