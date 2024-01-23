import re
import typing as typ

import datasets
import pydantic
from datasets import fingerprint


class ExactMatchModel(pydantic.BaseModel):
    """A model for ExactMatch."""

    predictions: list[str] = pydantic.Field(..., description="The column name containing the predictions.")
    answers: list[list[str]] = pydantic.Field(..., description="The column name containing the answer(s).")


class EcxactMatchAdapter:
    """A pipeline that evaluates the predicton to a multiple choice question based on ExactMatch."""

    Im = ExactMatchModel

    def __init__(self, prediction_column: str, answers_column: str, target_column: str) -> None:
        # Initialize any required variables or models here
        self.prediction_column = prediction_column
        self.answers_column = answers_column
        self.target_column = target_column
        self.symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        self.__dict__.update(state)

    def extract_answer_idx(self, prediciton: str, answers: list[str]) -> int:
        """Extracts the index of the selected answer option from a given answer string.
        :param answer_str: The answer string to extract the answer index from.
        :param options: The list of answer options to match against.
        :return: The index of the selected answer option, or -1 if the answer index couldn't be inferred.
        """
        symbols_pattern = r"(?:^|\()([A-J])(?:[\s.,:\)]|$)"
        exact_answers_pattern = "|".join([re.escape(option) for option in answers])
        answers_pattern = "|".join([re.escape(option.lower()) for option in answers])

        matches_symbol = re.findall(symbols_pattern, prediciton, re.MULTILINE)
        matches_exact = re.findall(exact_answers_pattern, prediciton, re.MULTILINE)
        matches_answer = re.findall(answers_pattern, prediciton, re.MULTILINE | re.IGNORECASE)

        if len(set(matches_exact)) == 1:
            predicted_idx = answers.index(matches_exact[0])
        elif len(set(matches_answer)) == 1:
            lowered_options = [option.lower() for option in answers]
            predicted_idx = lowered_options.index(matches_answer[0].lower())
        elif len(set(matches_symbol)) == 1:
            predicted_idx = self.symbols.index(matches_symbol[0])
        else:
            predicted_idx = -1
        return predicted_idx

    def extract_multiple_choice_prediction(self, prediction: str, answers: list[str]) -> int:
        """Parses the generated string into a dict containing reasoning path and prediction.
        :param completion: The generated string to be parsed.
        :param eg: A dictionary containing the query data.
        :return: A dict containing the parsed data.
        """
        first_letter = r"(^[A-J])"
        last_letter = r"[A-J]$"
        simple_pattern = r"answer([^.]*)\."
        option_pattern = "|".join([re.escape(option) for option in answers])
        symbol_pattern = r"\(?[A-J](?![A-Za-z0-9])[\)\.\:]?"
        first_letter_match = re.search(first_letter, prediction, re.MULTILINE)
        last_letter_match = re.search(last_letter, prediction, re.MULTILINE)
        answer_matches = re.search(simple_pattern, prediction, re.MULTILINE | re.IGNORECASE)
        option_matches = re.findall(option_pattern, prediction, re.MULTILINE | re.IGNORECASE)
        symbol_matches = re.findall(symbol_pattern, prediction, re.MULTILINE | re.IGNORECASE)

        answer_str = prediction
        if answer_matches:
            answer_str = answer_matches.group(1)
            predicted_idx = self.extract_answer_idx(answer_str, answers)
        elif len(set(option_matches)) == 1:
            lowered_options = [option.lower() for option in answers]
            predicted_idx = lowered_options.index(option_matches[0].lower())
        elif len(set(symbol_matches)) == 1:
            answer_str = symbol_matches[0]
            predicted_idx = self.extract_answer_idx(answer_str, answers)
        elif first_letter_match:
            predicted_idx = self.symbols.index(first_letter_match.group())
        elif last_letter_match:
            predicted_idx = self.symbols.index(last_letter_match.group())
        else:
            sep = "\n\n" if "\n\n" in prediction else "."
            answer_str = prediction.split(sep)[0].strip()
            first_sentence_check = self.extract_answer_idx(answer_str, answers)
            whole_sentence_check = self.extract_answer_idx(prediction, answers)
            predicted_idx = max(first_sentence_check, whole_sentence_check)

        return predicted_idx

    def __call__(self, batch: dict[str, list[typ.Any]], **kwargs: typ.Any) -> dict[str, list[typ.Any]]:
        """Extract content from a db."""
        m = self.Im(
            predictions=batch[self.prediction_column],
            answers=batch[self.answers_column],
        )
        predictions = []
        for prediction, answers in zip(m.predictions, m.answers):
            predictions.append(self.extract_multiple_choice_prediction(prediction, answers))

        batch.update({"prediction": predictions})
        return batch


class ExactMatchAdapter:
    """A pipeline that evaluates the predicton to a multiple choice query based on ExactMatch."""

    def __init__(
        self, prediction_column: str, answers_column: str, target_column: str, single_answer: bool = True
    ) -> None:
        self.prediction_column = prediction_column
        self.answers_column = answers_column
        self.target_column = target_column
        self.single_answer = single_answer
        self.symbols = "ABCDEFGHIJ"

    def _find_answer_index(self, prediction: str, answers: list[str]) -> typ.Union[int, list[int]]:
        symbols_regex = rf"\b[{self.symbols}]\b"
        answers_regex = "|".join(re.escape(answer) for answer in answers)
        symbol_matches = re.findall(symbols_regex, prediction)
        exact_matches = re.findall(answers_regex, prediction, re.IGNORECASE)

        if self.single_answer:
            if symbol_matches:
                return self.symbols.index(symbol_matches[0])
            if exact_matches:
                return next(
                    (
                        answers.index(answer)
                        for answer in answers
                        if answer.lower() in (match.lower() for match in exact_matches)
                    ),
                    -1,
                )
            return -1
        indices = []
        for match in symbol_matches:
            indices.append(self.symbols.index(match))
        for match in exact_matches:
            index = next((answers.index(answer) for answer in answers if answer.lower() == match.lower()), None)
            if index is not None:
                indices.append(index)
        return indices or [-1]

    def _extract_prediction(self, prediction: str, answers: list[str]) -> typ.Union[int, list[int]]:
        # Simplify the method to focus on extracting prediction
        return self._find_answer_index(prediction, answers)

    def __call__(self, batch: dict[str, list[typ.Any]]) -> dict[str, list[typ.Any]]:
        """Perform ExactMatch prediction on batch."""
        model = ExactMatchModel(predictions=batch[self.prediction_column], answers=batch[self.answers_column])
        batch["prediction"] = [
            self._extract_prediction(prediction, answers)
            for prediction, answers in zip(model.predictions, model.answers)
        ]
        return batch


@fingerprint.hashregister(EcxactMatchAdapter)
def _hash_fetch_from_ds(hasher: datasets.fingerprint.Hasher, obj: EcxactMatchAdapter) -> str:
    """Register the _IsIdxIn class to work with datasets.map()."""
    return hasher.hash(
        {
            "cls": obj.__class__,
        }
    )
