import hashlib
import json
import typing as typ
from pathlib import Path

import datasets

from experiments.medicalqa.prompt import QuestionAnswer
from experiments.medicalqa.responses import McQuestionAnswering, OutputModel


class MetricProtocol(typ.Protocol):
    """Protocol for metrics."""

    @property
    def metric(self) -> str:
        """Name of the metric."""
        ...

    @property
    def metric_dependencies(self) -> dict[str, "MetricProtocol"]:
        """Return the metric dependencies."""
        ...


class SampleMetricProtocol(MetricProtocol):
    """Protocol for sample/instance level metrics."""

    @property
    def metric(self) -> str:
        """Name of the metric."""
        ...

    @property
    def is_sample(self) -> bool:
        """Return True if metric is sample level."""
        return True

    @property
    def metric_dependencies(self) -> dict[str, MetricProtocol]:
        """Return the metric dependencies."""
        return {}

    def __call__(self, batch: dict[str, list[typ.Any]], output_model: typ.Type[OutputModel]) -> list[typ.Any]:
        """Generic metric protocol."""
        ...


class DatasetMetricProtocol(MetricProtocol):
    """Protocol for dataset-level metrics."""

    @property
    def metric(self) -> str:
        """Name of the metric."""
        ...

    @property
    def metric_dependencies(self) -> dict[str, "MetricProtocol"]:
        """Return the metric dependencies."""
        ...

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> typ.Any:
        """Generic metric protocol."""
        ...


class ExactMatch(SampleMetricProtocol):
    """Compute the accuracy of the predictions."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "exactmatch"

    def __call__(self, batch: dict[str, list[typ.Any]], output_model: typ.Type[OutputModel]) -> list[float]:
        """Compute ExactMatch."""
        if issubclass(output_model, McQuestionAnswering):
            # return [y_true == y_pred for y_true, y_pred in zip(batch["cop"], batch["pred_answer"])]
            if "cop" in batch:  # medmcqa
                return [
                    ["A", "B", "C", "D"][y_true] == y_pred for y_true, y_pred in zip(batch["cop"], batch["prediction"])
                ]

            if "answer_idx" in batch:  # USMLE
                return [y_true == y_pred for y_true, y_pred in zip(batch["answer_idx"], batch["prediction"])]

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class validJSON(SampleMetricProtocol):
    """Check if the response is a valid function call."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "validJSON"

    def __call__(
        self, batch: dict[str, list[typ.Any]], output_model: typ.Type[OutputModel]
    ) -> dict[str, list[typ.Any]]:
        """See if the response is a valid function call."""
        if issubclass(output_model, McQuestionAnswering):
            valid_json_responses = []
            dict_list = []
            for completion in batch["completion"]:
                # split the completion string by [/INST] and select the last element
                completion_answer = completion[0].split("[/INST]")[-1]
                # see if "function =" is in the completion string
                if "function =" in completion_answer:
                    # remove "function =" and parse the json
                    completion_answer_dict = completion_answer.replace("function =", "")
                    # remove the trailing newline and whitespace
                    completion_answer_dict_raw = completion_answer_dict.strip()

                    # check if the completion is valid json
                    try:
                        d = json.loads(completion_answer_dict_raw)
                        valid_json_responses.append(True)
                        dict_list.append(d)
                    except:  # noqa: E722
                        valid_json_responses.append(False)
                        # print(completion_answer_dict_raw)
                        dict_list.append({"name": "dummy", "arguments": {"answer": "dummy", "explanation": "dummy"}})
                else:
                    valid_json_responses.append(False)
                    dict_list.append({"name": "dummy", "arguments": {"answer": "dummy", "explanation": "dummy"}})

            return {"validJSON": valid_json_responses, "dicts": dict_list}

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class validQuestionAnswer(SampleMetricProtocol):
    """Check if the response is a valid question answer."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "validQuestionAnswer"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {"validJSON": validJSON()}

    def __call__(
        self, batch: dict[str, list[typ.Any]], output_model: typ.Type[OutputModel]
    ) -> dict[str, list[typ.Any]]:
        """See if the response is a valid question answer."""
        if issubclass(output_model, McQuestionAnswering):
            valid_qa = []
            explanation = []
            answer = []
            for valid_json, d in zip(batch["validJSON"], batch["dicts"]):
                if valid_json:
                    try:
                        QA = QuestionAnswer(**d["arguments"])  # noqa: N806
                        valid_qa.append(True)
                        explanation.append(QA.explanation)
                        answer.append(QA.answer.value)
                    except:  # noqa: E722
                        valid_qa.append(False)
                        explanation.append("None")
                        answer.append("None")
                else:
                    valid_qa.append(False)
                    explanation.append("None")
                    answer.append("None")

            return {"validQuestionAnswer": valid_qa, "pred_explanation": explanation, "prediction": answer}

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class validFunctionCalling(DatasetMetricProtocol):
    """Check if the response is a valid function call (both in terms of JSON and QA)."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "validFunctionCalling"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {}

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> dict[str, typ.Any]:
        """See if the response is a valid function call."""
        if issubclass(output_model, McQuestionAnswering):
            return {
                "validFunction": sum(dataset["valid_indicator"]) / len(dataset),
                "validJSON": sum(dataset["valid_json"]) / len(dataset),
                "validQuestionAnswer": sum(dataset["valid_function_args"]) / len(dataset),
            }

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class validity(DatasetMetricProtocol):
    """Check if the response is a valid function call (both in terms of JSON and QA)."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "validity"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {}

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> dict[str, typ.Any]:
        """See if the response is a valid function call."""
        if issubclass(output_model, McQuestionAnswering):
            data = dataset["validity"]

            total_counts = {}

            # Initialize dictionaries to store total counts and averages for each function
            for function, values in data[0].items():
                total_counts[function] = {
                    "valid_function_args": 0,
                    "valid_indicator": 0,
                    "valid_json": 0,
                    "not_none_output": 0,
                    "count": 0,
                }

            # Calculate totals
            for d in data:
                for function, values in d.items():
                    total_counts[function]["valid_function_args"] += values["valid_function_args"]
                    total_counts[function]["valid_indicator"] += values["valid_indicator"]
                    total_counts[function]["valid_json"] += values["valid_json"]
                    total_counts[function]["not_none_output"] += values["not_none_output"]
                    total_counts[function]["count"] += 1

            # Calculate averages
            average_results = {}
            for function, counts in total_counts.items():
                average_results[function] = {
                    "valid_function_args": counts["valid_function_args"] / counts["count"],
                    "valid_indicator": counts["valid_indicator"] / counts["count"],
                    "valid_json": counts["valid_json"] / counts["count"],
                    "not_none_output": counts["not_none_output"] / counts["count"],
                }
            return average_results

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class Accuracy(DatasetMetricProtocol):
    """Dataset level accuracy."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "accuracy"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {"exactmatch": ExactMatch()}

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> float:
        """Compute accuracy."""
        if issubclass(output_model, McQuestionAnswering):
            correct = 0

            if "cop" in dataset.features:  # medmcqa
                for y_true, y_pred in zip(dataset["cop"], dataset["prediction"]):
                    y_true_str = ["A", "B", "C", "D"][y_true]
                    if y_true_str == y_pred:
                        correct += 1
                return correct / len(dataset)
            if "answer_idx" in dataset.features:  # USMLE
                for y_true, y_pred in zip(dataset["answer_idx"], dataset["prediction"]):
                    if y_true == y_pred:
                        correct += 1
                return correct / len(dataset)

            raise ValueError(f"Metric {self.metric} not implemented for dataset")

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class Id(SampleMetricProtocol):
    """Dataset level accuracy."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "id"

    def __call__(self, batch: dict[str, list[typ.Any]], output_model: typ.Type[OutputModel]) -> list[str]:
        """Compute accuracy."""
        if issubclass(output_model, McQuestionAnswering):
            # if id exists do nothing
            if "id" in batch:
                return batch["id"]

            # else create an id by creating a SHA
            return [
                hashlib.sha256((str(batch["question"][i]) + str(batch["answer"][i])).encode()).hexdigest()
                for i in range(len(batch["question"]))
            ]

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class ReActEvaluation(DatasetMetricProtocol):
    """Dataset level ReAct evaluation."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "react"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {}

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> dict[str, typ.Any]:
        if issubclass(output_model, McQuestionAnswering):
            # calculate how many times the react algorithm is successful and how many times it fails
            # there is three cases: {"status": 400, "message": "Unknown action type."}, {"status": 200, "message": "Success."}, {"status": 400, "message": "Too long chain"}

            # calculate how many times each tool is used
            success = 0
            unknown_action_type = 0  # e.g. Search[S,clc,"",,]
            too_long_chain = 0
            action_and_thought_not_extracted = 0

            for sample in dataset["code"]:
                if sample["status"] == 200:
                    success += 1
                elif sample["status"] == 400 and sample["message"] == "Unknown action type.":
                    unknown_action_type += 1
                elif sample["status"] == 400 and sample["message"] == "Too long chain.":
                    too_long_chain += 1
                elif (
                    sample["status"] == 400
                    and sample["message"] == "Not possible to extract thought and action from the response."
                ):
                    action_and_thought_not_extracted += 1
                else:
                    raise ValueError("Unknown status or message")

            return {
                "success": success / len(dataset),
                "unknown_action_type": unknown_action_type / len(dataset),
                "too_long_chain": too_long_chain / len(dataset),
                "action_and_thought_not_extracted": action_and_thought_not_extracted / len(dataset),
                "avg_chain_length": sum(dataset["chain_length"]) / len(dataset),
            }

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class ToolEvaluation(DatasetMetricProtocol):
    """Dataset level tool usage."""

    @property
    def metric(self) -> str:
        """Return name of metric."""
        return "tool_usage"

    @property
    def metric_dependencies(self) -> dict[str, SampleMetricProtocol]:
        """Return the metric dependencies."""
        return {}

    def __call__(self, dataset: datasets.Dataset, output_model: typ.Type[OutputModel]) -> dict[str, typ.Any]:
        """See if the response is a valid function call."""
        if issubclass(output_model, McQuestionAnswering):
            # dataset[tools]: list[str] with tools used in the question

            # calculate how many times each tool is used
            tool_usage = {}
            for tools in dataset["tools"]:
                for tool in tools:
                    if tool in tool_usage:
                        tool_usage[tool] += 1
                    else:
                        tool_usage[tool] = 1

            # calculate the average tool usage
            for tool in tool_usage:
                tool_usage[tool] = tool_usage[tool] / len(dataset)

            return tool_usage

        raise ValueError(f"Metric {self.metric} not implemented for output model {output_model}")


class MultiProcEvaluation:
    """A utility function to extract content from ds interface."""

    def __init__(
        self,
        metrics: list[SampleMetricProtocol],
        output_model: typ.Type[OutputModel],
    ) -> None:
        """Initialize the class."""
        self.metrics = metrics
        self.output_model = output_model

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        self.__dict__.update(state)

    def __call__(self, batch: dict[str, list[typ.Any]]) -> dict[str, list[typ.Any]]:
        """Extract content from a db and calculate metrics."""
        # calculate sample level metrics
        for m in self.metrics:
            batch.update({m.metric: m(batch=batch, output_model=self.output_model)})

        return batch


class Metric:
    """Metric class."""

    metrics: list[MetricProtocol]
    output_model: typ.Type[OutputModel]
    result_dir: Path

    def __init__(
        self,
        metrics: list[MetricProtocol],
        output_model: typ.Type[OutputModel],
        result_dir: Path,
    ):
        self.metrics = metrics
        self.output_model = output_model
        self.result_dir = result_dir

    @classmethod
    def from_config(
        cls, metrics: list[MetricProtocol], output_model: typ.Type[OutputModel], result_dir: Path
    ) -> "Metric":
        """Initialize the class."""
        # add dependency metrics
        dependency_metrics = [
            v for metric in metrics for _, v in metric.metric_dependencies.items() if v not in metrics
        ]
        metrics = dependency_metrics + metrics

        return cls(
            metrics=metrics,  # type: ignore
            output_model=output_model,
            result_dir=result_dir,
        )

    def __call__(self, dataset: datasets.Dataset) -> None:
        """Call the metric."""
        # Calculate instance/sample level metrics and add to dataset
        dataset = dataset.map(
            MultiProcEvaluation(
                metrics=[m for m in self.metrics if hasattr(m.__class__, "is_sample")],  # type: ignore
                output_model=self.output_model,
            ),
            batched=True,
            batch_size=8,
            num_proc=1,
        )

        # Calculate dataset level metrics
        dataset_metrics = {
            m.metric: m(dataset=dataset, output_model=self.output_model)  # type: ignore
            for m in self.metrics
            if not hasattr(m.__class__, "is_sample")
        }

        # add average for sample level metrics
        for metric in self.metrics:
            if hasattr(metric.__class__, "is_sample") and isinstance(dataset[metric.metric][0], bool):
                dataset_metrics[f"avg_{metric.metric}"] = sum(dataset[metric.metric]) / len(dataset)

        # format and print metrics for entire dataset metrics
        print(dataset_metrics)

        # save metrics to file
        with open(self.result_dir / "metrics.json", "w") as json_file:
            json.dump(dataset_metrics, json_file, indent=4)

        # save dataset to file
        dataset.to_json(self.result_dir / "dataset.jsonl")

    def save_config(self, configs: list[dict[str, typ.Any]]) -> None:
        """Save the experiment configuration."""
        with open(self.result_dir / "config.json", "w") as json_file:
            json.dump(configs, json_file, indent=4)
