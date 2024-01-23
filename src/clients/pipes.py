import hashlib
import json
import typing as typ
from pathlib import Path

from clients.base import BaseClient

from experiments.medicalqa.agents import Agent


class ParallelResponseGenerator:
    """A HuggingFace approach to parallelize the querying of a LLM."""

    def __init__(
        self,
        agent: typ.Type[Agent],
        client: BaseClient,
        dataset: str,
        prompt_type: int,
        summarize: bool,
        two_step: bool,
        sumfilter: bool = False,
    ) -> None:
        """Initialize the class."""
        self.agent = agent
        self.client = client
        self.dataset = dataset
        self.prompt_type = prompt_type
        self.summarize = summarize
        self.two_step = two_step
        self.sumfilter = sumfilter

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        self.__dict__.update(state)

    def __call__(self, batch: dict[str, list[typ.Any]]) -> dict[str, list[typ.Any]]:
        """Call the client in parallel."""
        batch_length = len(next(iter(batch.values())))
        responses = []
        for i in range(batch_length):
            sample = {key: batch[key][i] for key in batch}
            # Create an instance of the seleted Agent for each sample
            agent = self.agent(
                client=self.client,
                dataset=self.dataset,
                prompt_type=self.prompt_type,
                request=sample,
                two_step=self.two_step,
                summarize=self.summarize,
                sumfilter=self.sumfilter,
            )

            # Run the agent
            response = agent.run()

            # Collect the response
            responses.append(response)

        # Update the batch by inserting the responses.
        # from list[dict[str, typ.Any]] to dict[str, list[typ.Any]]

        batch.update({key: [sample[key] for sample in responses] for key in responses[0]})
        return batch


class DumpReponsesToJson:
    """A pipeline to dump responses to json."""

    def __init__(self, save_dir: str) -> None:
        self.save_dir = Path(
            save_dir,
            "responses",
        ).resolve()

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        self.__dict__.update(state)

    def __call__(self, batch: dict[str, list[typ.Any]], **kws: typ.Any) -> dict[str, list[typ.Any]]:
        """Dump each sample in a batch to a json file."""
        batch_length = len(next(iter(batch.values())))
        for i in range(batch_length):
            sample = {key: batch[key][i] for key in batch}
            self.dump_to_json(sample)
        return batch

    def dump_to_json(self, sample: dict[str, typ.Any]) -> None:
        """Dump a sample to a json file."""
        sample_string = json.dumps(sample, sort_keys=True)
        file_name = hashlib.sha256(sample_string.encode()).hexdigest()
        path_to_file = Path(self.save_dir, f"{file_name}.json")

        path_to_file.parent.mkdir(parents=True, exist_ok=True)
        with open(path_to_file, "w") as file:
            json.dump(sample, file)
