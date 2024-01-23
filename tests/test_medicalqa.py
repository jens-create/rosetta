# tests for medicalqa.py
import pytest
from clients.vllm.client import VllmAPI
from components.loader import load_hf_dataset

from experiments.medicalqa.exps import (  # noqa
    LLama7ChatZeroShotExperiment,
    LLama7ChatZeroShotFindZebraExperiment,
    LLama7FoundationZeroShotExperiment,
    LLama13ChatFewShotExperiment,
    LLama13ChatZeroShotExperiment,
    LLama13ChatZeroShotFindZebraExperiment,
)
from experiments.medicalqa.models import MedMcQaLoader, Titans

experiments = [
    LLama7FoundationZeroShotExperiment,
    LLama7ChatZeroShotExperiment,
    LLama13ChatZeroShotExperiment,
    LLama13ChatFewShotExperiment,
    LLama7ChatZeroShotFindZebraExperiment,
    LLama13ChatZeroShotFindZebraExperiment,
]
model_names = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
]


@pytest.mark.parametrize("experiment, model_name", zip(experiments, model_names))
def test_experiment(experiment, model_name):
    sys = Titans()
    ds = MedMcQaLoader(num_samples=10)
    client = VllmAPI

    exp = experiment.from_partial(sys, ds, client)

    ds = exp.dataset
    dataset = load_hf_dataset(
        name_or_path=ds.name_or_path,
        split=ds.split,
        subset=ds.subset,
        num_samples=ds.num_samples,
        seed=ds.seed,
        cache_dir=str(ds.cache_dir),
    )
    agent = exp.agent(client=exp.client, dataset=ds.name_or_path, request=dataset[0])

    assert agent is not None
    assert agent.model == model_name