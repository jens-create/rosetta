import typer
from clients.pipes import ParallelResponseGenerator
from clients.vllm.client import VllmAPI
from components.loader import load_hf_dataset

from experiments.medicalqa.agents import (
    AnswerQuestionStructuredAgent,
    CoTAgent,
    DirectAgent,
    FewShotCoTAgent,
    FewShotDirectAgent,
    ReActAgent,
)
from experiments.medicalqa.agentsTools import (  # , WikipediaSummarizeAgent
    BaseAgentTools,
    WikipediaAgent,
    WikipediaCoTAgent,
    WikipediaDirectAgent,
    WikipediaFewShotDirectAgent,
)
from experiments.medicalqa.exps import (  # noqa
    CoT,
    Direct,
    Experiment,
    FewShotCoT,
    FewShotDirect,
    LLama7ChatZeroShotExperiment,
    LLama13ChatZeroShotExperiment,
    LLama13ChatZeroShotFindZebraExperiment,
    LLama70ChatZeroShotExperiment,
    LLama70ChatZeroShotFindZebraExperiment,
    LLamaCode7InstructZeroShotExperiment,
    LLamaCode13InstructZeroShotExperiment,
    LLamaCode34InstructZeroShotExperiment,
    Mistral7bInstruct,
    Raven,
    ReAct,
)

# from experiments.medicalqa.functions.wikipedia import search_wikipedia
from experiments.medicalqa.models import Local, MedMcQaLoader, Meluxina, Titans, USMLELoader  # noqa


def run(exp: Experiment) -> None:
    """Run the experiment."""
    ds = exp.dataset
    dataset = load_hf_dataset(
        name_or_path=ds.name_or_path,
        split=ds.split,
        subset=ds.subset,
        num_samples=ds.num_samples,
        seed=ds.seed,
        cache_dir=str(ds.cache_dir),
    )

    # run experiment with agent as orchestrator
    dataset = dataset.map(
        ParallelResponseGenerator(
            agent=exp.agent,
            client=exp.client,
            dataset=ds.name_or_path,
            prompt_type=exp.prompt_type,
            summarize=exp.summarize,
            two_step=exp.two_step,
            sumfilter=exp.sumfilter,
        ),
        batched=True,
        batch_size=1,
        num_proc=1,
        desc="Calling model...",
        keep_in_memory=True,
    )

    # Evaluate and save results
    if exp.metric is not None:
        exp.metric(dataset=dataset)


AGENTS = {
    "WikipediaFewShotDirect": WikipediaFewShotDirectAgent,
    "WikipediaDirect": WikipediaDirectAgent,
    "WikipediaCoT": WikipediaCoTAgent,
    "Wikipedia": WikipediaAgent,
    "BaseTools": BaseAgentTools,
    "Structured": AnswerQuestionStructuredAgent,
    "Direct": DirectAgent,
    "CoT": CoTAgent,
    "FewShotDirect": FewShotDirectAgent,
    "FewShotCoT": FewShotCoTAgent,
    "ReAct": ReActAgent,
}

DATASET = {
    "USMLE": USMLELoader(),
    "MedMCQA": MedMcQaLoader(),
}


# typer
def main(
    dataset: str = "USMLE",
    model: str = "mixtral8x7b-instructv1",  # "mistral7b-instructv1",
    agent: str = "WikipediaCoT",  # prompt: int = 11
    summarize: bool = True,
    summarize_filter: bool = True,
    two_step: bool = False,
) -> None:
    """Run the experiment."""
    print(
        f"Running!. Dataset: {dataset}, Model: {model}, Agent: {agent}, Summarize: {summarize}, SumFilter: {summarize_filter}, Two-step: {two_step}"
    )  # , Prompt: {prompt}")
    sys = Local()
    client = VllmAPI(**MODELS[model])
    exp = Experiment.from_config(
        model=model,
        system=sys,
        dataset=DATASET[dataset],
        client=client,
        agent=AGENTS[agent],
        prompt_type=11,
        summarize=summarize,
        sumfilter=summarize_filter,
        two_step=two_step,
    )
    run(exp)

    # Experiment done
    print(
        f"Done!. Dataset: {dataset}, Model: {model}, Agent: {agent}, Summarize: {summarize}, SumFilter: {summarize_filter}, Two-step: {two_step}"
    )


MODELS = {
    "mistral7b": {"checkpoint": "mistralai/Mistral-7B-v0.1", "endpoint": "localhost:8001"},
    "mixtral8x7b": {"checkpoint": "mistralai/Mixtral-8x7B-v0.1", "endpoint": "localhost:8002"},
    "mixtral8x7b-instructv1": {"checkpoint": "mistralai/Mixtral-8x7B-Instruct-v0.1", "endpoint": "localhost:8003"},
    "mistral7b-instructv1": {"checkpoint": "mistralai/Mistral-7B-Instruct-v0.1", "endpoint": "localhost:8004"},
    "mistral7b-instructv2": {"checkpoint": "mistralai/Mistral-7B-Instruct-v0.2", "endpoint": "localhost:8005"},
    "llama7b": {"checkpoint": "meta-llama/Llama-2-7b-hf", "endpoint": "localhost:8006"},
    "llama13b": {"checkpoint": "meta-llama/Llama-2-13b-hf", "endpoint": "localhost:8007"},
    # "llama70b": {"checkpoint": "meta-llama/Llama-2-70b-hf", "endpoint": "localhost:8008"},
    "llama7b-chat": {"checkpoint": "meta-llama/Llama-2-7b-chat-hf", "endpoint": "localhost:8009"},
    "llama13b-chat": {"checkpoint": "meta-llama/Llama-2-13b-chat-hf", "endpoint": "localhost:8010"},
    # "llama70b-chat": {"checkpoint": "meta-llama/Llama-2-70b-chat-hf", "endpoint": "localhost:8011"},
    "codellama7b-instruct": {"checkpoint": "codellama/CodeLlama-7b-Instruct-hf", "endpoint": "localhost:8012"},
    "codellama13b-instruct": {"checkpoint": "codellama/CodeLlama-13b-Instruct-hf", "endpoint": "localhost:8013"},
    # "codellama34b-instruct": {"checkpoint": "codellama/CodeLlama-34b-Instruct-hf", "endpoint": "localhost:8014"},
    "codellama7b": {"checkpoint": "codellama/CodeLlama-7b-hf", "endpoint": "localhost:8015"},
    "codellama13b": {"checkpoint": "codellama/CodeLlama-13b-hf", "endpoint": "localhost:8016"},
    # "codellama34b": {"checkpoint": "codellama/CodeLlama-34b-hf", "endpoint": "localhost:8017"},
}


if __name__ == "__main__":
    # sys = Local()
    # ds = MedMcQaLoader(num_samples=1000)
    # model = "mistral7b"
    # agent = "CoT"
    typer.run(main)
    # main(model, agent)

    # exp = Experiment.from_config(system=sys, dataset=ds, client=VllmAPI(**MODELS["mixtral70b"]), agent=AGENTS[agent])

    # exp =
    # exp = LLama7ChatZeroShotExperiment.from_partial(sys, ds, client)
    # exp = LLama13ChatZeroShotExperiment.from_partial(sys, ds, client)
    # exp = LLamaCode7InstructZeroShotExperiment.from_partial(sys, ds, client)
    # exp = LLamaCode13InstructZeroShotExperiment.from_partial(sys, ds, client)
    # exp = LLamaCode34InstructZeroShotExperiment.from_partial(sys, ds, client)
    # exp = Mistral7bInstruct.from_partial(sys, ds, client) # mistral 7b v0.1

    # Mistral7B
    # exp = ReAct.from_partial(sys, ds, client)
    # exp = Direct.from_partial(sys, ds, client)  # {'accuracy': 0.433}
    # exp = CoT.from_partial(sys, ds, client) #{'accuracy': 0.373} # 82 samples are not extracted properly.
    # exp = FewShotCoT.from_partial(sys, ds, client) #{'accuracy': 0.495}
    # exp = FewShotDirect.from_partial(sys, ds, client) # {'accuracy': 0.469}

    # Mixtral70B
    # FewShotCoT: {'accuracy': 0.569}
    # FewShotDirect: {'accuracy': 0.549} # muligvis noget galt...
    # CoT: {'accuracy': 0.479}
    # Direct: {'accuracy': 0.532}

    # USMLE
    # Mixtral70B
    # FewShotCoT: {'accuracy': 0.581}
    # FewShotDirect:
    # CoT:
    # Direct:

    # main(exp)
