from pathlib import Path

import typer
import yaml
from rosetta_cli.components.experiments import run_vllm_experiment
from rosetta_cli.components.loader import load_hf_dataset
from rosetta_cli.components.models import ExperiemntConfig, HfDsetLoader

app = typer.Typer()


@app.command()
def dataset(files: list[Path]) -> None:
    """Load HuggingFace datasets."""
    for path in files:
        dset_config = HfDsetLoader(path=path)
        dataset = load_hf_dataset(**dset_config.model_dump(exclude_none=True))
        print(dataset)


@app.command()
def experiment(files: list[Path]) -> None:
    """Run experiement with VLLM given a experiement configuration."""
    for path in files:
        with open(path, "r") as file:
            config_dict = yaml.safe_load(file)
        exp_config = ExperiemntConfig(**config_dict)
        run_vllm_experiment(exp_config.builder, exp_config.evaluator, exp_config.model, exp_config.prompt)


if __name__ == "__main__":
    app()
