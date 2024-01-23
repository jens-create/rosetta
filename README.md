
# Inference-oriented Framework for Large Language Models Reasoning Capabilities in Medicine.
> 🚧 **Warning**: Work in progress


The goal of this project is to create a framework for evaluating large language models on inference tasks in medicine. The majority of the code is the client, whereas the vLLM framework is to expose hosted LLM models through an API.


## Installation

### Server setup
I have used both the DTU Titans and the MeluXina servers for running experiments. The following steps are for setting up the server.

- run the install_vllm.sh script `bash scripts/install_vllm.sh`


### Client setup
The client is the main part of the project. It is used for running experiments and evaluating the models. The following steps are for setting up the client.

- Install poetry `curl -sSL https://install.python-poetry.org | python3 -`
- Might have to select python executable manually - i.e. `poetry env use /home/s183568/.conda/envs/py311/bin/python`
- Install dependencies `poetry install`



## Running experiments

### Hosting models
- Spin up models on the server using `sbatch scripts/meluxina/init_vllm.sh` or `sbatch scripts/titans/init_vllm_xl.sh`, where the latter is for the large (~>30b parameters) models.
- Create a double ssh bridge between local computer and compute node through meluxina login node. For example, `ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 meluxina -p 8822` and `ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 mel2129`.


### Running experiments on client side
Typer is used for running experiments. The following steps are for running experiments on the client side.

- Run experiments using `poetry run python experiments/medicalqa/main.py --model=<model_name> --agent=<agent_name>`, where `<model_name>` is the name of the model to be used and `<agent_name>` is the name of the agent to be used. For example, `poetry run python experiments/medicalqa/main.py --model=mixtral70b --agent=FewShotCoT`.
- For multiple experiments on the same model,  `python experiments/medicalqa/main_multiproc.py` should be used.


