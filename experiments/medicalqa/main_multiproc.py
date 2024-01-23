import multiprocessing
import subprocess
from itertools import product


def run_experiment(agent, model):
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={agent} --model=llama7b --agent=Direct"
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={model} --model=mixtral8x7b-instructv1 --agent=Structured --prompt={agent}"
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={model} --model=mistral7b-instructv1 --agent=Structured --prompt={agent}"
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={agent} --model={model} --agent=Wikipedia"  # --prompt=4"
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={agent} --model=mixtral8x7b-instructv1 --agent=Wikipedia --no-summarize --{model}"  # --prompt=4"
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={agent} --model=mixtral8x7b-instructv1 --agent=BaseTools --summarize --{model}"  # --prompt=4"
    # Execute the command here, e.g., using os.system or subprocess.run

    # BaseAgentTools
    # command = f"poetry run python experiments/medicalqa/main.py --dataset={agent} --model=mixtral8x7b-instructv1 --agent=WikipediaCoT"  # --prompt=4"

    command = "poetry run python experiments/medicalqa/main.py --dataset=USMLE --model=mixtral8x7b --agent=BaseTools"  # --prompt=4"

    # Execute the command
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    # models = ["llama7b", "llama13b", "llama7b-chat", "llama13b-chat"]
    # models = ["mistral7b", "mistral7b-instructv1", "mistral7b-instructv2"]
    # models = [
    #     "mixtral8x7b",
    #     "mixtral8x7b-instructv1",
    #     "llama70b",
    #     "llama70b-chat",
    #     "codellama34b",
    #     "codellama34b-instruction",
    # ]
    # models = ["codellama13b", "codellama13b-instruct"]
    # models = ["mixtral8x7b-instructv1"]
    # models = ["mixtral8x7b-instructv1", "mistral7b-instructv1"]
    # models = ["llama70b"]
    # models = ["llama70b-chat"]
    # agents = ["Direct", "CoT", "FewShotDirect", "FewShotCoT"]
    # models = ["mistral7b-instructv1", "mistral7b-instructv2", "llama13b-chat", "codellama13b-instruct"]
    # agents = ["Direct", "FewShotDirect", "FewShotCoT"]
    agents = ["USMLE", "MedMCQA"]
    models = ["two-step", "no-two-step"]
    agents = ["s"]
    models = [-1]
    # agents = [11]

    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        pool.starmap(run_experiment, product(agents, models))
