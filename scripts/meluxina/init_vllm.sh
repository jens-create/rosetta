#!/bin/bash
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=4                        # number of tasks (2 tasks for 2 datasets)
#SBATCH --gpus-per-task=1                  # number of gpu per task (2 GPUs per task)
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=03:00:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200149                  # project account
#SBATCH --qos=default                      # SLURM qos
#SBATCH --error=job/%J.err
#SBATCH --output=job/%J.out

# display args
echo "===================================="
echo "ARGS       = $@"
echo "===================================="

echo Running on host $USER@$HOSTNAME
echo Node: $(hostname)
echo Start: $(date +%F-%R:%S)
echo -e Working dir: $(pwd)
echo Dynamic shared libraries: $LD_LIBRARY_PATH

# Source credentials
source credentials.txt

echo "====== starting experiment ========="

# MODELS = {
#     "mistral7b": {"checkpoint": "mistralai/Mistral-7B-v0.1", "endpoint": "localhost:8001"},
#     "mixtral8x7b": {"checkpoint": "mistralai/Mixtral-8x7B-v0.1", "endpoint": "localhost:8002"},
#     "mistral8x7b-instructv1": {"checkpoint": "mistralai/Mixtral-8x7B-v0.1", "endpoint": "localhost:8003"},
#     "mistral7b-instructv1": {"checkpoint": "mistralai/Mistral-7B-Instruct-v0.1", "endpoint": "localhost:8004"},
#     "mistral7b-instructv2": {"checkpoint": "mistralai/Mistral-7B-Instruct-v0.2", "endpoint": "localhost:8005"},
#     "llama7b": {"checkpoint": "meta-llama/Llama-2-7b-hf", "endpoint": "localhost:8006"},
#     "llama13b": {"checkpoint": "meta-llama/Llama-2-13b-hf", "endpoint": "localhost:8007"},
#     "llama70b": {"checkpoint": "meta-llama/Llama-2-70b-hf", "endpoint": "localhost:8008"},
#     "llama7b-chat": {"checkpoint": "meta-llama/Llama-2-7b-chat-hf", "endpoint": "localhost:8009"},
#     "llama13b-chat": {"checkpoint": "meta-llama/Llama-2-13b-chat-hf", "endpoint": "localhost:8010"},
#     "llama70b-chat": {"checkpoint": "meta-llama/Llama-2-70b-chat-hf", "endpoint": "localhost:8011"},
#     "codellama7b-instruct": {"checkpoint": "codellama/CodeLlama-7b-Instruct-hf", "endpoint": "localhost:8012"},
#     "codellama13b-instruct": {"checkpoint": "codellama/CodeLlama-13b-Instruct-hf", "endpoint": "localhost:8013"},
#     "codellama34b-instruct": {"checkpoint": "codellama/CodeLlama-34b-Instruct-hf", "endpoint": "localhost:8014"},
#     "codellama7b": {"checkpoint": "codellama/CodeLlama-7b-hf", "endpoint": "localhost:8015"},
#     "codellama13b": {"checkpoint": "codellama/CodeLlama-13b-hf", "endpoint": "localhost:8016"},
#     "codellama34b": {"checkpoint": "codellama/CodeLlama-34b-hf", "endpoint": "localhost:8017"},
# }

# Run the experiment with the appropriate dataset and prompt using srun
#srun --exclusive -n 1 -o job/%J-mistral7b.out -e job/%J-mistral7b.err python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-v0.1 --port 8001 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-mistral7b-instructv1.out -e job/%J-mistral7b-instructv1.err python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.1 --port 8004 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-mistral7b-instructv2.out -e job/%J-mistral7b-instructv2.err python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8005 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &

#srun --exclusive -n 1 -o job/%J-llama7b.out -e job/%J-llama7b.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-hf --port 8006 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-llama13b.out -e job/%J-llama13b.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-13b-hf --port 8007 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-llama7b-chat.out -e job/%J-llama7b-chat.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --port 8009 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-llama13b-chat.out -e job/%J-llama13b-chat.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-13b-chat-hf --port 8010 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &


#srun --exclusive -n 1 -o job/%J-codellama7b-instruct.out -e job/%J-codellama7b-instruct.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-7b-Instruct-hf --port 8012 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-codellama13b-instruct.out -e job/%J-codellama13b-instruct.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-13b-Instruct-hf --port 8013 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-codellama7b.out -e job/%J-codellama7b.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-7b-hf --port 8015 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-codellama13b.out -e job/%J-codellama13b.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-13b-hf --port 8016 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &

# instruct models
srun --exclusive -n 1 -o job/%J-mistral7b-instructv1.out -e job/%J-mistral7b-instructv1.err python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.1 --port 8004 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
srun --exclusive -n 1 -o job/%J-mistral7b-instructv2.out -e job/%J-mistral7b-instructv2.err python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8005 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
srun --exclusive -n 1 -o job/%J-llama13b-chat.out -e job/%J-llama13b-chat.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-13b-chat-hf --port 8010 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
srun --exclusive -n 1 -o job/%J-codellama13b-instruct.out -e job/%J-codellama13b-instruct.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-13b-Instruct-hf --port 8013 --tensor-parallel-size 1 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &

wait