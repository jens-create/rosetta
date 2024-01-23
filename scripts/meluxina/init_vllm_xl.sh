#!/bin/bash
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                        # number of tasks (2 tasks for 2 datasets)
#SBATCH --gpus-per-task=4                  # number of gpu per task (2 GPUs per task)
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=02:00:00                    # time (HH:MM:SS)
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

# Load CUDA module
#module load CUDA/11.7

# Source credentials
source credentials.txt

echo "====== starting experiment ========="


# MODELS = {
#     "mistral7b": {"checkpoint": "mistralai/Mistral-7B-v0.1", "endpoint": "localhost:8001"},
#     "mixtral8x7b": {"checkpoint": "mistralai/Mixtral-8x7B-v0.1", "endpoint": "localhost:8002"},
#     "mistral8x7b-instructv1": {"checkpoint": "mistralai/Mixtral-8x7B-Instruct-v0.1", "endpoint": "localhost:8003"},
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model mistralai/Mixtral-8x7B-v0.1 --port 8002 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8003 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-hf --port 8008 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8011 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model codellama/CodeLlama-34b-Instruct-hf --port 8014 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model codellama/CodeLlama-34b-hf --port 8017 --tensor-parallel-size 4 --download-dir /project/scratch/p200149/vllm

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.1 --port 8004 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm"

#srun --exclusive -n 1 -o job/%J-mixtral8x7b.out -e job/%J-mixtral8x7b.err python -m vllm.entrypoints.api_server --model mistralai/Mixtral-8x7B-v0.1 --port 8002 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-mistral8x7b-instructv1.out -e job/%J-mistral8x7b-instructv1.err python -m vllm.entrypoints.api_server --model mistralai/Mixtral-8x7B-v0.1 --port 8003 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-llama70b.out -e job/%J-llama70b.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-hf --port 8008 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-llama70b-chat.out -e job/%J-llama70b-chat.err python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8011 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-codellama34b-instruct.out -e job/%J-codellama34b-instruct.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-34b-Instruct-hf --port 8014 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &
#srun --exclusive -n 1 -o job/%J-codellama34b.out -e job/%J-codellama34b.err python -m vllm.entrypoints.api_server --model codellama/CodeLlama-34b-hf --port 8017 --tensor-parallel-size 4 --trust-remote-code --download-dir "/project/scratch/p200149/vllm" $@ &

#wait