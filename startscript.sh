#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=RASDA
#SBATCH --account=
#SBATCH --output=RASDA.out
#SBATCH --error=RASDA.err
#SBATCH --partition=booster
#SBATCH --account=
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive

# set modules 2024
ml --force purge
ml Stages/2024 GCCcore/.12.3.0 Python/3.11.3 

ml NCCL/default-CUDA-12 PyTorch/2.1.2 torchvision/0.16.2

source ray_juwels_env/bin/activate

dataDir="//p/scratch/cslfse/aach1/imagenet-1K-tfrecords"
storagePath="/p/scratch/cslfse/aach1/ray_results"

COMMAND="adaptive_ray.py --scheduler ASHA --num-samples 4  --num-workers 4 --num-epochs 8 \
--base-cpus 12 --base-gpus 1 --grace-period 1 --seed 111 --data-dir $dataDir  --storage-path $storagePath --report-interval 5 --res-changer "

echo $COMMAND
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"

# CUDA, InfiniBand, srun and python flags
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NCCL_SOCKET_IFNAME="ib0"
export PYTHONPATH="${PYTHONPATH}:$PWD"

num_gpus=4

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1


####### this part is taken from the ray example slurm script #####
set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

port=37465

# InfiniBand version of IP address
export ip_head="$head_node"i:"$port"


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --node-ip-address="$node_i"i \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    ##sleep 5
done

echo "Ready"

python -u $COMMAND
