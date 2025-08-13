#!/bin/bash
#SBATCH --job-name=protenix_mg_41   # job name
#SBATCH --time=1-00:00:00   	   # max job run time dd-hh:mm:ss
#SBATCH --ntasks-per-node=1      # tasks (commands) per compute node
#SBATCH --cpus-per-task=48          # CPUs (threads) per command
#SBATCH --mem=90G                  # total memory per node
#SBATCH --gres=gpu:a100           # request 1 A100 GPU per node
#SBATCH --exclude=g053,g002,g084
#SBATCH --output=/scratch/user/dimi/Protenix/outerror/out_.%j.%x	# save stdout to file
#SBATCH --error=/scratch/user/dimi/Protenix/outerror/error_.%j.%x        # save stderr to file

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=FAIL              #Send email on all job events
#SBATCH --mail-user=dimi@tamu.edu    #Send all emails to email_address

cd /scratch/user/dimi/Protenix

module purge
export SINGULARITYENV_TF_FORCE_UNIFIED_MEMORY=1
export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

module load GCCcore/11.3.0
module load Python/3.10.4
module load Anaconda3/2024.02-1
module load WebProxy/0000

export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_CCBIN=$(which g++)

source activate protenix
echo activated protenix
which python

export LAYERNORM_TYPE=openfold
export USE_DEEPSPEED_EVO_ATTENTION=false
export CUTLASS_PATH=/scratch/user/dimi/cutlass
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"

N_sample=5
N_step=200
N_cycle=10
seed=1729

input_json_path="./input_jsons/mg/mg_41.json"
dump_dir="./output/mg"
model_name="protenix_base_default_v0.5.0"

python3 -m runner.inference \
--model_name ${model_name} \
--seeds ${seed} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step}
