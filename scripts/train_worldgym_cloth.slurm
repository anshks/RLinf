#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=worldgym_cloth_grpo_pi05
#SBATCH --open-mode=append
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --requeue

cd /projects/work/yang-lab2/as20482/RLinf

singularity exec --fakeroot --bind /scratch,/projects --nv --overlay $LAB2/RLinf/overlay-25GB-500K.ext3:rw \
    ../rlinf_agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0.sif /bin/bash -c '
set -e
source ~/.bashrc
source /opt/venv/openpi/bin/activate

export TRANSFORMERS_CACHE=/scratch/${USER}/.cache/huggingface
export HF_HOME=/scratch/${USER}/.cache/huggingface

ray stop --force || true
unset RAY_ADDRESS

# Run WorldGym cloth folding training with GRPO
bash examples/embodiment/run_embodiment.sh worldgym_cloth_grpo_pi05
'
