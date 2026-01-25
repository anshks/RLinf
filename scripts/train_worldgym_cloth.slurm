#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=worldgym_cloth_grpo_pi05
#SBATCH --output=Rlinf_logs/%j_%x.out
#SBATCH --error=Rlinf_logs/%j_%x.err
#SBATCH --time=12:00:00
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"


cd /scratch/nl2752/RLinf
SIF=/scratch/nl2752/RLinf_env/rlinf_agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0.sif
OVERLAY=/scratch/nl2752/RLinf_env/overlay-50G-10M.ext3

singularity exec --fakeroot --bind /scratch,/projects --nv --overlay ${OVERLAY}:rw \
    ${SIF} /bin/bash -c '
set -e
source ~/.bashrc
source /opt/venv/openpi/bin/activate

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

export OPENAI_API_KEY="Your OPENAI API KEY"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_ENTITY="nl2752-new-york-university"
export WANDB_PROJECT="cloth-folding-rl"      
export RAY_TMPDIR=/scratch/nl2752/ray_tmp  
mkdir -p $RAY_TMPDIR

ray stop --force || true
unset RAY_ADDRESS

# Run WorldGym cloth folding training with GRPO
bash examples/embodiment/run_embodiment.sh worldgym_cloth_grpo_pi05
'
