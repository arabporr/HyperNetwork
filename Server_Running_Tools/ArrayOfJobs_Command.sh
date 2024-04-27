#!/bin/bash
#SBATCH --job-name=HN_Run_1
#SBATCH --qos=m2
#SBATCH --cpus-per-task 6
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=07:30:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=3
#SBATCH --array=1-25
#SBATCH --output=outputs/test_%A_%a.out
#SBATCH --error=errors/test_%A_%a.err

# Environment Setup
module purge
module load python/3.12.0
pip3 install --upgrade pip
pip3 -U -q install pandas numpy tensorflow cuda-python torch torchvision seaborn plotly matplotlib ipywidgets tqdm

# Run experiment
sed -n "${SLURM_ARRAY_TASK_ID}p" Commands.sh | bash
