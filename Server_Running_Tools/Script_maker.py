import os

script_content = """#!/bin/bash
#SBATCH --job-name=HN_{i}
#SBATCH --qos=normal
#SBATCH -c 6
#SBATCH --mem=20G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=13:59:00
#SBATCH --output=HN_{i}/slurm-%j.out
#SBATCH --error=HN_{i}/slurm-%j.err

# Environment Setup
module purge
module load python/3.12.0
pip3 install --upgrade pip
pip3 install -U -q pandas numpy tensorflow cuda-python torch torchvision seaborn plotly matplotlib ipywidgets tqdm

# Run Experiments
python3 main.py --data_index {i}
"""

num_files = 25
os.makedirs("scripts", exist_ok=True)

for i in range(num_files):
    script_filename = f"Slurm_{i}.sh"
    with open(os.path.join("scripts", script_filename), "w") as f:
        f.write(script_content.format(i=i))
