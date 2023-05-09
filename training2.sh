#!/bin/sh

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -p gpu --gres=gpu:1

module load python/3.7.4
cd <final-project-directory>
source <path to venv>/bin/activate

python -u <final-project-directory>/main.py
