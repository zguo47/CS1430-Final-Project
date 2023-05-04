#!/bin/sh

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 7:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linghai_liu@brown.edu

module load python/3.7.4
cd /users/lliu58/data/lliu58/topo-reps
source /users/lliu58/my_cool_science/bin/activate

python -u /users/lliu58/data/lliu58/topo-reps/main.py
python -u /users/lliu58/data/lliu58/topo-reps/reverse_correlation.py

deactivate
