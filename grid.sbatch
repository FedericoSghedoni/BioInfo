#!/bin/bash
#SBATCH --job-name=grid4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --account=ai4bio2023
#SBATCH --partition=all_usr_prod
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1

python grid_search4.py