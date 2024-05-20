#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --account=ai4bio2023
#SBATCH --partition=all_usr_prod
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1

python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --dataset_kwargs label_name=experiment --model_kwargs train_only_classifier=True --pretrained_model_path ./logs_resnet50/rxrx1_seed:0_epoch:best_model.pth --log_dir ./logs_resnet50_experiment