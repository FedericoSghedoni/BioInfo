#!/bin/bash
#SBATCH --job-name=cmix2_rc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --account=ai4bio2023
#SBATCH --partition=all_usr_prod
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1

python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --resume False --n_epochs 30 --model google/vit-base-patch16-224 --additional_train_transform cutmix2_rc --batch_size 16 --lr 0.0002 --weight_decay 0.01 --scheduler_kwargs num_warmup_steps=10000 --log_dir ./logp_4