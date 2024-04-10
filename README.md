# BioInfo

git clone --depth=1 https://github.com/FedericoSghedoni/BioInfo
git submodule init
git submodule update

-Accedere alla vpn
-Connessione SSH


python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --process_output_function multiclass_logits_to_pred

srun -Q --partition=students-dev --gres=gpu:2 --pty bash
srun -Q --immediate=100 --cpus-per-task=2 --mem=5G --partition=all_serial --gres=gpu:2 --time 4:00:00 --pty bash