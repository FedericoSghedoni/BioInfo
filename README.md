# BioInfo

git clone --depth=1 https://github.com/FedericoSghedoni/BioInfo
git submodule init
git submodule update

-Accedere alla vpn
-Connessione SSH

per addestrare:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0

per chiedere risorse debugging:
srun -Q --immediate=100 --cpus-per-task=2 --mem=5G --account ai4bio2023 --partition=all_serial --gres=gpu:2 --time 4:00:00 --pty bash

per chiedere risorse train:
srun -Q --immediate=100 --cpus-per-task=2 --mem=5G --account ai4bio2023 --partition=all_usr_prod --gres=gpu:2 --time 24:00:00 --pty bash

per riprendere da checkpoint:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --resume True

per usare solo una frazione del dataset:
--frac 0.05

per cambiare dir di log:
--log_dir ./log

per usare vit come modello:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --model google/vit-base-patch16-224

per staccare il terminale:
screen
CTRL+A CTRL*D

per riattaccare il terminale:
screen -r

per chiudere definitivamente il terminale:
exit

per provare cutmix:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --model google/vit-base-patch16-224 --log_dir ./log --additional_train_transform cutmix

per ottenere gli embeddings di resnet: (va aggiunto il codice in identity.py)
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --log_dir ./logs_resnet50 --eval_only True --eval_splits val --evaluate_all_splits False --load_featurizer_only True --pretrained_model_path ./logs_resnet50/rxrx1_seed:0_epoch:best_model.pth


python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --dataset_kwargs label_name=experiment --model_kwargs train_only_classifier=True --log_dir ./ciao
