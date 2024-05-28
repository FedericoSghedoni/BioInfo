# BioInfo

git clone --depth=1 https://github.com/FedericoSghedoni/BioInfo

per addestrare:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0

per chiedere risorse debugging:
srun -Q --immediate=100 --cpus-per-task=2 --mem=5G --account ai4bio2023 --partition=all_serial --gres=gpu:2 --time 4:00:00 --pty bash

per chiedere risorse train:
srun -Q --immediate=100 --cpus-per-task=2 --mem=5G --account ai4bio2023 --partition=all_usr_prod --gres=gpu:2 --time 24:00:00 --pty bash

per riprendere da checkpoint:
--resume True

per usare solo una frazione del dataset:
--frac 0.05

per usare vit come modello:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --model google/vit-base-patch16-224

per usare cutmix:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --model google/vit-base-patch16-224 --log_dir ./log --additional_train_transform cutmix

per ottenere gli embeddings di resnet: (va aggiunto il codice in identity.py)
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --log_dir ./logs_resnet50 --eval_only True --eval_splits val --evaluate_all_splits False --load_featurizer_only True --pretrained_model_path ./logs_resnet50/rxrx1_seed:0_epoch:best_model.pth

per addestrare testa di classificazione 'experiment' con resnet:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --dataset_kwargs label_name=experiment --split_scheme mixed-to-test --model_kwargs train_only_classifier=True --load_featurizer_only True --pretrained_model_path ./logs_resnet50/rxrx1_seed:0_epoch:best_model.pth --log_dir ./logs_resnet50_experiment

per addestrare testa di classificazione 'experiment' con vit:
python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --resume False --n_epochs 20 --model google/vit-base-patch16-224 --batch_size 16 --lr 0.0002 --weight_decay 0.01 --scheduler_kwargs num_warmup_steps=15000 --split_scheme mixed-to-test --dataset_kwargs label_name=experiment --model_kwargs train_only_classifier=True --load_featurizer_only True --pretrained_model_path ./log_vit_mu0/rxrx1_seed:0_epoch:best_model.pth --log_dir ./logs_vit_mix_mu0_experiment

#embeddings vit
 python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --resume False --n_epochs 20 --model google/vit-base-patch16-224 --batch_size 16 --lr 0.0002 --weight_decay 0.01 --scheduler_kwargs num_warmup_steps=15000 --eval_only True --eval_splits val --evaluate_all_splits False  --pretrained_model_path ./logs_vit/rxrx1_seed:0_epoch:best_model.pth --log_dir ./logs_vit

#embeddings vit Mix
  python WildsDataset/examples/run_expt.py --dataset rxrx1 --algorithm ERM --root_dir data --device 0 --resume False --n_epochs 20 --model google/vit-base-patch16-224 --batch_size 16 --lr 0.0002 --weight_decay 0.01 --scheduler_kwargs num_warmup_steps=15000 --eval_only True --eval_splits val --evaluate_all_splits False --additional_train_transform cutmix2 --pretrained_model_path ./logs_vit_mix/rxrx1_seed:0_epoch:best_model.pth --log_dir ./logs_vit_mix 

