import itertools
import subprocess

def main():
    # Definire i possibili valori degli iperparametri
    hyperparameters = {
        'weight_decay': [0.01, 0.001, 1e-5],
        'scheduler_kwargs': ['num_warmup_steps=10000', 'num_warmup_steps=5415'],
        'learning_rate': [2e-4, 1e-4],
        'additional_train_transform': ['cutmix2', 'cutmix2_rc'],
        'model_kwargs': ['pretrained=True, hidden_dropout_prob=0.1', 'pretrained=True, hidden_dropout_prob=0.01', 'pretrained=True, hidden_dropout_prob=0.001', 'pretrained=True, hidden_dropout_prob=0.0001']
    }

    # Generare tutte le combinazioni possibili di iperparametri
    hyperparameter_combinations = list(itertools.product(*hyperparameters.values()))

    # Indice dell'ultima combinazione iperparametrica testata
    last_index = 0
    
    resume = False

    # Carica l'ultimo indice testato
    with open('last_index3.txt', 'r') as f:
        last_index = int(f.read())

    if last_index != 0:
        resume = True

    # Eseguire la grid search
    for idx, hyperparams in enumerate(hyperparameter_combinations):
        if idx == last_index:
            weight_decay, scheduler_kwargs, learning_rate, additional_train_transform, model_kwargs = hyperparams
            command =   f"python WildsDataset/examples/run_expt.py " \
                        f"--dataset rxrx1 " \
                        f"--algorithm ERM " \
                        f"--root_dir data " \
                        f"--device 0 " \
                        f"--resume {resume} " \
                        f"--n_epochs 15 " \
                        f"--model google/vit-base-patch16-224 " \
                        f"--additional_train_transform {additional_train_transform} " \
                        f"--batch_size 16 " \
                        f"--lr {learning_rate} " \
                        f"--weight_decay {weight_decay} " \
                        f"--scheduler_kwargs {scheduler_kwargs} " \
                        f"--model_kwargs {model_kwargs} " \
                        f"--log_dir ./logx_dropout"
            
            # Salva l'indice su file o in una variabile persistente
            with open('last_index3.txt', 'w') as f:
                f.write(str(last_index))
            
            with open('output3.txt', 'a') as f:
                f.write(f"{idx}. Running command for hyperparameters {hyperparams}: {command}\n")
            subprocess.run(command, shell=True)
            
            last_index += 1
            resume = False
        
            
    # Azzera l'indice su file o in una variabile persistente
    with open('last_index3.txt', 'w') as f:
        f.write("0")

if __name__ == "__main__":
    main()