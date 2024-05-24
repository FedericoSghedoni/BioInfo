import itertools
import subprocess
import time

def main():
    # Definire i possibili valori degli iperparametri
    hyperparameters = {
        'weight_decay': [0.01],
        'scheduler_kwargs': ['num_warmup_steps=10000', 'num_warmup_steps=5415', 'num_warmup_steps=13415'],
        'learning_rate': [2e-4, 1e-4],
        'additional_train_transform': ['cutmix2'],
        'model_kwargs': ['pretrained=True, hidden_dropout_prob=0.01', 'pretrained=True, hidden_dropout_prob=0.001', 'pretrained=True, hidden_dropout_prob=0.0001']
    }
    
    # file index
    file_index = 4

    # Generare tutte le combinazioni possibili di iperparametri
    hyperparameter_combinations = list(itertools.product(*hyperparameters.values()))

    # Indice dell'ultima combinazione iperparametrica testata
    last_index = 0
    
    resume = False

    # Carica l'ultimo indice testato
    with open(f'last_index{file_index}.txt', 'r') as f:
        last_index = int(f.read())

    if last_index != 0:
        resume = False
        
    # Segnare l'ora di inizio
    start_time = time.time()
    max_duration = 16 * 60 * 60  # 16 ore in secondi

    # Eseguire la grid search
    for idx, hyperparams in enumerate(hyperparameter_combinations):
        # Controlla se sono passate più di 16 ore
        elapsed_time = time.time() - start_time
        if elapsed_time > max_duration:
            break
        
        if idx == last_index:
            weight_decay, scheduler_kwargs, learning_rate, additional_train_transform, model_kwargs = hyperparams
            command =   f"python WildsDataset/examples/run_expt.py " \
                        f"--dataset rxrx1 " \
                        f"--algorithm ERM " \
                        f"--root_dir data " \
                        f"--device 0 " \
                        f"--resume {resume} " \
                        f"--n_epochs 20 " \
                        f"--model google/vit-base-patch16-224 " \
                        f"--additional_train_transform {additional_train_transform} " \
                        f"--batch_size 16 " \
                        f"--lr {learning_rate} " \
                        f"--weight_decay {weight_decay} " \
                        f"--scheduler_kwargs {scheduler_kwargs} " \
                        f"--model_kwargs {model_kwargs} " \
                        f"--log_dir ./logy_dropout"
            
            with open(f'output{file_index}.txt', 'a') as f:
                f.write(f"{idx}. Running command for hyperparameters {hyperparams}: {command}\n")
            subprocess.run(command, shell=True)
            
            resume = False
            last_index += 1
            
            # Salva l'indice su file o in una variabile persistente
            with open(f'last_index{file_index}.txt', 'w') as f:
                f.write(str(last_index))

if __name__ == "__main__":
    main()