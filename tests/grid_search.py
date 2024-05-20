import itertools
import subprocess

def main():
    # Definire i possibili valori degli iperparametri
    hyperparameters = {
        'batch_size': [16],
        'learning_rate': [1e-3, 2e-4, 1e-4],
        'weight_decay': [0.01, 0.001, 1e-5],
        'scheduler_kwargs': ['num_warmup_steps=5415', 'num_warmup_steps=10000']
    }

    # Generare tutte le combinazioni possibili di iperparametri
    hyperparameter_combinations = list(itertools.product(*hyperparameters.values()))

    # Indice dell'ultima combinazione iperparametrica testata
    last_index = 0
    
    resume = False

    # Carica l'ultimo indice testato
    with open('last_index.txt', 'r') as f:
        last_index = int(f.read())

    if last_index != 0:
        resume = True

    # Eseguire la grid search
    for idx, hyperparams in enumerate(hyperparameter_combinations):
        if idx == last_index:
            batch_size, learning_rate, weight_decay, scheduler_kwargs = hyperparams
            command =   f"python WildsDataset/examples/run_expt.py " \
                        f"--dataset rxrx1 " \
                        f"--algorithm ERM " \
                        f"--root_dir data " \
                        f"--device 0 " \
                        f"--resume {resume} " \
                        f"--model google/vit-base-patch16-224 " \
                        f"--batch_size {batch_size} " \
                        f"--lr {learning_rate} " \
                        f"--weight_decay {weight_decay} " \
                        f"--scheduler_kwargs {scheduler_kwargs}"
            
            # Salva l'indice su file o in una variabile persistente
            with open('last_index.txt', 'w') as f:
                f.write(str(last_index))
            
            with open('output.txt', 'a') as f:
                f.write(f"{idx}. Running command for hyperparameters {hyperparams}: {command}\n")
            subprocess.run(command, shell=True)
            
            last_index += 1
            resume = False
        
            
    # Azzera l'indice su file o in una variabile persistente
    with open('last_index.txt', 'w') as f:
        f.write("0")

if __name__ == "__main__":
    main()

'''
Aggiungere Dropout (in vit.py, e come parametro nella chiamata a run_exp.py)
'''