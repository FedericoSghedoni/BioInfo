import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Definisci i parametri di interesse
params_of_interest = ['Model', 'Optimizer', 'Transform', 'Additional train transform', 'Scheduler', 'Dropout', 'Batch size', 'Lr', 'Weight decay', 'Scheduler kwargs']

# Inizializza un dizionario per memorizzare i valori dei parametri
param_values = {param: [] for param in params_of_interest}

# Percorri tutte le cartelle "logs"
logs_folders = []
for log in ['logs', 'logg', 'logc', 'logp']:
    logs_folders = logs_folders + [folder for folder in os.listdir('.') if log in folder and os.path.isdir(os.path.join('.', folder))]
    
for folder in logs_folders:
    log_file = os.path.join(folder, 'log.txt')
    
    # Leggi le prime 88 righe del file log.txt
    with open(log_file, 'r') as f:
        lines = f.readlines()[:88]

    # Estrai i valori dei parametri di interesse
    for param in params_of_interest:
        found = False
        for line in lines:
            if param in line:
                value = line.split(': ')[-1].strip().strip('}')
                if param == 'Scheduler':
                    value = value.split('_')[0]
                if value == 'google/vit-base-patch16-224':
                    value = 'vit'
                param_values[param].append(value)
                found = True
                break
        if not found:
            param_values[param].append('None')

# Crea il DataFrame
df = pd.DataFrame(param_values)

df = df.transpose()

# Rinomina le colonne con i nomi delle cartelle "logs"
df.columns = logs_folders

# setta numero valori da leggere da ogni file
n_val = 30
# Inizializza un dizionario per memorizzare i valori degli iperparametri
param_values = {}
param_values.update({f'{i}_ac_id': [] for i in range(len(logs_folders))})
param_values.update({f'{i}_ac_test': [] for i in range(len(logs_folders))})
param_values.update({f'{i}_ac_train': [] for i in range(len(logs_folders))})

# Percorri tutte le cartelle "logs"
for idx, folder in enumerate(logs_folders):
    id_test_eval_file = os.path.join(folder, 'id_test_eval.csv')
    test_eval_file = os.path.join(folder, 'test_eval.csv')
    train_eval_file = os.path.join(folder, 'train_eval.csv')

    # Prova a leggere i primi n_val valori di acc_avg da ciascun file
    try:
        id_test_eval_df = pd.read_csv(id_test_eval_file)
        test_eval_df = pd.read_csv(test_eval_file)
        train_eval_df = pd.read_csv(train_eval_file)
    except pd.errors.EmptyDataError:
        continue

    param_values[f'{idx}_ac_id'] = id_test_eval_df['acc_avg'].head(n_val).apply(lambda x: round(x*100, 2)).tolist()
    param_values[f'{idx}_ac_test'] = test_eval_df['acc_avg'].head(n_val).apply(lambda x: round(x*100, 2)).tolist()
    param_values[f'{idx}_ac_train'] = train_eval_df['acc_avg'].head(n_val).apply(lambda x: round(x*100, 2)).tolist()
    
    # Aggiungi valori vuoti se una colonna ha meno di n_val valori
    for key in param_values:
        if len(param_values[key]) < n_val:
            num_missing_values = n_val - len(param_values[key])
            param_values[key] += [None] * num_missing_values

# Crea il DataFrame
eval_df = pd.DataFrame(param_values)

# Rinomina le colonne con i nomi delle cartelle "logs"
eval_df.columns = param_values.keys()

# Definire una funzione di ordinamento personalizzata che considera l'ordine numerico
def custom_sort(label):
    # Estrarre il numero dall'etichetta della colonna
    numero = int(label.split('_')[0])  # supponendo che il numero sia nella terza parte dell'etichetta
    return numero

# Definire l'ordine delle colonne
desired_order = sorted([col for col in eval_df.columns], key=custom_sort)

# Riordinare le colonne
eval_df = eval_df.reindex(columns=desired_order)


def render_mpl_table(data, col_width=0.1, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#fdf1e0', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], rowLabels =False, header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    if rowLabels:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)
    else:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    # Coloriamo alternativamente le colonne
    num_cols = len(data.columns)
    for i in range(header_columns, num_cols):
        color = row_colors[i % len(row_colors)]
        for cell in mpl_table._cells:
            if cell[1] == -1:
                mpl_table._cells[cell].set_text_props(weight='bold', color='black')
            if cell[1] == i:
                mpl_table._cells[cell].set_facecolor(color)
            
    return ax

# Impostazione della figura e delle tabelle
fig, axs = plt.subplots(2)

# Eliminazione dello spazio tra le tabelle
plt.subplots_adjust(hspace=0)

# Ingrandimento dell'intera figura
fig.set_size_inches(70, 25)

for ax in axs:
    ax.axis('off')

# Disegno delle tabelle
render_mpl_table(df, header_columns=0, ax=axs[0], font_size=14, rowLabels=True, row_colors=['#faf7e5','#f3fae5'])
render_mpl_table(eval_df, header_columns=0, ax=axs[1], font_size=9, rowLabels=True, row_colors=['#faf7e5','w','#faf7e5','#f3fae5','w','#f3fae5'])

# Impostazione del nome degli indici sopra la colonna degli indici
#axs[1].text(-0.5, 1.1, 'Labels', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)

#plt.tight_layout()
plt.savefig('table_image.png', dpi=300)
plt.show()


