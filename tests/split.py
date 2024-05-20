import pandas as pd

df = pd.read_csv("./data/rxrx1_v1.0/metadata_old.csv")

# Validation: 4 experiments, 2 sites per experiment

print(df[df['dataset'] == 'test'].experiment.unique())

for exp in ['HUVEC-24']:
    df.loc[df['experiment'] == exp, 'dataset'] = 'val'
    
df.to_csv('./data/rxrx1_v1.0/metadata.csv', index=False)

print(df[df['dataset'] == 'val'].experiment.unique())


#        import numpy as np
#        import pandas as pd
#        import os
#        embeddings = outputs.hidden_states[0][:, 0, :].squeeze(1).cpu()
#        # Carica gli embeddings dal file CSV
#        csv_file = 'image_embeddings.csv'
#        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
#            df_existing = pd.read_csv(csv_file)
#        else:
#            df_existing = pd.DataFrame()
#        # Convertire i nuovi embeddings in DataFrame
#        df_new = pd.DataFrame(embeddings)
#        df_new.columns = df_new.columns.astype(str)
#        # Aggiungi eventuali colonne necessarie (ad esempio, una colonna per l'etichetta dell'esperimento)
#        df_new['experiment'] = 'HUVEC-17'  
#        # Aggiungi i nuovi embeddings agli esistenti
#        df_combined = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
#        print(df_combined.shape)
#        # Salva tutti gli embeddings nel file CSV
#        df_combined.to_csv(csv_file, index=False)