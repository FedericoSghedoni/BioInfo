from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
from PIL import Image

class ViTClassifier(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        #self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
          
    def __call__(self, x):
        x["pixel_values"] = x["pixel_values"].squeeze(1) 
        outputs = super().__call__(**x, output_hidden_states=True) # False se non serve leggere gli embeddings
        
        #import numpy as np
        #import pandas as pd
        #import os
        #embeddings = outputs.hidden_states[0][:, 0, :].squeeze(1).cpu()
        ## Carica gli embeddings dal file CSV
        #csv_file = 'image_embeddings.csv'
        #if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        #    df_existing = pd.read_csv(csv_file)
        #else:
        #    df_existing = pd.DataFrame()
        ## Convertire i nuovi embeddings in DataFrame
        #df_new = pd.DataFrame(embeddings)
        #df_new.columns = df_new.columns.astype(str)
        ## Aggiungi eventuali colonne necessarie (ad esempio, una colonna per l'etichetta dell'esperimento)
        #df_new['experiment'] = 'HUVEC-17'  
        ## Aggiungi i nuovi embeddings agli esistenti
        #df_combined = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
        #print(df_combined.shape)
        ## Salva tutti gli embeddings nel file CSV
        #df_combined.to_csv(csv_file, index=False)

        return outputs.logits

