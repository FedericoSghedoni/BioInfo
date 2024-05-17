import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms

# Caricare gli embeddings dal file CSV
df = pd.read_csv('image_embeddings.csv')

# Separare gli embeddings e le etichette
embeddings = df.drop(columns=['label']).values
labels = df['label'].values

# Eseguire UMAP sugli embeddings
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding_umap = reducer.fit_transform(embeddings)

# Creare un DataFrame per i risultati UMAP
df_umap = pd.DataFrame(embedding_umap, columns=['UMAP1', 'UMAP2'])
df_umap['label'] = labels

# Funzione per creare jointplot
import seaborn as sns
import matplotlib.pyplot as plt

def create_jointplot(data, title):
    jp = sns.jointplot(x='UMAP1', y='UMAP2', data=data, hue='label', palette='tab10', height=7, marginal_kws=dict(bins=50, fill=True))
    jp.fig.suptitle(title)
    jp.fig.tight_layout()
    jp.fig.subplots_adjust(top=0.95)  # Adatta il titolo
    return jp

# Creare la visualizzazione UMAP
plt.figure(figsize=(15, 5))
create_jointplot(df_umap, 'UMAP of Image Embeddings by Cell Type and Experiment')
plt.tight_layout()
plt.show()
