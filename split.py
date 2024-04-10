import os
import sys

# Aggiungi il percorso al PATH
sys.path.append("/usr/local/anaconda3/lib/python3.9/site-packages")

import pandas as pd

df = pd.read_csv("data/rxrx1_v1.0/metadata.csv")

# Validation: 4 experiments, 2 sites per experiment

print(df[df['datasets'] == 'test'])