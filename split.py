import pandas as pd

df = pd.read_csv("metadata.csv")

# Validation: 4 experiments, 2 sites per experiment

print(df[df['dataset'] == 'test'].experiment.unique())

for exp in ['HEPG2-08', 'HUVEC-20', 'RPE-09', 'U2OS-05']:
    df.loc[df['experiment'] == exp, 'dataset'] = 'val'
    
df.to_csv('metadata_val.csv', index=False)

print(df[df['dataset'] == 'val'].experiment.unique())