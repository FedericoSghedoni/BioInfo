import pandas as pd

df = pd.read_csv('image_embeddings.csv')

#df = df[df['experiment'] != 'HUVEC-23']
#    
#df.to_csv("image_embeddings.csv", index=False)

print(df['experiment'].value_counts())

#df = pd.read_csv('./data/rxrx1_v1.0/metadata.csv')
#
#print(df['experiment'].value_counts())