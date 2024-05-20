import pandas as pd
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms


class CutMix:
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha

    def getitem(self):
        #img1, target1 = self.dataset[index]
        #img2, target2 = self.random_sample(index)

        img1, target1 = self.dataset[0]
        img2, target2 = self.dataset[1]

        # Apply CutMix
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img1.size, lam)
        img1.paste(img2.crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1))
        img1.save('image_cut.png')
        
        # Adjust the target label
        target = target1 * lam + target2 * (1. - lam)

        return img1, target

    def random_sample(self, index):
        while True:
            idx = np.random.randint(len(self.dataset))
            if idx != index and self.dataset.targets[idx] == self.dataset.targets[index]:
                return self.dataset[idx]

    def rand_bbox(self, size, lam):
        W, H = size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

df = pd.read_csv('/homes/fsghedoni/BioInfo/data/rxrx1_v1.0/metadata.csv')

# Raggruppa le istanze per etichetta "sirna_id"
grouped_by_sirna = df.groupby('sirna_id')

# Seleziona tutte le coppie con la stessa etichetta "sirna_id"
#for sirna_id, group_df in grouped_by_sirna:
#    if len(group_df) >= 2:  # Se ci sono almeno due istanze con la stessa etichetta
#        # Seleziona tutte le coppie di istanze con la stessa etichetta
#        for index1, row1 in group_df.iterrows():
#            for index2, row2 in group_df.iterrows():
#                if index1 != index2:  # Assicurati che le righe siano diverse
#                    print("Image 1:")
#                    print(row1)
#                    print("Image 2:")
#                    print(row2)
#                    print("----")

image1 = Image.open('/homes/fsghedoni/BioInfo/data/rxrx1_v1.0/images/HEPG2-08/Plate1/B05_s1.png')
image1.save('image1.png')
image2 = Image.open('/homes/fsghedoni/BioInfo/data/rxrx1_v1.0/images/HEPG2-10/Plate1/L10_s1.png')
image2.save('image2.png')

# Definisci le etichette per le due immagini
label1 = 836  # Etichetta per image1
label2 = 836  # Etichetta per image2

# Crea un dataset artificiale con le due immagini e le loro etichette
dataset = [(image1, label1), (image2, label2)]
cutmix_dataset = CutMix(dataset)

cutmix_dataset.getitem()

'''
    # cutmix augmentation
    if args.cutmix != 0:
        perm = torch.randperm(X.shape[0]).cuda()
        img_height, img_width = X.size()[2:]
        lambd = np.random.beta(args.cutmix, args.cutmix)

        column = np.random.uniform(0, img_width)
        row = np.random.uniform(0, img_height)
        height = (1 - lambd) ** 0.5 * img_height
        width = (1 - lambd) ** 0.5 * img_width

        r1 = round(max(0, row - height / 2))
        r2 = round(min(img_height, row + height / 2))
        c1 = round(max(0, column - width / 2))
        c2 = round(min(img_width, column + width / 2))

        if r1 < r2 and c1 < c2:
            X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]
            lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)
            Y = Y * lambd + Y[perm] * (1 - lambd)
                     
Questo codice implementa l'aumentazione dei dati utilizzando la tecnica CutMix. Ecco una spiegazione linea per linea:

1. `if args.cutmix != 0:`: Controlla se l'argomento `cutmix` è diverso da zero, il che indica che l'aumentazione CutMix deve essere applicata.

2. `perm = torch.randperm(X.shape[0]).cuda()`: Genera una permutazione casuale degli indici delle immagini nel batch e sposta i risultati sulla GPU (se disponibile).

3. `img_height, img_width = X.size()[2:]`: Ottiene l'altezza e la larghezza delle immagini nel batch.

4. `lambd = np.random.beta(args.cutmix, args.cutmix)`: Genera un valore casuale da una distribuzione beta, utilizzando `args.cutmix` come parametro della distribuzione beta. Questo valore determina la probabilità di eseguire l'aumentazione CutMix per ciascuna immagine nel batch.

5. `column = np.random.uniform(0, img_width)`: Genera casualmente la coordinata x del punto di inizio della maschera CutMix.

6. `row = np.random.uniform(0, img_height)`: Genera casualmente la coordinata y del punto di inizio della maschera CutMix.

7. `height = (1 - lambd) ** 0.5 * img_height`: Calcola l'altezza della maschera CutMix.

8. `width = (1 - lambd) ** 0.5 * img_width`: Calcola la larghezza della maschera CutMix.

9. `r1 = round(max(0, row - height / 2))`: Calcola la coordinata y dell'angolo superiore sinistro della maschera CutMix.

10. `r2 = round(min(img_height, row + height / 2))`: Calcola la coordinata y dell'angolo inferiore destro della maschera CutMix.

11. `c1 = round(max(0, column - width / 2))`: Calcola la coordinata x dell'angolo superiore sinistro della maschera CutMix.

12. `c2 = round(min(img_width, column + width / 2))`: Calcola la coordinata x dell'angolo inferiore destro della maschera CutMix.

13. `if r1 < r2 and c1 < c2:`: Controlla se la maschera CutMix non si sovrappone ai bordi dell'immagine.

14. `X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]`: Applica l'aumentazione CutMix sostituendo una parte dell'immagine originale con una parte di un'altra immagine presa casualmente dallo stesso batch.

15. `lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)`: Calcola il valore di `lambd` per normalizzare le etichette corrispondenti.

16. `Y = Y * lambd + Y[perm] * (1 - lambd)`: Normalizza le etichette utilizzando `lambd`.
'''
