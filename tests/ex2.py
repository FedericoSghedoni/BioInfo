import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

# Carica l'immagine di esempio
image_path = "example.png"
image = Image.open(image_path)

# Ottieni le dimensioni dell'immagine originale
original_width, original_height = image.size

# Definisci la trasformazione RandomResizedCrop
transform = transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1))

# Applica la trasformazione all'immagine
transformed_image = transform(image)

# Ottieni le dimensioni dell'immagine trasformata
transformed_width, transformed_height = transformed_image.size

# Crea una griglia di subplots per visualizzare le immagini affiancate
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Visualizza l'immagine originale con le dimensioni nel titolo
axs[0].imshow(image)
axs[0].set_title(f'Immagine originale ({original_width}x{original_height})')

# Visualizza l'immagine trasformata con le dimensioni nel titolo
axs[1].imshow(transformed_image)
axs[1].set_title(f'Immagine trasformata ({transformed_width}x{transformed_height})')

# Nascondi i ticks sugli assi
for ax in axs:
    ax.axis('off')

plt.savefig("confronto_immagini.png")

# Mostra la griglia di subplots
plt.show()
