import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
from PIL import Image

def initialize_vit_transform(is_training):
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    def featurizer(x: torch.Tensor) -> torch.Tensor:
        # Utilizza ViTFeatureExtractor per pre-elaborare l'immagine
        inputs = feature_extractor(images=x, return_tensors="pt")
        return inputs
    t_featurizer = transforms.Lambda(lambda x: featurizer(x))
    
    if is_training:
        transforms_ls = [
            #t_random_rotation,
            #transforms.RandomHorizontalFlip(),
            #transforms.Resize((224, 224)),
            #transforms.ToTensor(),
            t_featurizer,
            #t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform



# return_tensors="pt"

pipe = False

# Carica il modello ViT pre-addestrato
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Carica l'estrattore di caratteristiche
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Carica e pre-elabora un'immagine
image = Image.open("example.png")

if pipe:
    transform = initialize_vit_transform(is_training=True)
    inputs = transform(image)
    print(type(inputs))
    input_ids_size = inputs["pixel_values"].size()
    print("Dimensioni del tensore inputs:", input_ids_size)
else:
    inputs = feature_extractor(images=image, return_tensors="pt")
    print(type(inputs))


# Classifica l'immagine
outputs = model(**inputs)
print(outputs, '\n', outputs.logits.size())
logits = outputs.logits
# Ottieni le predizioni
predicted_class_idx = logits.argmax(-1).item()
print(predicted_class_idx)
