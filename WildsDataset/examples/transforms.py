import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast

from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform_name=None, **transform_kwargs
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None
    elif transform_name == "bert":
        return initialize_bert_transform(config)
    elif transform_name == 'rxrx1':
        return initialize_rxrx1_transform(is_training)
    elif transform_name == 'vit':
        return initialize_vit_transform(is_training, dataset, additional_transform_name)

    # For images
    normalize = True
    if transform_name == "image_base":
        transform_steps = get_image_base_transform_steps(config, dataset)
    elif transform_name == "image_resize":
        transform_steps = get_image_resize_transform_steps(
            config, dataset
        )
    elif transform_name == "image_resize_and_center_crop":
        transform_steps = get_image_resize_and_center_crop_transform_steps(
            config, dataset
        )
    elif transform_name == "poverty":
        if not is_training:
            return None
        transform_steps = []
        normalize = False
    else:
        raise ValueError(f"{transform_name} not recognized")

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )
    if additional_transform_name == "fixmatch":
        if transform_name == 'poverty':
            transformations = add_poverty_fixmatch_transform(config, dataset, transform_steps)
        else:
            transformations = add_fixmatch_transform(
                config, dataset, transform_steps, default_normalization
            )
        transform = MultipleTransforms(transformations)
    elif additional_transform_name == "randaugment":
        if transform_name == 'poverty':
            transform = add_poverty_rand_augment_transform(
                config, dataset, transform_steps
            )
        else:
            transform = add_rand_augment_transform(
                config, dataset, transform_steps, default_normalization
            )
    elif additional_transform_name == "weak":
        transform = add_weak_transform(
            config, dataset, transform_steps, normalize, default_normalization
        )
    else:
        if transform_name != "poverty":
            # The poverty data is already a tensor at this point
            transform_steps.append(transforms.ToTensor())
        if normalize:
            transform_steps.append(default_normalization)
        transform = transforms.Compose(transform_steps)

    return transform


def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def initialize_rxrx1_transform(is_training):
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

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

def initialize_vit_transform(is_training, dataset, additional_transform_name):

    def detuple(x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, tuple):
            img, metadata = x
        else:
            img = x
        return img
    t_detuple = transforms.Lambda(lambda x: detuple(x))
    
    def cutmix(x: torch.Tensor) -> torch.Tensor:
        img, metadata = x

        # Filtra gli indici del dataset di train
        train_indices = np.where(dataset._split_array == dataset.split_dict['train'])[0]
        # Filtra gli indici escludendo l'esperimento di img
        experiment_indices = np.where(dataset._metadata_array[:, 1] != metadata[1])[0]
        # Filtra gli indici in base alla label y
        label_indices = np.where(dataset._y_array == metadata[5])[0]
        # Interseca tutti gli insiemi di indici per ottenere gli indici che soddisfano tutte le condizioni
        desired_indices = np.intersect1d(train_indices, experiment_indices)
        desired_indices = np.intersect1d(desired_indices, label_indices)
        # Scegli casualmente uno di questi indici
        desired_index = np.random.choice(desired_indices)
        # Ottieni l'immagine corrispondente all'indice desiderato
        img2 = dataset.get_input(desired_index)

        lam = np.random.beta(1, 1)
        W, H = img.size
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
        
        # Apply CutMix
        img.paste(img2.crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1))
        
        if 'cutmix_rc' in additional_transform_name:
            return (img, metadata)
        else:
            return img
    t_cutmix = transforms.Lambda(lambda x: cutmix(x))
    
    def cutmix2(x: torch.Tensor) -> torch.Tensor:
        img, metadata = x

        # Filtra gli indici del dataset di train
        train_indices = np.where(dataset._split_array == dataset.split_dict['train'])[0]
        # Filtra gli indici escludendo l'esperimento di img
        experiment_indices = np.where(dataset._metadata_array[:, 1] != metadata[1])[0]
        # Filtra gli indici in base alla label y
        label_indices = np.where(dataset._y_array == metadata[5])[0]
        # Interseca tutti gli insiemi di indici per ottenere gli indici che soddisfano tutte le condizioni
        desired_indices = np.intersect1d(train_indices, experiment_indices)
        desired_indices = np.intersect1d(desired_indices, label_indices)
        # Scegli casualmente uno di questi indici
        desired_index = np.random.choice(desired_indices)
        # Ottieni l'immagine corrispondente all'indice desiderato
        img2 = dataset.get_input(desired_index)

        lam = np.random.beta(1, 1)
        W, H = img.size
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
        
        # Apply CutMix
        img.paste(img2.crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1))

        return img
    t_cutmix2 = transforms.Lambda(lambda x: cutmix2(x))
    
    def mixup(x: torch.Tensor) -> torch.Tensor:
        img, metadata = x

        # Filtra gli indici del dataset di train
        train_indices = np.where(dataset._split_array == dataset.split_dict['train'])[0]
        # Filtra gli indici escludendo l'esperimento di img
        experiment_indices = np.where(dataset._metadata_array[:, 1] != metadata[1])[0]
        # Filtra gli indici in base alla label y
        label_indices = np.where(dataset._y_array == metadata[5])[0]
        # Interseca tutti gli insiemi di indici per ottenere gli indici che soddisfano tutte le condizioni
        desired_indices = np.intersect1d(train_indices, experiment_indices)
        desired_indices = np.intersect1d(desired_indices, label_indices)
        # Scegli casualmente uno di questi indici
        desired_index = np.random.choice(desired_indices)
        # Ottieni l'immagine corrispondente all'indice desiderato
        img2 = dataset.get_input(desired_index)

        lam = np.random.beta(1, 1)
        img = TF.to_tensor(img)
        img2 = TF.to_tensor(img2)
        
        # Apply MixUp
        mixed_img = lam * img + (1 - lam) * img2
        mixed_img = TF.to_pil_image(mixed_img)

        return mixed_img
    t_mixup = transforms.Lambda(lambda x: mixup(x))
    
    class t_randomapplyone:
        def __init__(self, prob):
            self.prob = prob
            
        def __call__(self, x):
            if np.random.rand() < self.prob:
                x = t_cutmix(x)
            if np.random.rand() < self.prob:
                x = t_mixup(x)
            if self.prob == 0:
                transform = np.random.choice([t_cutmix, t_mixup])
                x = transform(x)
            return x
        
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

    if  is_training and additional_transform_name == 'cutmix':
        transforms_ls = [
            t_randomapplyone(prob=0),
            t_detuple,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif  is_training and additional_transform_name == 'cutmix2':
        transforms_ls = [
            t_cutmix2,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif  is_training and additional_transform_name == 'cutmix_rc':
        transforms_ls = [
            t_randomapplyone(prob=0.5),
            t_detuple,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif  is_training and additional_transform_name == 'cutmix_rc2':
        transforms_ls = [
            t_randomapplyone(prob=1),
            t_detuple,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif  is_training and additional_transform_name == 'cutmix2_rc':
        transforms_ls = [
            t_cutmix2,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif  is_training and additional_transform_name == 'cutmix2_rc2':
        transforms_ls = [
            t_cutmix2,
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1)),
            transforms.ToTensor(),
            t_standardize,
        ]
    elif is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            t_detuple,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform
        

def get_image_base_transform_steps(config, dataset) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config.target_resolution is not None:
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_resize_transform_steps(config, dataset)
    target_resolution = _get_target_resolution(config, dataset)
    transform_steps.append(
        transforms.CenterCrop(target_resolution),
    )
    return transform_steps


def get_image_resize_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    return [
        transforms.Resize(scaled_resolution)
    ]

def add_fixmatch_transform(config, dataset, base_transform_steps, normalization):
    return (
        add_weak_transform(config, dataset, base_transform_steps, True, normalization),
        add_rand_augment_transform(config, dataset, base_transform_steps, normalization)
    )

def add_poverty_fixmatch_transform(config, dataset, base_transform_steps):
    return (
        add_weak_transform(config, dataset, base_transform_steps, False, None),
        add_poverty_rand_augment_transform(config, dataset, base_transform_steps)
    )

def add_weak_transform(config, dataset, base_transform_steps, should_normalize, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution,
            ),
        ]
    )
    if should_normalize:
        weak_transform_steps.append(transforms.ToTensor())
        weak_transform_steps.append(normalization)
    return transforms.Compose(weak_transform_steps)

def add_rand_augment_transform(config, dataset, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)

def poverty_rgb_color_transform(ms_img, transform):
    from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
    poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

    def unnormalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
        return result

    def normalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
        return ms_img

    color_transform = transforms.Compose([
        transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
        transform,
        transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
    ])
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
    return ms_img

def add_poverty_rand_augment_transform(config, dataset, base_transform_steps):
    def poverty_color_jitter(ms_img):
        return poverty_rgb_color_transform(
            ms_img,
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1))

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.Lambda(lambda ms_img: ms_cutout(ms_img)),
        # transforms.Lambda(lambda ms_img: viz(ms_img)),
    ])

    return transforms.Compose(strong_transform_steps)

def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)
