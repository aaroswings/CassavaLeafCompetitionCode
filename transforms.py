import albumentations as A
import albumentations.augmentations.transforms as transforms
import albumentations.augmentations.crops.transforms as crops
from albumentations.augmentations.geometric import rotate as rotate
from albumentations.augmentations.geometric import transforms as geometric
from albumentations.imgaug import transforms as imgaug
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import math
import numpy as np
import pandas as pd

config = pd.read_json('train_config.json', typ='series')
TARGET_SIZE = int(config['target_size'])
if 'target_size_valid' in config.keys():
    TARGET_SIZE_VALID = config['target_size_valid']
else:
    TARGET_SIZE_VALID = TARGET_SIZE

def get_train_transforms():
    return A.Compose([
        transforms.PadIfNeeded(600, 800),
        geometric.ShiftScaleRotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99, scale_limit=0.8),
        geometric.Perspective(pad_mode=cv2.BORDER_REFLECT,interpolation=cv2.INTER_AREA),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(config['rrc_scale_min'], config['rrc_scale_max']),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        transforms.Transpose(),
        transforms.Flip(),
        transforms.CoarseDropout(),
        transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.3,
            p=.7),
        transforms.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=60,
            val_shift_limit=0, 
            p=0.6),
        transforms.RandomGridShuffle(p=0.1),
        A.OneOf([
                transforms.MotionBlur(blur_limit=(3, 9)),
                transforms.GaussianBlur(),
                transforms.MedianBlur()
            ], p=0.2),
        transforms.ISONoise(p=.3),
        transforms.GaussNoise(var_limit=255., p=.3),
        A.OneOf([
            transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
            transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=100, alpha_affine=100, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])

def get_presize_combine_transforms():
    transforms_presize = A.Compose([
        transforms.Transpose(),
        transforms.Flip(),
        #transforms.PadIfNeeded(600, 800),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(config['rrc_scale_min'], config['rrc_scale_max']),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99),
    ])
    
    transforms_postsize = A.Compose([
        geometric.Perspective(pad_mode=cv2.BORDER_REFLECT,interpolation=cv2.INTER_AREA),
        transforms.CoarseDropout(),
        transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.5,
            p=.7),
        transforms.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=60,
            val_shift_limit=0, 
            p=0.6),
        transforms.Equalize(p=0.05),
        transforms.FancyPCA(p=0.05),
        transforms.RandomGridShuffle(p=0.1),
        A.OneOf([
                transforms.MotionBlur(blur_limit=(3, 9)),
                transforms.GaussianBlur(),
                transforms.MedianBlur()
            ], p=0.2),
        transforms.ISONoise(p=.3),
        transforms.GaussNoise(var_limit=127., p=.3),
        A.OneOf([
            transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
            transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=4, alpha_affine=4, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])
    return transforms_presize, transforms_postsize

def get_presize_combine_transforms_V2():
    transforms_presize = A.Compose([
        transforms.Transpose(),
        transforms.Flip(),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(config['rrc_scale_min'], config['rrc_scale_max']),
            ratio=(.75, 1.6),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99),
    ])
    
    transforms_postsize = A.Compose([
        geometric.Perspective(
            scale=.2,
            pad_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_AREA, p = 0.1),
        transforms.CoarseDropout(),
        transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1, p=0.3),
        transforms.RandomToneCurve(scale=.1, by_channels=True, p=0.2),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.3,
            p=.7),
        transforms.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20,
            val_shift_limit=0, 
            p=0.5),
        transforms.Equalize(p=0.05),
        transforms.FancyPCA(p=0.05),
        transforms.RandomGridShuffle(p=0.1),
        A.OneOf([
                transforms.MotionBlur(blur_limit=(3, 9)),
                transforms.GaussianBlur(),
                transforms.MedianBlur()
            ], p=0.1),
        transforms.ISONoise(p=.3),
        transforms.GaussNoise(var_limit=127., p=.3),
        A.OneOf([
            transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
            transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=4, alpha_affine=4, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])
    return transforms_presize, transforms_postsize

def get_presize_combine_transforms_V3():
    transforms_presize = A.Compose([
        transforms.Transpose(),
        geometric.Perspective(
            scale=[0, .1],
            pad_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_AREA, p = .3),
        transforms.Flip(),
        transforms.PadIfNeeded(600, 800),
        geometric.ShiftScaleRotate(interpolation=cv2.INTER_LANCZOS4, p = 0.95, scale_limit=0.0),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(config['rrc_scale_min'], config['rrc_scale_max']),
            ratio=(.75, 1.6),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        #rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99),
    ])
    
    transforms_postsize = A.Compose([
        #imgaug.IAAPiecewiseAffine(),

        transforms.CoarseDropout(),
        transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1, by_channels=True, p=0.2),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.4,
            p=.8),
        transforms.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=30,
            val_shift_limit=0, 
            p=0.5),
        #transforms.Equalize(p=0.05),
        #transforms.FancyPCA(p=0.05),
        transforms.RandomGridShuffle(p=0.1),
        A.OneOf([
                transforms.MotionBlur(blur_limit=(3, 9)),
                transforms.GaussianBlur(),
                transforms.MedianBlur()
            ], p=0.1),
        transforms.ISONoise(p=.2),
        transforms.GaussNoise(var_limit=127., p=.3),
        A.OneOf([
            transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
            transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=4, alpha_affine=4, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])
    return transforms_presize, transforms_postsize

def get_presize_combine_transforms_V4():
    transforms_presize = A.Compose([
        transforms.PadIfNeeded(600, 800),
        geometric.Perspective(
            scale=[0, .1],
            pad_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_AREA, p = .3),
        transforms.Flip(),
        geometric.ShiftScaleRotate(interpolation=cv2.INTER_LANCZOS4, p = 0.95, scale_limit=0.0),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(config['rrc_scale_min'], config['rrc_scale_max']),
            ratio=(.70, 1.4),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        transforms.Transpose()
        #rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99),
    ])
    
    transforms_postsize = A.Compose([
        #imgaug.IAAPiecewiseAffine(),

        transforms.CoarseDropout(),
        transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1, p=0.2),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.4,
            p=.8),
        transforms.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=50,
            val_shift_limit=0, 
            p=0.5),
        transforms.Equalize(p=0.05),
        transforms.FancyPCA(p=0.05),
        transforms.RandomGridShuffle(p=0.1),
        A.OneOf([
                transforms.MotionBlur(blur_limit=(3, 9)),
                transforms.GaussianBlur(),
                transforms.MedianBlur()
            ], p=0.1),
        transforms.ISONoise(p=.2),
        transforms.GaussNoise(var_limit=127., p=.3),
        A.OneOf([
            transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
            transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=4, alpha_affine=4, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])
    return transforms_presize, transforms_postsize

def get_presize_combine_tune_transforms():
    transforms_presize = A.Compose([
        transforms.Transpose(),
        transforms.Flip(),
        #transforms.PadIfNeeded(600, 800),
        crops.RandomResizedCrop(
            TARGET_SIZE, TARGET_SIZE,
            scale=(.75, 1),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.99),
    ])
    
    transforms_postsize = A.Compose([
        transforms.CoarseDropout(),
        # transforms.CLAHE(p=.1),
        transforms.RandomToneCurve(scale=.1),
        transforms.RandomBrightnessContrast(
            brightness_limit=.1, 
            contrast_limit=0.2,
            p=.7),
        transforms.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=60,
            val_shift_limit=0, 
            p=0.6),
        #transforms.Equalize(p=0.1),
        #transforms.FancyPCA(p=0.05),
        #transforms.RandomGridShuffle(p=0.1),
        #A.OneOf([
        #        transforms.MotionBlur(blur_limit=(3, 9)),
        #        transforms.GaussianBlur(),
        #        transforms.MedianBlur()
        #    ], p=0.2),
        transforms.ISONoise(p=.3),
        transforms.GaussNoise(var_limit=255., p=.3),
        #A.OneOf([
        #     transforms.GridDistortion(interpolation=cv2.INTER_AREA, distort_limit=[0.7, 0.7], p=0.5),
        #     transforms.OpticalDistortion(interpolation=cv2.INTER_AREA, p=.3),
        # ], p=.3),
        geometric.ElasticTransform(alpha=4, sigma=100, alpha_affine=100, interpolation=cv2.INTER_AREA, p=0.3),
        transforms.CoarseDropout(),
        transforms.Normalize(),
        ToTensorV2()
    ])
    return transforms_presize, transforms_postsize

def get_valid_transforms():
    return A.Compose([
        transforms.Transpose(),
        transforms.PadIfNeeded(600, 800),
        rotate.Rotate(interpolation=cv2.INTER_LANCZOS4, p = 0.90),
        crops.RandomResizedCrop(
            TARGET_SIZE_VALID, TARGET_SIZE_VALID,
            scale=(.75, 1),
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
        transforms.Flip(),
        #transforms.RandomToneCurve(scale=.1),
        #transforms.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.3, p=.7),
        #transforms.HueSaturationValue(hue_shift_limit=10, 
        #                           sat_shift_limit=10,
        #                           val_shift_limit=5, p=0.6),
        transforms.Normalize(),
        ToTensorV2()
    ])

def rectangle_cutmix(img1, img2, min_edge_len, max_edge_len):
    dx = math.floor(
        min_edge_len + np.random.uniform() * (max_edge_len - min_edge_len))
    dy = math.floor(
        min_edge_len + np.random.uniform() * (max_edge_len - min_edge_len))
    x0 = np.random.randint(0, img1.shape[0] - dx)
    y0 = np.random.randint(0, img1.shape[1] - dy)

    x1 = np.random.randint(0, img2.shape[0] - dx)
    y1 = np.random.randint(0, img2.shape[1] - dy)

    img1[x0:(x0 + dx), y0:(y0+dy)] = img2[x1:(x1 + dx), y1:(y1+dy)]
    
    return img1
