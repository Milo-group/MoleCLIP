import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


def augmentation(aug, image_size = 224):

    if aug == "none":
        return A.Compose([
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])
    
    elif aug == "default":
        return A.Compose([
            A.Downscale(scale_min=0.7, scale_max=0.85, p=0.75, interpolation=dict(downscale=cv2.INTER_AREA, upscale=cv2.INTER_NEAREST)),
            A.InvertImg(p=1),
            A.LongestMaxSize(max_size=np.random.randint(int(image_size * 0.6), image_size), p=0.5),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=1, position=A.PadIfNeeded.PositionType.RANDOM),
            A.PixelDropout(dropout_prob=0.01, p=0.3),
            A.InvertImg(p=1),
            A.PixelDropout(dropout_prob=0.01, p=0.3),
            A.Blur(p=0.3, blur_limit=3),
            A.GaussNoise(p=0.3),
            A.ToGray(p=0.25),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])

    elif aug == "intense":
        return A.Compose([A.geometric.rotate.Rotate(limit=360, border_mode=1),
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.75, interpolation = dict(downscale=cv2.INTER_AREA, upscale=cv2.INTER_NEAREST)),
            A.InvertImg(p=1),
            A.LongestMaxSize(max_size=np.random.randint(int(image_size * 0.5), image_size), p = 0.75),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=1, position = A.PadIfNeeded.PositionType.RANDOM),  # Pad to the original dimensions
            A.PixelDropout(dropout_prob=0.05, p=0.5),
            A.InvertImg(p=1),
            A.PixelDropout(dropout_prob=0.05, p=0.5),
            A.Blur(p=0.5, blur_limit=3),
            A.GaussNoise(p=0.5),
            A.ToGray(p=0.5),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()])
            
    else:
        raise Exception(f"Invalid augmentation type '{aug}'")

