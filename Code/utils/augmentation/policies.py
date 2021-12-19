import torchvision.transforms as T

from utils.augmentation.transforms import *
from utils.augmentation.wrapper import RandAugmentationWithProb

def default_transform(n_select=2, trans_prob=0.2):
    transforms_info = [
        ("NullTransform"),
        ("CentorCrop", .05),
        ("HorizontalFlip"),
        ("ChangeMagnitude", 0.1),
        ("VerticalStretch", .05),
        ("HorizontalStretch", .06),
    ]
    return T.Compose([
        RandAugmentationWithProb(
            transforms_info=transforms_info,
            n_select=n_select,
            trans_prob=trans_prob,
        )
    ])