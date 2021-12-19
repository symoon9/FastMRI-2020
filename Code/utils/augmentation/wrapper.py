"""Augmentation methods.
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from abc import ABC
from typing import List, Tuple


class Augmentation(ABC):
    """Abstract class used by all augmentation methods."""

    def __init__(self) -> None:
        """Initialize."""

    def _apply_augment(self, seq, transf: Tuple):
        """Apply and get the augmented image.

        Args:
            seq: sequence of data
            Transf: the tuple of transf, args, (transf, args) ex) ("Jittering", 0.1)

        returns:
            seq
        """
        if isinstance(transf, Tuple):
            aug_fn = getattr(__import__("utils.augmentation.transforms", fromlist=[""]), transf[0])(*transf[1:])
        else:
            aug_fn = getattr(__import__("utils.augmentation.transforms", fromlist=[""]), transf)()
        return aug_fn(seq)


class RandAugmentation(Augmentation):
    """Random augmentation class.

    References:
        RandAugment: Practical automated data augmentation with a reduced search space
        (https://arxiv.org/abs/1909.13719)

    """

    def __init__(
        self,
        transforms_info: List[Tuple],
        n_select: int = 2,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.n_select = n_select
        self.transforms_info = transforms_info

    def __call__(self, seq):
        """Run augmentations."""
        chosen_transforms = random.sample(self.transforms_info, k=self.n_select)
        for transf in chosen_transforms:
            seq = self._apply_augment(seq, transf)
        return seq


class RandAugmentationWithProb(Augmentation):
    """Random augmentation class + probability

    References:
        RandAugment: Practical automated data augmentation with a reduced search space
        (https://arxiv.org/abs/1909.13719)

    """

    def __init__(
        self,
        transforms_info: List[Tuple],
        n_select: int = 2,
        trans_prob: float = 0.2
    ) -> None:
        """Initialize."""
        super().__init__()
        self.transforms_info = transforms_info
        self.n_select = n_select
        self.trans_prob = trans_prob

    def __call__(self, seq):
        """Run augmentations."""
        chosen_transforms = random.sample(self.transforms_info, k=self.n_select)
        for transf in chosen_transforms:
            if random.random() < self.trans_prob:
                seq = self._apply_augment(seq, transf)
        return seq
