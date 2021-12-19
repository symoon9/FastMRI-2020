import numpy as np
import torch
import torchvision.transforms as T
import random
from copy import deepcopy

class NullTransform:
    def __init__(self):
        pass
    def __call__(self, img: list):
        '''
        img: 3D (channel, H, W) torch tensor
        '''
        return img

class CentorCrop:
    def __init__(self, crop_range: float = .05):
        self.crop_range = crop_range

    def __call__(self, img: list):
        '''
        img: list of 3D (channel, H, W) torch tensor
        '''
        crop_pct = random.random()*self.crop_range
        origin_size = int(img[0].shape[-1])
        crop_size = int(origin_size*(1-crop_pct))
        done_img = []
        for i in img:
            tmp_img = T.CenterCrop(crop_size)(i)
            tmp_img = T.Resize(origin_size)(tmp_img)
            done_img.append(tmp_img)
        return done_img

class HorizontalFlip:
    def __init__(self):
        pass
    def __call__(self, img: list):
        '''
        img: 3D (channel, H, W) torch tensor
        '''
        done_img = []
        for i in img:
            done_img.append(T.functional.hflip(i))
        return done_img

class ChangeMagnitude:
    def __init__(self, mag_range: float = .1):
        self.mag_range = mag_range
    def __call__(self, img: list):
        '''
        img: 3D (channel, H, W) torch tensor
        '''
        mag_pct = random.random()*self.mag_range

        if random.random()>0.5:
            mag_pct = -mag_pct
        
        done_img=[i*(1-mag_pct) for i in img]
        return done_img

class VerticalStretch:
    def __init__(self, stretch_range: float = .05):
        self.stretch_range = stretch_range
    def __call__(self, img: list):
        '''
        img: list of 3D (channel, H, W) torch tensor
        '''
        stretch_pct = random.random()*self.stretch_range
        origin_size = int(img[0].shape[-1])
        stretch_size = int(origin_size*stretch_pct)
        done_img = []
        for i in img:
            stretch_img = i[...,stretch_size:-stretch_size,:]
            if stretch_img.shape[-1]==0 or stretch_img.shape[-2]==0:
                done_img.append(i)
            else:
                done_img.append(T.Resize(size=[origin_size]*2)(stretch_img))
        return done_img

class HorizontalStretch:
    def __init__(self, stretch_range: float = .06):
        self.stretch_range = stretch_range
    def __call__(self, img: torch.Tensor):
        '''
        img: 3D (channel, H, W) torch tensor
        '''
        stretch_pct = random.random()*self.stretch_range
        origin_size = int(img[0].shape[-1])
        stretch_size = int(origin_size*stretch_pct)
        done_img = []
        for i in img:
            stretch_img = i[...,stretch_size:-stretch_size]
            if stretch_img.shape[-1]==0 or stretch_img.shape[-2]==0:
                done_img.append(i)
            else:
                done_img.append(T.Resize(size=[origin_size]*2)(stretch_img))
        return done_img

# class CentorCrop:
#     def __init__(self, crop_range: float = .05):
#         self.crop_range = crop_range

#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         crop_pct = random.random()*self.crop_range
#         origin_size = int(img.shape[-1])
#         crop_size = int(origin_size*(1-crop_pct))
#         crop_img = T.CenterCrop(crop_size)(img)
#         crop_img = T.Resize(origin_size)(crop_img)
#         return crop_img

# class HorizontalFlip:
#     def __init__(self):
#         pass
#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         return T.functional.hflip(img)

# class Jittering:
#     def __init__(self, p: float = .3):
#         pass
#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         pass

# class ChangeMagnitude:
#     def __init__(self, mag_range: float = .1):
#         self.mag_range = mag_range
#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         mag_pct = random.random()*self.mag_range
#         if random.random()>0.5:
#             mag_pct = -mag_pct
#         return img*(1-mag_pct)

# class VerticalStretch:
#     def __init__(self, stretch_range: float = .05):
#         self.stretch_range = stretch_range
#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         if img.shape[-1]==0 or img.shape[-2]==0:
#             raise Exception(f"img size error: {img.shape}")
#         stretch_pct = random.random()*self.stretch_range
#         origin_size = int(img.shape[-1])
#         stretch_size = int(origin_size*stretch_pct)
#         stretch_img = img[:,stretch_size:-stretch_size,:]
#         if stretch_img.shape[-1]==0 or stretch_img.shape[-2]==0:
#             stretch_resized_img = img
#             # print("img size error")
#         else:
#             stretch_resized_img = T.Resize(size=[origin_size]*2)(stretch_img)
#             # print("*"*30)
#         return stretch_resized_img

# class HorizontalStretch:
#     def __init__(self, stretch_range: float = .06):
#         self.stretch_range = stretch_range
#     def __call__(self, img: torch.Tensor):
#         '''
#         img: 3D (channel, H, W) torch tensor
#         '''
#         if img.shape[-1]==0 or img.shape[-2]==0:
#             raise Exception(f"img size error: {img.shape}")
#         stretch_pct = random.random()*self.stretch_range
#         origin_size = int(img.shape[-1])
#         stretch_size = int(origin_size*stretch_pct)
#         stretch_img = img[:,:,stretch_size:-stretch_size]
#         if stretch_img.shape[-1]==0 or stretch_img.shape[-2]==0:
#             stretch_resized_img = img
#             # print("img size error")
#         else:
#             stretch_resized_img = T.Resize(size=[origin_size]*2)(stretch_img)
#             # print("*"*30)
#         return stretch_resized_img

# class GaussianBlur:
#     def __init__(self, kernel_size=(3,3), sigma=(0.1, 0.6)):
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#     def __call__(self, img: torch.Tensor):
#         return T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)(img)


# class GaussianNoise:
#     def __init__(self, mean=0., std=4e-6):
#         self.std = std
#         self.mean = mean
        
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)