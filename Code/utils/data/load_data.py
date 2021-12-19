# import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import torch

class SliceData(Dataset):
    def __init__(self, root, input_key, target_key, layer, forward=False):
        # self.transform = transform
        self.root = root
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.layer = layer

        self.input_files = sorted(_get_filelist(self.input_key))
        self.target_files = sorted(_get_filelist(self.target_key))

        
    def _get_filelist(key):
        files = list(os.listdir(os.path.join(self.root, key)))
        if self.layer == -1:
            return files
        else:
            postfix = f'_{layer}.npy'
            return [f for files in files if postfix in f.split('/')[-1]]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, i):
        input = np.load(self.input_files[i])
        target = np.load(self.target_files[i])
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        # return self.transform(input, target, attrs, fname.name, dataslice)
        return input, target


def create_data_loaders(data_path, args, isforward=False):
    assert type(args.layer) is int
    assert args.layer>=-1 and args.layer<=15

    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        # transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        layer=args.layer,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size
    )
    return data_loader
