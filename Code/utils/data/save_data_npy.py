import os
import numpy as np 
import h5py
from pathlib import Path
import argparse


def save_to_numpy(root, save_dir, spc_dir='train'):
    files = list(Path(root).iterdir())
    examples = []
    for fname in sorted(files):
        with h5py.File(fname, "r") as hf:
            for key in hf.keys():
                path = os.path.join(save_dir, spc_dir, key)
                os.makedirs(path, exist_ok=True)
                for i in range(hf[key].shape[0]):
                    fname = str(fname)
                    file_name = fname.split('/')[-1].split('.')[0]+'_'+str(i)+'.npy'
                    np.save(os.path.join(path, file_name), hf[key][i, :, :], allow_pickle=False)

def parse():
    parser = argparse.ArgumentParser(description='Saving image files to .npy',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_train', type=str, default='./Data/train/')
    parser.add_argument('--root_valid', type=str, default='./Data/val/')
    parser.add_argument('--root_test', type=str, default='./Data/image_Leaderboard/')
    parser.add_argument('--save_dir', tye=str, default='./Data/image2')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print('Start for train data')
    save_to_numpy(args.root_train, save_dir=args.save_dir)
    print('Start for valid data')
    save_to_numpy(args.root_valid, save_dir=args.save_dir, spc_dir='val')
    print('Start for test data')
    save_to_numpy(args.root_test, save_dir=args.save_dir, spc_dir='test')
    print('Done!')