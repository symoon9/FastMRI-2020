import pickle as pkl
from pathlib import Path
import h5py
import os

root_train = '../SNU-Challenge/Data/train/'
root_valid = '../SNU-Challenge/Data/val/'
save_dir = './Data/'
input_key = 'image_input'

def make_file_list(root, save_dir=save_dir, isvalid=False):
    files = list(Path(root).iterdir())
    examples = []
    for fname in sorted(files):
        num_slices = _get_metadata(fname)

        examples += [
            (fname, slice_ind) for slice_ind in range(num_slices)
        ]
    
    file_name = 'val_tuple_list.pkl' if isvalid else 'train_tuple_list.pkl'
    with open(save_dir+file_name, 'wb') as f:
        pkl.dump(examples, f, pkl.HIGHEST_PROTOCOL)

    return examples

def _get_metadata(fname):
    with h5py.File(fname, "r") as hf:
        num_slices = hf[input_key].shape[0]
    return num_slices

if __name__ == '__main__':
    train_list = make_file_list(root_train)
    # val_list = make_file_list(root_valid, isvalid=True)
    print('Done!')