import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
import h5py
import re

from utils.augmentation.policies import default_transform

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

MINMAX='minmax'
Z_NORM='z-norm'

kp_fn = re.compile("^brain_processed\d+.h5$")
kp_tfn = re.compile("^brain_test_processed\d+.h5$")
nan_train_kspace=(20, 27, 83, 86, 112, 167, 179, 182, 191, 264, 277, 281, 309, 317, 329, 337, 343, 374)
nan_val_kspace=(21, 36, 37,43)
# nan_train_kspace=[]
# nan_val_kspace=[]
abnormal = 167

def normalize(array_2d):
    '''
    'minmax'
    array_2d: 2D numpy array
    '''
    # assert len(array_2d.shape)==2
    min_ = array_2d.min()
    max_ = array_2d.max()
    scaled_array = (array_2d-min_)/(max_-min_)
    return scaled_array

def z_normalize(array_2d):
    '''
    'z-norm'
    array_2d: 2D numpy array
    '''
    # assert len(array_2d.shape)==2
    mean_ = array_2d.mean()
    std_ = array_2d.std()
    scaled_array = (array_2d-mean_)/std_
    return scaled_array

def load_npy(filepath):
    img = np.load(filepath, allow_pickle=True)
    return img

def save_npy(filepath, img):
    with open(filepath, 'wb') as f:
        np.save(f, img)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])
def is_h5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])
def is_val_file(filename):
    return "test" in filename
def is_k_processed_file(filename):
    return kp_fn.match(filename) != None
def is_k_processed_test_file(filename):
    return kp_tfn.match(filename) != None

def load_h5(filepath, key, key2=None):
    try:
        hf = h5py.File(filepath, "r")
        if key2==None:
            return hf[key][:]
        else:
            return hf[key][:], hf[key2][:]
    except:
        print(filepath)
        print(hf.keys())
        print(key)
        return

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, img_options=None, target_transform=None, 
                gt_dir='image_label', input_dir='image_grappa',
                augmentation=True, aug_n_select=2, aug_prob=0.4, norm=MINMAX):
        assert os.path.exists(data_dir)
        
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
    
        clean_files_can = sorted(os.listdir(os.path.join(data_dir, gt_dir)))
        noisy_files_can = sorted(os.listdir(os.path.join(data_dir, input_dir)))
        clean_files = []
        noisy_files = []
        for i, fn in enumerate(clean_files_can):
            fn_sp = fn.split('_')[-2]
            if str(abnormal) in fn_sp:
                pass 
            else:
                clean_files.append(clean_files_can[i])
                noisy_files.append(noisy_files_can[i])
        
        self.clean_filenames = [os.path.join(data_dir, gt_dir, x) for x in clean_files if is_numpy_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, input_dir, x) for x in noisy_files if is_numpy_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

        self.norm = norm
        self.do_augmentation = augmentation
        self.transform = default_transform(n_select=aug_n_select, trans_prob=aug_prob)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean_npy = load_npy(self.clean_filenames[tar_index])
        clean_npy = z_normalize(clean_npy) if self.norm==Z_NORM else normalize(clean_npy)
        clean_npy = clean_npy[np.newaxis, ...] # make shape to (1, H, W)
        clean = torch.from_numpy(np.float32(clean_npy))

        noisy_npy = load_npy(self.noisy_filenames[tar_index])
        noisy_npy = z_normalize(noisy_npy) if self.norm==Z_NORM else normalize(noisy_npy)
        noisy_npy = noisy_npy[np.newaxis, ...] # make shape to (1, H, W)
        noisy = torch.from_numpy(np.float32(noisy_npy))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop images
        ps = self.img_options['patch_size']
        if ps != None:
            H = clean.shape[1]
            W = clean.shape[2]
            r = np.random.randint(0, H - ps) if not H-ps else 0
            c = np.random.randint(0, W - ps) if not H-ps else 0
            if H-ps==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - ps)
                c = np.random.randint(0, W - ps)
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps]

        if self.do_augmentation:
            aug_img = self.transform([clean, noisy])
            clean = aug_img[0]
            noisy = aug_img[1]

        return clean, noisy, clean_filename, noisy_filename


class DataLoaderVal(Dataset):
    def __init__(self, data_dir ,target_transform=None, gt_dir='image_label', input_dir='image_grappa', norm=MINMAX):
        assert os.path.exists(data_dir)

        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        
        clean_files = sorted(os.listdir(os.path.join(data_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, input_dir)))

        self.clean_filenames = [os.path.join(data_dir, gt_dir, x) for x in clean_files if is_numpy_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, input_dir, x) for x in noisy_files if is_numpy_file(x)]
        
        self.tar_size = len(self.clean_filenames)  
        self.norm = norm

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean_npy = load_npy(self.clean_filenames[tar_index])
        clean_minmax = (clean_npy.min(), clean_npy.max())
        clean_npy = z_normalize(clean_npy) if self.norm==Z_NORM else normalize(clean_npy)
        clean_npy = clean_npy[np.newaxis, ...] # make shape to (1, H, W)
        clean = torch.from_numpy(np.float32(clean_npy))

        noisy_npy = load_npy(self.noisy_filenames[tar_index])
        noisy_minmax = (noisy_npy.min(), noisy_npy.max())
        noisy_npy = z_normalize(noisy_npy) if self.norm==Z_NORM else normalize(noisy_npy)
        noisy_npy = noisy_npy[np.newaxis, ...] # make shape to (1, H, W)
        noisy = torch.from_numpy(np.float32(noisy_npy))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return clean, noisy, clean_filename, noisy_filename, clean_minmax, noisy_minmax

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, data_dir, input_key='image_grappa', target_transform=None, norm='minmax'):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(data_dir))
        self.noisy_filenames = [os.path.join(data_dir, x) for x in noisy_files if is_h5_file(x)]
        self.tar_size = len(self.noisy_filenames)  
        self.input_key = input_key
        self.norm = norm

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        noisy_npy = load_h5(self.noisy_filenames[tar_index], self.input_key)
        noisy_minmax = [(n.min(), n.max()) for n in noisy_npy]
        for i in range(noisy_npy.shape[0]):
            noisy_npy[i] = z_normalize(noisy_npy[i]) if self.norm==Z_NORM else normalize(noisy_npy[i])
        noisy_npy = noisy_npy[:, np.newaxis, ...]
        noisy = torch.from_numpy(np.float32(noisy_npy))

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return noisy, noisy_minmax, noisy_filename

class KspaceDataLoaderTrain(Dataset):
    def __init__(self, kspace_dir, image_dir, img_options=None, target_transform=None, 
                kspace_key='kspace_processed', image_input_key='image_grappa', image_target_key='image_label', 
                augmentation=False, aug_n_select=2, aug_prob=0.4, norm=MINMAX):
        assert os.path.exists(kspace_dir)
        assert os.path.exists(image_dir)
        
        super(KspaceDataLoaderTrain, self).__init__()
        self.target_transform = target_transform
    
        # image_files = sorted(fn for fn in os.listdir(image_dir))
        image_files = []
        kspace_files_can = sorted(fn for fn in os.listdir(kspace_dir) if is_k_processed_file(fn) and not is_val_file(fn))
        kspace_files = []
        for kfc in kspace_files_can:
            num = int(re.findall(r'\d+', kfc)[0])
            image_files.append(f'brain{num}.h5')
            if num in nan_train_kspace:
                kspace_files.append("NaN")
            else:
                kspace_files.append(kfc)

        self.image_filenames = [os.path.join(image_dir, x) for x in image_files]
        self.kspace_filenames = [os.path.join(kspace_dir, x) for x in kspace_files]
        self.image_input_key = image_input_key
        self.image_target_key = image_target_key
        self.kspace_key = kspace_key
        
        self.img_options=img_options

        self.tar_size = len(self.image_filenames)  # get the size of target
        print(self.tar_size)

        self.norm = norm
        self.do_augmentation = augmentation
        if self.do_augmentation:
            self.transform = default_transform(n_select=aug_n_select, trans_prob=aug_prob)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        img_input_npy, img_target_npy = load_h5(self.image_filenames[tar_index], 
                                                self.image_input_key,
                                                self.image_target_key)
        img_input_npy = z_normalize(img_input_npy) if self.norm==Z_NORM else normalize(img_input_npy)
        img_input_npy = img_input_npy[:,np.newaxis, ...]
        img_target_npy = z_normalize(img_target_npy) if self.norm==Z_NORM else normalize(img_target_npy)
        img_target_npy = img_target_npy[:,np.newaxis, ...]
        img_input = torch.from_numpy(np.float32(img_input_npy))
        img_target = torch.from_numpy(np.float32(img_target_npy))


        # image_filename = os.path.split(self.image_filenames[tar_index])[-1]
        # kspace_filename = os.path.split(self.kspace_filenames[tar_index])[-1]

        if "NaN" in self.kspace_filenames[tar_index]:
            kspace=None
        else:
            kspace_npy = load_h5(self.kspace_filenames[tar_index], self.kspace_key)
            kspace_img_npy = np.array(abs(ifft(kspace_npy, range(kspace_npy.ndim))), dtype=np.float32)
            kspace_img_npy = z_normalize(kspace_img_npy) if self.norm==Z_NORM else normalize(kspace_img_npy)

            assert img_input_npy.shape[0]==kspace_npy.shape[0] and img_target_npy.shape[0]==kspace_npy.shape[0]
            
            if self.do_augmentation:
                kspace_img = torch.from_numpy(kspace_img_npy)
                aug_img = self.transform([kspace_img, img_input, img_target])
                kspace_img = aug_img[0]
                img_input = aug_img[1]
                img_target = aug_img[2]
                kspace_img_npy = kspace_img.numpy()

            kspace_trans_cplx = fft(kspace_img_npy, range(kspace_img_npy.ndim))
            kspace_real = np.real(kspace_trans_cplx)
            kspace_imag = np.imag(kspace_trans_cplx)
            sh = kspace_npy.shape
            kspace_reshape = np.stack((kspace_real, kspace_imag), axis=1).reshape(sh[0], -1, sh[-2], sh[-1])
            kspace = torch.from_numpy(np.float32(kspace_reshape))

        return kspace, img_input, img_target

class KspaceDataLoaderVal(Dataset):
    def __init__(self, kspace_dir, image_dir, img_options=None, target_transform=None, 
                kspace_key='kspace_processed', image_input_key='image_grappa', image_target_key='image_label', 
                norm=MINMAX):

        assert os.path.exists(kspace_dir)
        assert os.path.exists(image_dir)
        
        super(KspaceDataLoaderVal, self).__init__()
        self.target_transform = target_transform
    
        image_files = []
        kspace_files_can = sorted(fn for fn in os.listdir(kspace_dir) if is_k_processed_test_file(fn) and is_val_file(fn))
        kspace_files = []
        for kfc in kspace_files_can:
            num = re.findall(r'\d+', kfc)[0]
            image_files.append(f'brain_test{num}.h5')
            if num in nan_val_kspace:
                kspace_files.append("NaN")
            else:
                kspace_files.append(kfc)

        self.image_filenames = [os.path.join(image_dir, x) for x in image_files if is_h5_file(x)]
        self.kspace_filenames = [os.path.join(kspace_dir, x) for x in kspace_files if is_h5_file(x)]
        self.image_input_key = image_input_key
        self.image_target_key = image_target_key
        self.kspace_key = kspace_key
        
        self.img_options=img_options

        self.tar_size = len(self.image_filenames)  # get the size of target

        self.norm = norm
    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        img_input_npy, img_target_npy = load_h5(self.image_filenames[tar_index], 
                                                self.image_input_key,
                                                self.image_target_key)

        img_input_minmax = (img_input_npy.min(), img_input_npy.max())
        img_target_minmax = (img_target_npy.min(), img_target_npy.max())

        img_input_npy = z_normalize(img_input_npy) if self.norm==Z_NORM else normalize(img_input_npy)
        img_input_npy = img_input_npy[:,np.newaxis, ...]
        img_target_npy = z_normalize(img_target_npy) if self.norm==Z_NORM else normalize(img_target_npy)
        img_target_npy = img_target_npy[:,np.newaxis, ...]
        img_input = torch.from_numpy(np.float32(img_input_npy))
        img_target = torch.from_numpy(np.float32(img_target_npy))


        # image_filename = os.path.split(self.image_filenames[tar_index])[-1]
        # kspace_filename = os.path.split(self.kspace_filenames[tar_index])[-1]

        if "NaN" in self.kspace_filenames[tar_index]:
            kspace=None
        else:
            kspace_npy = load_h5(self.kspace_filenames[tar_index], self.kspace_key)
            kspace_img_npy = np.array(abs(ifft(kspace_npy, range(kspace_npy.ndim))), dtype=np.float32)
            kspace_img_npy = z_normalize(kspace_img_npy) if self.norm==Z_NORM else normalize(kspace_img_npy)

            assert img_input_npy.shape[0]==kspace_npy.shape[0] and img_target_npy.shape[0]==kspace_npy.shape[0]
            
            kspace_trans_cplx = fft(kspace_img_npy, range(kspace_img_npy.ndim))
            kspace_real = np.real(kspace_trans_cplx)
            kspace_imag = np.imag(kspace_trans_cplx)
            sh = kspace_npy.shape
            kspace_reshape = np.stack((kspace_real, kspace_imag), axis=1).reshape(sh[0], -1, sh[-2], sh[-1])
            kspace = torch.from_numpy(np.float32(kspace_reshape))

        return kspace, img_input, img_target, img_input_minmax, img_target_minmax