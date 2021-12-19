
#cd SNU_challenge/Code/
#python preprocessing_kspace.py -d ../Data/kspace_Leaderboard -n -1 -l test -o y
#python preprocessing_kspace.py -d ../Data/kspace_Leaderboard -n 21 -l test -o y
#python preprocessing_kspace.py -d ../Data/kspace_Leaderboard -n 43 37 36 21 -l test -o y
#python preprocessing_kspace.py -d ../Data/kspace -n  20 27 83 86 112 179 182 191 264 277 281 309 317 329 337 343 374  -l train -o y

import argparse
import shutil
from pathlib import Path
import os 
import re
import time
import h5py
import numpy as np
import pandas as pd
from numpy.fft import fftshift, fftn, ifftshift, ifftn
from tqdm.auto import tqdm, trange
import torch
import torchvision
import torchvision.transforms as transforms

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

def espirit(X, k, r, t, c):
    
    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)
    
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)

    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r-k+1)**p, k**p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            for zdx in range(max(1, C.shape[2] - k + 1)):
                block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64) 
                A[idx, :] = block.flatten()
                idx = idx + 1

    U, S, VH = np.linalg.svd(A, full_matrices=True)
    
    V = VH.conj().T

    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)

    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            kerimgs[:,:,:,jdx,idx] = fft(ker, axes) * np.sqrt(sx * sy * sz)/np.sqrt(k**p)

    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx,jdx,kdx,:,:]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if (s[ldx]**2 > c):
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]

    return maps

def espirit_proj(x, esp):
    
    ip = np.zeros(x.shape).astype(np.complex64)
    proj = np.zeros(x.shape).astype(np.complex64)
    
    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
            ip[:, :, :, qdx] = ip[:, :, :, qdx] + x[:, :, :, pdx] * esp[:, :, :, pdx, qdx].conj()

    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
            proj[:, :, :, pdx] = proj[:, :, :, pdx] + ip[:, :, :, qdx] * esp[:, :, :, pdx, qdx]

    return (ip, proj, x - proj)

class ComplexPCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None

    @property
    def explained_variance_ratio_(self):
        return self.s

    def fit(self, matrix): # with truncated SVD
        self.mean_ = matrix.mean(axis=0)
        _, self.s, vh = np.linalg.svd(matrix, full_matrices=False)

        assert 0<=self.n_components and self.n_components<=self.s.shape[0]
        
        n=self.s.shape[0]
        for i in range(n-self.n_components):
            _ = np.delete(_, -1, axis = 1)
            self.s = np.delete(self.s, -1, axis = 0)
            vh = np.delete(vh, -1, axis = 0)
            
        self.components_ = vh 

    def transform(self, matrix):
        data = matrix - self.mean_
        result = data @ self.components_.T
        return result

    def inverse_transform(self, matrix):
        result = matrix @ np.conj(self.components_)
        return self.mean_ + result
    
    def get_components_(self):
        return self.components_
    
def preprocessing_kspace(args):
    
    file_list = os.listdir(args.dir)
    file_list_h5 = [file for file in file_list if file.endswith(".h5")]
    
    file_name = file_list_h5[0].replace(".h5","")
    file_name = re.sub(r'[0-9]+', '', file_name)
    file_num = len(file_list_h5)
    
    file_targets=[]
    if args.num_target[0] == '-1':
        print("[Preprocessing kspace] ALL #{}".format(file_num))
        file_targets = list(i+1 for i in range(len(file_list_h5)))
    else:
        file_targets = args.num_target
        print("[Preprocessing kspace] #", ", # ".join(file_targets))

    outliers = []
    
    for file_target in tqdm(file_targets, desc='kspace.h5 Data'):
        
        print("[Open] " + str(Path(args.dir) / str(file_name + file_target + ".h5")))
        
        # kspace store array
        kspace_processed = []

        # kspace load
        file = h5py.File(Path(args.dir) / str(file_name + file_target + ".h5"), 'r') 
                         
        kspace = file['kspace']
        if args.train_or_test=="train":
            mask = file['mask']
        elif args.train_or_test=="test":
            pass
        else:
            print(1)
            raise cmdError()

        # center crop
        X = []
        for c in tqdm(range(kspace.shape[0]), desc='center crop'):
            XX = []
            for l in range(kspace.shape[1]):
                if args.train_or_test=="train":
                    img_kspace = kspace[c, l, :, :]*mask
                elif args.train_or_test=="test":
                    img_kspace = kspace[c, l, :, :]
                else:
                    raise cmdError()
                
                img = ifft(img_kspace, range(img_kspace.ndim))
                img_crop = transforms.CenterCrop(size=(384, 384))(torch.tensor(img)).numpy()
                img_kspace_crop = fft(img_crop, range(img_kspace.ndim))
                XX.append(img_kspace_crop)
            X.append(XX)
        X = np.array(X)

        # esiprit & pca
        for layer in tqdm(range(X.shape[0]), desc='esiprit & pca'):
            try:
                kspace_l = np.expand_dims(X[layer].reshape(-1,384,384),axis=0)
                kspace_l = np.transpose(kspace_l, (0, 2, 3, 1))
                x_l = ifft(kspace_l, (0,1,2))

                esp = espirit(kspace_l, args.esp_kernel, args.esp_region, args.esp_t, args.esp_c)
                ip, proj, null = espirit_proj(x_l, esp)
                proj = np.transpose(proj[0],(2,0,1))
                proj = fft(proj,range(proj.ndim))

                proj = proj.reshape([-1,384*384])
                pca = ComplexPCA(n_components=4)
                pca.fit(proj)
                kspace_pca_reduced = pca.transform(proj)
                kspace_pca_recovered = pca.inverse_transform(kspace_pca_reduced)
                kspace_pca_components_ = pca.get_components_()
                image_pca_components_ = kspace_pca_components_.reshape([-1,384,384])
                #plot_img_a_coil(image_pca_components_)

                kspace_processed.append(image_pca_components_)
            
            except:
                outliers.append([args.train_or_test, file_target, layer]) # [ train_or_test, num, layer]
                kspace_processed.append(np.zeros([4,384,384], dtype=np.complex128))

        # store
        kspace_processed = np.array(kspace_processed)
        
        st_file_name = Path(args.store_dir) / str(file_name + "_processed" + file_target + ".h5")
        if os.path.isfile(st_file_name):
            os.remove(st_file_name)
        st_file = h5py.File(st_file_name, 'w') 
        st_file.create_dataset('kspace_processed', data=kspace_processed)
        st_file.close()
        file.close()
        
        print("kspace_processed.shape: ", kspace_processed.shape)
        print("[Store] ", st_file_name)
    
    if args.outliers_store=="y":
        outlier_file = Path(args.store_dir) / "outliers{}.npy".format(time.strftime('%Y-%m-%d_%I:%M:%S_%p', time.localtime()))
        if os.path.isfile(outlier_file):
            os.remove(outlier_file)
        np.save(outlier_file, np.array(outliers))
        print("[Store] ", str(outlier_file))
    
    if len(outliers) : 
        print("outliers: [ train_or_test, num, layer ]\n", outliers)
    else :
        print("outliers: Nope!")
    
class cmdError(Exception):
    pass
    
def parse():
    
    parser = argparse.ArgumentParser(description='Preprocessing train or test kspace with Espirit and PCA',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='../Data/kspace', help='Directory of kspace', required=True)
    parser.add_argument('-s', '--store-dir', type=str, default='../Data/kspace_processed', help='Store directory of kspace_processed')
    parser.add_argument('-n', '--num-target', nargs='+', default='-1', help='List of target kspace numbers', required=True) # all: [-1]
    parser.add_argument('-l', '--train-or-test', type=str, default="train", help='Kspace data for train or test', required=True)
    parser.add_argument('-o', '--outliers-store', type=str, default="y", help='outliers list store [y/n]', required=True)
    
    parser.add_argument('-k', '--esp-kernel', type=int, default=6, help='Espirit kernel size')
    parser.add_argument('-r', '--esp-region', type=int, default=24, help='Espirit region size')
    parser.add_argument('-t', '--esp-t', type=float, default=0.01, help='Espirit t factor')
    parser.add_argument('-c', '--esp-c', type=float, default=0.9925, help='Espirit c factor')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse()
    Path(args.store_dir).mkdir(parents=True, exist_ok=True)
    preprocessing_kspace(args)