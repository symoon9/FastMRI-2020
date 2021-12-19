# Code for split validation data and save to var/ dir

import os
from pathlib import Path
from glob import glob
import re 

val_list = list(range(407-40, 407+1))
root = '/content/drive/Shareddrives/FastMRI-Duya/Duya/Data/'
# files = list(Path(root+'train/').iterdir())
files = [y for x in os.walk(Path(root+'train/')) for y in glob(os.path.join(x[0], '*.npy'))]
print(len(files))
print(files[-1].split('/')[-2], re.search(r'\d+', files[-1].split('/')[-1]).group(0))


os.makedirs(root+'val/image_grappa')
os.makedirs(root+'val/image_input')
os.makedirs(root+'val/image_label')

for fn in files:
    fsp = fn.split('/')
    n = fsp[-1]
    d = fsp[-2]
    if int(re.search(r'\d+', fn.split('/')[-1]).group(0)) in val_list:
        os.system(f"mv {fn} {os.path.join(root, 'val', d, n)}")
    
