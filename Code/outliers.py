
# cd Code
# python outliers.py -d ../Data/kspace_processed
import numpy as np
import os
from pathlib import Path
import argparse

def outliers_merge(args):
    
    file_list = os.listdir(args.dir)
    file_list_npy = [file for file in file_list if file.endswith(".npy")]
    print(file_list_npy)
    
    outliers = []
    for file in file_list_npy:
        outlier = np.load(Path(args.dir) / file)
        outliers.append(outlier)
        
    np.save(Path(args.dir) / "outliers.npy", np.array(outliers))
    print("[Store] ", str(Path(args.dir) / "outliers.npy"))
        
def parse():
    
    parser = argparse.ArgumentParser(description='Merge outliers files',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='../Data/kspace_processed', help='Directory of outliers files', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse()
    outliers_merge(args)