import argparse
import shutil
from utils.learning.train_part import train
from pathlib import Path


def parse():
    args = argparse.Namespace(
      GPU_NUM=0,
      batch_size=4,
      num_epochs=3,
      lr=1e-3,
      report_interval=500,
      net_name='test_Unet',
      data_path_train='/FastMRI/SNU_challenge/Data/train/',
      data_path_val='/FastMRI/SNU_challenge/Data/val/',
      in_chans=1,
      out_chans=1,
      input_key='image_input',
      target_key='image_label',
      max_key='max',
      result_path='../result'
    )

    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = Path(args.result_path) / args.net_name / 'checkpoints'
    args.val_dir = Path(args.result_path) / args.net_name / 'reconstructions_val'
    args.main_dir = Path(args.result_path) / args.net_name / __file__
    
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.val_dir).mkdir(parents=True, exist_ok=True)

    train(args)
