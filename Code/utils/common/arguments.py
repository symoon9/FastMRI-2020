import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Train Network for FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Status')
    parser.add_argument('-g', '--gpu', type=str, default='0,1', help='GPU number to allocate')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--report_interval', type=int, default=500, help='Report interval')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--layer', type=int, default=-1, help='Layer to be train: -1 for all layers')
    parser.add_argument('--checkpoint', type=int, default=50, help='Save at checkpoint after this epochs')
    
    # train/val data load
    parser.add_argument('--train_workers', type=int, default=16, help='train_dataloader workers')
    parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
    
    # model
    parser.add_argument('-m', '--model', type=str, default='Uformer', help='Name of network')
    parser.add_argument('--in_chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out_chans', type=int, default=1, help='Size of output channels for network')
    
    # optimizer, scheduler, earlystopping and loss
    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--scheduler', type=str, default ='cosine', help='optimizer for training')
    parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
    parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')
    parser.add_argument('--step_size', type=int, default=10, help='step size for StepLR scheduler')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='gamma for StepLR scheduler')
    parser.add_argument('--early_stop', action='store_true', default=False, help='do early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='set patience for early stopping')
    parser.add_argument('--val_loss_mask', action='store_true', default=False, help='apply mask to valid loss')
    parser.add_argument('--max_grad_norm', type=float, default=-1.0, help='gradient clipping')
    
    # preprocess & augmentation
    parser.add_argument('--normalization', type=str, default='minmax', help='Which norm to apply')
    parser.add_argument('--augmentation', action='store_true', default=False, help='do augmentation')
    parser.add_argument('--aug_n_select', type=int, default=2, help='augmentation n_select')
    parser.add_argument('--aug_prob', type=float, default=0.2, help='augmentation probability')
    
    # pretraining
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer32/models/model_best.pth', help='path of pretrained_weights')
    
    # k-space
    parser.add_argument('--use_kspace', action='store_true',default=False, help='using k-space')
    parser.add_argument('--wrapper_model', type=str, default='KI', help='k-space wrapper model')
    parser.add_argument('--kspace_path_processed', type=str, default='./Data/kspace_processed')
    parser.add_argument('--kspace_path_target_train', type=str, default='./Data/kspace')
    parser.add_argument('--kspace_path_target_val', type=str, default='./Data/kspace_Leaderboard')
    parser.add_argument('--kspace_processed_key', type=str, default='kspace_processed')
    parser.add_argument('--kspace_target_key', type=str, default='kspace')
    parser.add_argument('--pretrained_image', type=str, default=None)

    # args for wandb
    parser.add_argument('--use_wandb', action='store_true',default=False)
    parser.add_argument('--wandb_project', type=str, default ='Uformer(ItoI)',  help='name of wandb project')
    parser.add_argument('--wandb_name', type=str, default ='anonymous',  help='name for wandb run')

    # directories
    parser.add_argument('-rp', '--result_path', type=Path, default='./result', help='Directory for saving results')
    parser.add_argument('-cd', '--code_dir', type=Path, default='./Code', help='Directory for codes')
    parser.add_argument('--data_path_train', type=Path, default='./Data/image/train/', help='Directory of train data')
    parser.add_argument('--data_path_val', type=Path, default='./Data/image/test/', help='Directory of validation data')
    parser.add_argument('--input_key', type=str, default='image_grappa', help='Name of input key')
    parser.add_argument('--target_key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max_key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--config_dir', type=str, default='./Code/model_configs/', help='Directory of config file')
    parser.add_argument('--data_path_test', type=str, default='./Data/Image_Leaderboard/')
    parser.add_argument('--data_save_test', type=Path, default='./Data/Image_Reconsturcted/')
    
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')

    args = parser.parse_args()
    return args