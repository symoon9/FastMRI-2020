import shutil
import time
import os
import datetime
from tqdm import tqdm
import glob
import random
import wandb
import h5py

from natsort import natsorted
from einops import rearrange, repeat
from pdb import set_trace as stx

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import adamp

from collections import defaultdict, OrderedDict
from utils.data.load_data import create_data_loaders
from utils.data.data_loader import DataLoaderTrain, DataLoaderVal, DataLoaderTest
from utils.loss.loss_function import ssim_loss, SSIMLoss, ChallengeSSIM
from utils.common.utils import save_reconstructions, yaml_to_dict


class Trainer():
    def __init__(self, args):
        self.args = args
        self.debug = self.args.debug
        self.wandb = self.args.use_wandb
        self.kspace = self.args.use_kspace

        # set GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        torch.backends.cudnn.benchmark = True

        if self.wandb:
            wandb_name = str(self.args.result_path.stem)
            wandb.init(project=self.args.wandb_project, name=wandb_name, entity='duya')

    def _select_model(self, config_dir):
        model_name = self.args.model.lower()
        if model_name == 'unet':
            model = 'Unet'
        elif model_name == 'uformer':
            model = 'Uformer'
        elif model_name == 'edcnn':
            model = 'EDCNN'
        elif model_name == 'dudenet':
            model = "DnCNN"
        elif model_name == 'adnet':
            model = "ADNet"
        elif model_name == 'duya_net':
            model = "DuyaNet"
        else:
            raise Exception(f'Model {self.args.model} does not exist')

        yaml_file = os.path.join(config_dir, f'{model_name}.yaml')
        yaml_dict = yaml_to_dict(yaml_file)
        os.system(f'cp {self.args.code_dir}/model_configs/{model_name}.yaml {self.args.result_path}/{model_name}.yaml')
        model_config = yaml_dict.get('model', False)
        if model_config is False:
            raise Exception("There's no configuration for model!")
        return getattr(__import__("utils.model", fromlist=[""]), model)(**model_config), yaml_dict

    def _load_checkpoint(self, model, weights):
        checkpoint = torch.load(weights)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    def _load_optim(self, optimizer, weights):
        checkpoint = torch.load(weights)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for p in optimizer.param_groups: lr = p['lr']
        return lr

    def _optimizer(self, optim_str, model):
        if optim_str == 'adam':
            return optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)
        elif optim_str == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)
        elif optim_str == 'adamp':
            return adamp.AdamP(model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)
        # elif optim_str == 'nadam':
        else:
            raise Exception(f"No optimizer such as {optim_str}")

    def _scheduler(self, sched_str, optimizer):
        if sched_str == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        self.args.num_epochs, 
                                                        eta_min=1e-6)
        elif sched_str == 'step':
            return optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=self.args.step_size, 
                                            gamma=self.args.step_gamma)
        else:
            raise Exception(f"No scheduler such as {optim_str}")

    def _make_dataloader(self):
        print('===> Loading datasets')
        img_options_train = {'patch_size': self.config.get('patch_size', None)}
        train_dataset = DataLoaderTrain(self.args.data_path_train, img_options_train, 
                                        input_dir=self.args.input_key, 
                                        augmentation=self.args.augmentation, aug_n_select=self.args.aug_n_select, aug_prob=self.args.aug_prob,
                                        norm=self.args.normalization
                                        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True, 
                                    num_workers=self.args.train_workers, pin_memory=True, drop_last=False)

        val_dataset = DataLoaderVal(self.args.data_path_val, input_dir=self.args.input_key, norm=self.args.normalization)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size, shuffle=False, 
                num_workers=self.args.eval_workers, pin_memory=False, drop_last=False)

        len_trainset = train_dataset.__len__()
        len_valset = val_dataset.__len__()
        print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
        return train_dataset, train_loader, val_dataset, val_loader

    def _make_kspace_dataset(self):
        print('===> Loading kspace datasets')
        train_dataset = DataLoaderTrain(input_dir=self.args.kspace_path_processed, 
                                        target_dir=self.args.kspace_path_target_train, 
                                        input_key=self.args.kspace_processed_key, 
                                        target_key=self.args.kspace_target_key, 
                                        # augmentation=self.args.augmentation, aug_n_select=self.args.aug_n_select, aug_prob=self.args.aug_prob,
                                        norm=self.args.normalization
                                        )
        val_dataset = DataLoaderVal(input_dir=self.args.kspace_path_processed, 
                                    target_dir=self.args.kspace_path_target_val, 
                                    input_key=self.args.kspace_processed_key, 
                                    target_key=self.args.kspace_target_key, 
                                    norm=self.args.normalization)
        len_trainset = train_dataset.__len__()
        len_valset = val_dataset.__len__()
        print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
        return train_dataset, val_dataset

    def _load_start_epoch(self, weights):
        checkpoint = torch.load(weights)
        epoch = checkpoint["epoch"]
        return epoch

    def _load_pretrained_model(self, weights, model, optimizer):
        checkpoint = torch.load(weights)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # for p in optimizer.param_groups: lr = p['lr']
        epoch = checkpoint["epoch"]

        return epoch #lr

    def _inverse_normalize(self, array, minmax):
        array_list = []
        min_list = minmax[0].tolist()
        max_list = minmax[1].tolist()
        for layer, minmax_one in enumerate(zip(min_list, max_list)):
            min_, max_ = minmax_one
            array_npy = array[layer].cpu().numpy()
            array_list.append(array_npy*(max_-min_)+min_)
        array_npy = np.array(array_list, dtype=np.float32)
        return torch.from_numpy(array_npy)

    def _inverse_normalize_one(self, array, minmax):
        min_ = minmax[0]
        max_ = minmax[1]
        array_npy = array.cpu().numpy()
        array_in = array_npy*(max_-min_)+min_
        return torch.from_numpy(array_in)

    def train(self):
        if self.debug:
            print('Start training')
        # data parallel
        model, self.config = self._select_model(self.args.config_dir)
        model = torch.nn.DataParallel(model)
        model.cuda()

        start_epoch = 0

        # get optimizer & scheduler
        learning_rate = self.args.learning_rate
        optimizer = self._optimizer(self.args.optimizer.lower(), model)
        scheduler = self._scheduler(self.args.scheduler.lower(), optimizer)

        # get loss function
        loss_func = ChallengeSSIM(mask=False).cuda()
        val_loss_func = ChallengeSSIM(mask=self.args.val_loss_mask)

        # dataloader
        train_dataset, train_loader, val_dataset, val_loader = self._make_dataloader()

        # resume training: train again with pretrained model
        if self.args.resume:
            assert os.path.isfile(self.args.pretrain_weights)
            start_epoch = self._load_pretrained_model(self.args.pretrain_weights, model, optimizer)
            # start_epoch, learning_rate = self._load_pretrained_model(self.args.pretrain_weights, model, optimizer)

        # set early stopping
        EARLY_STOP = self.args.early_stop
        if EARLY_STOP:
            patience = self.args.early_stop_patience
            patience_cnt = 0
            early_stop_occur=False
        
        best_ssim = 0
        best_epoch = 0
        best_iter = 0
        torch.cuda.empty_cache()

        print(f'............... {self.args.model} ...............')
        for epoch in range(start_epoch, self.args.num_epochs):
            model.train()
            epoch_loss = 0
            epoch_start_time = time.time()

            len_train_loader = len(train_loader)
            len_val_loader = len(val_loader)

            # train
            pbar = tqdm(enumerate(train_loader), total=len_train_loader)
            for i, data in pbar:
                optimizer.zero_grad()

                target = data[0].cuda()
                input_ = data[1].cuda()

                with torch.cuda.amp.autocast():
                    restored = model(input_)
                    restored = torch.clamp(restored,0,1)
                    loss = loss_func(target, restored)

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm != -1.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if np.isnan(loss.item()):
                    print(restored)
                    raise Exception("NaN loss occured!")

                epoch_loss += loss.item()
                if self.wandb:
                    wandb.log({"train batch loss": 1-loss.item()})

                pbar.set_description(
                    f"Train:\t[{epoch:03d}/{self.args.num_epochs:03d}] "
                    f"Loss:\t{epoch_loss / (i+1):.6f}"
                )
                pbar.set_postfix_str(
                    f"lr: [{scheduler.get_last_lr()[0]:.6f}]"
                )

            # validation
            with torch.no_grad():
                model.eval()
                # ssim_val = []
                pbar = tqdm(enumerate(val_loader), total=len_val_loader)
                val_loss_sum = 0
                for ii, data_val in pbar:
                    '''
                    data from val_loader:
                    => clean, noisy, clean_filename, noisy_filename, clean_minmax, noisy_minmax
                    '''
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    target_filename = data_val[2]
                    input_filename = data_val[3]
                    target_minmax = data_val[4]
                    input_minmax = data_val[5]

                    with torch.cuda.amp.autocast():
                        restored = model(input_)
                        restored = torch.clamp(restored,0,1)
                    restored_in = self._inverse_normalize(restored, input_minmax)
                    target_in = self._inverse_normalize(target, target_minmax)
                    val_loss = val_loss_func(restored_in, target_in).item()
                    val_loss_sum += val_loss
                    pbar.set_description(
                        f"Val:\t[{epoch:03d}/{self.args.num_epochs:03d}] "
                        f"Loss:\t{val_loss/(i+1):.6f}"
                    )

                val_loss_avg = 1-(val_loss_sum/len_val_loader)

                if val_loss_avg >= best_ssim:
                    patience_cnt = 0
                    best_ssim = val_loss_avg
                    best_epoch = epoch
                    self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=True)

                elif EARLY_STOP:
                    patience_cnt += 1
                    if patience_cnt==patience:
                        early_stop_occur=True

                if self.wandb:
                    wandb.log({"val loss":val_loss_avg})
                print("[Ep %d \tSSIM: %.4f\t] ----  [best_Ep %d \tBest_SSIM %.4f] " % (epoch, val_loss_avg, best_epoch, best_ssim))
                
                torch.cuda.empty_cache()

            scheduler.step()

            epoch_loss = 1-(epoch_loss/len_train_loader)
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            
            self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=False, lastest=True)
            if epoch%self.args.checkpoint==0:
                self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=False)
            if self.wandb:
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
                wandb.log({"train loss": epoch_loss})

        print("End Training...")
        return model


    def _validate(self, args, model, data_loader):
        model.eval()
        reconstructions = defaultdict(dict)
        targets = defaultdict(dict)
        inputs = defaultdict(dict)
        start = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            for iter, data in pbar:
                input, target, _, fnames, slices = data
                input = input.cuda(non_blocking=True)
                output = model(input)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    targets[fnames[i]][int(slices[i])] = target[i].numpy()
                    inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()
                pbar.set_description(
                    f"Valid "
                    # f"Loss:{total_loss / total:.4f} "
                    f'Time:{time.perf_counter() - start_iter:.4f}s'
                )
        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        for fname in targets:
            targets[fname] = np.stack(
                [out for _, out in sorted(targets[fname].items())]
            )
        for fname in inputs:
            inputs[fname] = np.stack(
                [out for _, out in sorted(inputs[fname].items())]
            )
            metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
        num_subjects = len(reconstructions)
        return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


    def _save_model(self, args, epoch, model, optimizer, best_val_loss, is_new_best, lastest=False):
        if is_new_best:
            file_name = 'model_best.pt' 
        elif not is_new_best and not lastest:
            file_name = 'model_checkpoint.pt'
        elif lastest:
            file_name = 'model_lastest.pt'
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            },
            f=os.path.join(args.exp_dir, file_name)
        )

    def test(self, model):
        with torch.no_grad():
            self.args.data_save_test.mkdir(parents=True, exist_ok=True)

            ckpt = torch.load(self.args.result_path / "checkpoints/model_best.pt")
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            test_dataset = DataLoaderTest(self.args.data_path_test, input_key=self.args.input_key)
            pbar = tqdm(enumerate(test_dataset), total=len(test_dataset))
            
            for i, data in pbar:
                input_ = data[0].cuda()
                input_minmax = data[1]
                input_filename = data[2]
                layers = []
                for l in range(input_.shape[0]):
                    with torch.cuda.amp.autocast():
                        restored = model(input_[l:l+1])
                        restored = torch.clamp(restored,0,1)
                    restored_in = self._inverse_normalize_one(restored, input_minmax[l]).numpy()
                    layers.append(restored_in)
                layers = np.array(layers)
                hf = h5py.File(os.path.join(self.args.data_save_test, input_filename), "w")
                hf.create_dataset('reconstruction', data=layers)
                hf.close()

        print("finish test!")

                    
        
