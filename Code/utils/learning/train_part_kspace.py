import shutil
import time
import os
import datetime
from tqdm import tqdm
import glob
import random
import wandb

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
from utils.data.data_loader import KspaceDataLoaderTrain, KspaceDataLoaderVal
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

        yaml_file = os.path.join(config_dir, f'k_{model_name}.yaml')
        yaml_dict = yaml_to_dict(yaml_file)
        os.system(f'cp {self.args.code_dir}/model_configs/k_{model_name}.yaml {self.args.result_path}/k_{model_name}.yaml')
        model_config = yaml_dict.get('model', False)
        if model_config is False:
            raise Exception("There's no configuration for model!")
        return getattr(__import__("utils.model", fromlist=[""]), model), yaml_dict

    def _get_wrapper_model(self, model_abs, model_config, img_pretrain=None):
        wmodel_name = self.args.wrapper_model.lower()
        if wmodel_name == 'ki':
            wmodel = 'KI'
        elif wmodel_name == 'ik':
            wmodel = 'IK'
        else:
            raise Exception(f'Wrapper Model {self.args.wrapper_model} does not exist')
        model = getattr(__import__("utils.model.wrapper", fromlist=[""]), wmodel)(model_abs, model_config)
        
        if img_pretrain != None:
            print('===> Loading weights...')
            ckpt = torch.load(img_pretrain)
            own_state = model.i_model1.state_dict()
            new_state_dict = {}
            for name, params in ckpt["state_dict"].items():
                for on, op in own_state.items():
                    if name[7:] in on:
                        params.requires_grad = False
                        new_state_dict.update({on: params})
            model.i_model1.load_state_dict(new_state_dict)
            # for params in model.i_model1.parameters():
            #     params.requires_grad = False
        return model

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
                                                        100, #self.args.num_epochs, 
                                                        eta_min=1e-6)
        elif sched_str == 'step':
            return optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=self.args.step_size, 
                                            gamma=self.args.step_gamma)
        else:
            raise Exception(f"No scheduler such as {optim_str}")

    def _make_kspace_dataset(self):
        print('===> Loading kspace datasets')
        train_dataset = KspaceDataLoaderTrain(kspace_dir=self.args.kspace_path_processed, 
                                        image_dir=self.args.data_path_train, 
                                        kspace_key=self.args.kspace_processed_key, 
                                        image_input_key=self.args.input_key, 
                                        augmentation=self.args.augmentation, aug_n_select=self.args.aug_n_select, aug_prob=self.args.aug_prob,
                                        norm=self.args.normalization
                                        )
        val_dataset = KspaceDataLoaderVal(kspace_dir=self.args.kspace_path_processed, 
                                    image_dir=self.args.data_path_val, 
                                    kspace_key=self.args.kspace_processed_key, 
                                    image_input_key=self.args.input_key, 
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

    def train(self):
        if self.debug:
            print('Start training')
        # data parallel
        model_abs, self.config = self._select_model(self.args.config_dir)
        model = self._get_wrapper_model(model_abs, self.config['model'], img_pretrain=self.args.pretrained_image)
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
        train_dataset, val_dataset= self._make_kspace_dataset()

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

            len_train_dataset = len(train_dataset)
            len_val_dataset = len(val_dataset)

            # train
            pbar = tqdm(enumerate(train_dataset), total=len_train_dataset)
            for i, data in pbar:
                optimizer.zero_grad()
                batch_loss = 0
                len_layer = data[1].shape[0]
                for l in range(data[1].shape[0]):
                    kspace = data[0]
                    if kspace!=None:
                        kspace = kspace[l:l+1].cuda()
                    img_input = data[1][l:l+1].cuda()
                    img_target = data[2][l:l+1].cuda().reshape((1,img_input.shape[-2],img_input.shape[-1]))

                    with torch.cuda.amp.autocast():
                        restored = model(img_input, kspace)
                        restored = torch.clamp(restored,0,1)
                        restored = restored.reshape((1,img_input.shape[-2],img_input.shape[-1]))
                        loss = loss_func(img_target, restored)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if np.isnan(loss.item()):
                        print(restored)
                        raise Exception("NaN loss occured!")
                    batch_loss += loss.item()

                epoch_loss += batch_loss/len_layer
                if self.wandb:
                    wandb.log({"train batch loss": 1-batch_loss/len_layer})

                pbar.set_description(
                    f"Train:\t[{epoch:03d}/{self.args.num_epochs:03d}] "
                    f"Loss:\t{epoch_loss / (i+1):.6f}"
                )
                pbar.set_postfix_str(
                    f"lr: [{scheduler.get_last_lr()[0]:.6f}]"
                )
            self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=False, lastest=True)
            
            # validation
            with torch.no_grad():
                model.eval()
                # ssim_val = []
                pbar = tqdm(enumerate(val_dataset), total=len_val_dataset)
                val_loss_sum = 0
                for ii, data_val in pbar:
                    '''
                    data from val_loader:
                    => clean, noisy, clean_filename, noisy_filename, clean_minmax, noisy_minmax
                    '''
                    batch_loss = 0
                    len_layer = data_val[1].shape[0]
                    for l in range(len_layer):
                        kspace = data_val[0]
                        if kspace!=None:
                            kspace = kspace[l:l+1].cuda()
                        img_input = data_val[1][l:l+1].cuda()
                        img_target = data_val[2][l:l+1].cuda().reshape((1,img_input.shape[-2],img_input.shape[-1]))
                        input_minmax = data_val[3]
                        target_minmax = data_val[4]

                        with torch.cuda.amp.autocast():
                            restored = model(input_)
                            restored = torch.clamp(restored,0,1)
                            restored = restored.reshape((1,img_input.shape[-2],img_input.shape[-1]))
                        restored_in = self._inverse_normalize(restored, input_minmax)
                        target_in = self._inverse_normalize(target, target_minmax)
                        val_loss = val_loss_func(restored_in, target_in).item()
                        batch_loss += val_loss

                    val_loss_sum += batch_loss/len_layer
                    pbar.set_description(
                        f"Val:\t[{epoch:03d}/{self.args.num_epochs:03d}] "
                        f"Loss:\t{val_loss_sum/(ii+1):.6f}"
                    )

                val_loss_avg = 1-(val_loss_sum/len_val_dataset)

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

            epoch_loss = 1-(epoch_loss/len_train_dataset)
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            
            self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=False, lastest=True)
            if epoch%self.args.checkpoint==0:
                self._save_model(self.args, epoch, model, optimizer, best_ssim, is_new_best=False)
            if self.wandb:
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
                wandb.log({"train loss": epoch_loss})

        print("End Training...")


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

