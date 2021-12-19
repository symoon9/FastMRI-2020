#!/usr/bin/env bash

dir_name=`date +%Y-%m-%d_%H-%M-%S.%6N`

cd /FastMRI/SNU_challenge
mkdir -p results/$dir_name

python3 Code/train.py \
--debug true \
--gpu '0,1' \
--batch_size 16 \
--num_epochs 100 \
--learning_rate 1e-3 \
--checkpoint 30 \
--augmentation \
--aug_n_select 2 \
--aug_prob 0.15 \
--train_workers 1 \
--eval_workers 1 \
--model 'edcnn' \
--optimizer 'adamp' \
--result_path './results/'${dir_name}'/' \
--code_dir './Code' \
--data_path_train './Data/image/train/' \
--data_path_val './Data/image/test/' \
--input_key 'image_grappa' \
--config_dir './Code/model_configs/' \
--val_loss_mask \
--max_grad_norm 5.0