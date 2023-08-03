# %load_ext tensorboard
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential, Dropout, Linear

from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

import argparse
import os
# from models import Model
from torchvision.transforms import Resize
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights
from train import train_one_epoch
from dataset import ClassificationDataset
import datetime
import tensorflow as tf



def parse_args():

    parser = argparse.ArgumentParser(description='Finetuning for cloth classification')

    # parser.add_argument('--model', type=str, help='Specify which pretrained model to use')

    # task specifications
    # parser.add_argument('--task', required=True, 
    #                     help='Training or testing')

    # paths
    parser.add_argument('--datapath', type=str,help='Data folder path')
    
    # training specifications
    parser.add_argument('--seed', type=int, default = 101, 
                        help='Seed to reproduce result and debug code')

    parser.add_argument('--epoch', type=int, default = 20, 
                        help='How many epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=1e-05,
                        help='Specify initial learning rate (default: 1e-05).')
    
    parser.add_argument('--save-model', action='store_true', help='Model save path')
    parser.add_argument('--save-path', type=str, default='checkpoints/',
                        help='Model save path')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Model gets saved after how many epochs') 

    parser.add_argument('--use-gpu', action='store_true',
                        help='Model uses gpu to train') 
    
                        
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()
    print(args)
    # log_dir = "logs/fit_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tb_writer = tf.compat.v1.summary.FileWriter(log_dir)


    # INPUT VALIDATIOn
    assert args.datapath != None, "give datapath"
    
    tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


    args.dataset = ClassificationDataset(args.datapath, shuffle = True, transform=tfms)
    split_ratio = 0.8
    train_split_len = int(args.dataset.__len__()*split_ratio)
    val_split_len = args.dataset.__len__() - train_split_len

    args.train_set, args.val_set = torch.utils.data.random_split(args.dataset, [train_split_len, val_split_len])
    args.train_loader, args.val_loader = DataLoader(args.train_set, batch_size=args.batch_size, shuffle=True), DataLoader(args.val_set, batch_size=args.batch_size, shuffle=True)

    # print(args.dataloader)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    # 
    # args.model = Model('efficientnet_v2_s', n_class=3)
    args.model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)

    print(args.model.classifier)

    args.model.classifier = Sequential(
        Dropout(p=0.4, inplace=True),
        Linear(in_features=1280, out_features=3, bias=True)
    )


    for name, parameter in args.model.features.named_parameters():
        parameter.requires_grad = False
    for name, parameter in args.model.classifier.named_parameters():
        parameter.requires_grad = True

    best_vloss = 1_000_000.
    args.loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizers specified in the torch.optim package
    args.optimizer = torch.optim.SGD(args.model.parameters(), lr=0.001, momentum=0.9)

    epoch_number = 0

    args.model = args.model.to('cuda:0')

    for epoch in range(args.epoch):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        args.model.train(True)
        # avg_loss = train_one_epoch(epoch_number, writer)
        # avg_loss = train_one_epoch(epoch_number, args, tb_writer=tb_writer)
        avg_loss = train_one_epoch(epoch_number, args)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        args.model.eval()

        # # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(args.val_loader):
                vinputs, vlabels = vdata

                vinputs, vlabels = vinputs.to('cuda:0'), vlabels.to('cuda:0')

                voutputs = args.model(vinputs)
                vloss = args.loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        # avg_vloss = 0
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()
        print(avg_loss, avg_vloss, epoch_number + 1)
        # Track best performance, and save the args.model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # args.model.save('checkpoint_{epoch}.pb')
            torch.save({
                'epoch': epoch,
                'model_state_dict': args.model.state_dict(),
                'optimizer_state_dict': args.optimizer.state_dict(),
                'loss': best_vloss,
                }, os.path.join(args.save_path, f'epoch_{epoch}.pt'))

        epoch_number += 1
        