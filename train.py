# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:26:48 2020

@author: vwahlstr
"""
import os
#from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim 
import torch.nn.functional
#import torchaudio
import numpy as np

import logging


def train(model: nn.Module, 
          device: str, 
          train_loader, 
          criterion, 
          optimizer, 
          scheduler, 
          epoch):
    
    
    # Set model to train
    model.train()
    data_len = len(train_loader.dataset)
    
    
    for batch_idx, batch in enumerate(train_loader):
            
        # Get data from loader
        spectrograms, labels, input_lengths, label_lengths = batch 
        spectrograms, labels = spectrograms.to(device), labels.to(device)


        # Zero the optimizer accumulation of gradients
        optimizer.zero_grad()

        # Forward pass the data 
        output = model(spectrograms)  # (batch, time, n_class)
        #output = torch.nn.functional.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        # Calculate loss
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
            
        # Log the data to board
        experiment.log_metric('loss', loss.item(), step=iter_meter.get())
        experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())
            
        # Update optimizer and learning rate scheduler
        optimizer.step()
        scheduler.step()
        iter_meter.step()
            
            
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(spectrograms), 
                data_len,
                100. * batch_idx / len(train_loader), 
                loss.item()))
