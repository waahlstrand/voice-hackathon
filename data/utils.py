#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:51:02 2020

@author: vws
"""
import os
import torch
import torchaudio
import torch.utils.data
import torch.nn as nn

import audio.utils
from audio.audio_sample import AudioSample
import data.datasets
TRAIN_URL   = "train-clean-100"
TEST_URL    = "test-clean"

def get_dataset(data_dir, split="train", download=True, as_audiodataset=True):
    
    print(as_audiodataset)
    
    if not os.path.isdir(data_dir):
        print("- Creating data directory.")
        os.makedirs(data_dir)
    else:
        print("- Data directory already exists.")
    
    if split == "train":
        print("- Fetching training data.")
        
        if as_audiodataset:
            dataset = data.datasets.AudioDataset(data_dir, url=TRAIN_URL, download=download)
        else:
            dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url=TRAIN_URL, download=download)
            
        print("Data fetched!")
        return dataset
    elif split == "test":
        print("- Fetching test data.")
        
        if as_audiodataset:
            dataset = data.datasets.AudioDataset(data_dir, url=TEST_URL, download=download)
        else:
            dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url=TEST_URL, download=download)
            
        print("Data fetched!")
        return dataset
    
    
    else:
        raise Exception('Split should be train or valid')
    
    
def get_dataloader(data_dir, 
                   batch_size, 
                   alphabet, 
                   n_features, 
                   use_cuda, 
                   split="train",
                   download=True,
                   as_audiodataset=False):
    
    dataset = get_dataset(data_dir, split, download, as_audiodataset)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    collate_fn = lambda x: collate_audio_to_features(x, 
                                                                                     alphabet, 
                                                                                     n_features,
                                                                                     split),
                                    **kwargs)
    
    return data_loader
    
    
def collate_audio_to_features(batch, alphabet, n_features, split):
    
    
    features_list = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    # Loop a batch of data
    for item in batch:
        
            
        if isinstance(item, AudioSample):
            item.featurize(n_features, split)
            
            features = item.features
            
            label    = torch.Tensor(alphabet.encode(item.utterance.lower()))
        else:
            waveform, sample_rate, utterance, _, _, _ = item
            
            # Transform waveform to power features
            features = audio.utils.waveform_to_features(waveform = waveform, 
                                                        sample_rate = sample_rate, 
                                                        n_features = n_features, 
                                                        split = split)
            
            label = torch.Tensor(alphabet.encode(utterance.lower()))
            
        # Save batch content for collation
        features_list.append(features.squeeze().transpose(0,1))
        labels.append(label)
        
        # Save the lengths to make sure we know how long each sample is
        input_lengths.append(features.shape[0]//2)
        label_lengths.append(len(label))
        
    # Padded sequences
    features_list = nn.utils.rnn.pad_sequence(features_list, batch_first=True).transpose(1, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return features_list, labels, input_lengths, label_lengths
    
    

   
        
