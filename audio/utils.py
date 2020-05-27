#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:13:35 2020

@author: vws
"""

import os
import torch
import torchaudio
import torch.utils.data as data
import torch.nn as nn

def waveform_to_features(waveform, 
                         sample_rate, 
                         n_features= 128, 
                         n_fft = 256,
                         win_length = None,
                         hop_length = None,
                         f_min = 0,
                         f_max = None,
                         split="test", 
                         feature_type="mfcc"):
    
    # Common default values
    win_length  = win_length if win_length else n_fft
    hop_length  = hop_length if hop_length else win_length // 2
    f_max       = f_max if f_max else sample_rate // 2 # Nyquist rate
    
    # Choose the type of feature transform
    if feature_type == "spectrogram":
        feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, 
                                                                 n_fft = n_fft,
                                                                 win_length = win_length,
                                                                 hop_length = hop_length,
                                                                 f_min = f_min,
                                                                 f_max = f_max,
                                                                 n_mels=n_features)
    elif feature_type == "mfcc":
        melkwargs = {"n_fft": n_fft,
                     "win_length": win_length,
                     "hop_length": hop_length,
                     "f_min": f_min,
                     "f_max": f_max,
                     "n_mels": n_features}
        
        feature_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                       n_mfcc=n_features, 
                                                       melkwargs=melkwargs)
        
    else:
        raise Exception("Feature type must be spectrogram or mfcc")
        
    # Create feature transform pipeline
    if split == "train":
        
        waveform_transform = nn.Sequential(
                feature_transform,
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
                )
        
    elif split == "test":
        
        waveform_transform = nn.Sequential(
                feature_transform
                )
        
    else:
        raise Exception('Split should be train or valid')    
    
    return waveform_transform(waveform)