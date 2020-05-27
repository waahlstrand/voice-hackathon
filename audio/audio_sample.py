#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:12:06 2020

@author: vws
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import waveform_to_features
import miniaudio
import os

    
class AudioSample:
    
    def __init__(self, 
                 file_path,
                 waveform, 
                 sample_rate, 
                 utterance, 
                 speaker_id, 
                 chapter_id, 
                 utterance_id):
        
        self.file_path      = file_path
        self.waveform       = waveform
        self.sample_rate    = sample_rate 
        self.utterance      = utterance
        self.speaker_id     = speaker_id
        self.chapter_id     = chapter_id
        self.utterance_id   = utterance_id
        
        self.features = None
        
    def play(self):
        
        
        
        stream = miniaudio.stream_file(self.file_path)
        with miniaudio.PlaybackDevice() as device:
            device.start(stream)
            input("Audio file playing in the background. Enter to stop playback: ")
        
    def featurize(self, n_features, split="test"):
        
        self.features = waveform_to_features(self.waveform, self.sample_rate, n_features, split=split)
        
    
    def plot(self, kind="waveform", n_features=None):
        
        if kind == "waveform":
            self._plot_waveform()
            
        elif kind == "mfcc":
            self._plot_mfcc(n_features)
            
        else:
            raise Exception("Plot kind must be waveform or mfcc.")
    
    
    
    def _plot_waveform(self):
        
        plt.plot(self.waveform.numpy().squeeze())
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        
    
    def _plot_mfcc(self, n_features=None):
        
        if self.features is None:
            if n_features is None:
                raise Exception("Please enter n_features to plot.")
                
            self.featurize(n_features)
            
        
        f,(ax1,ax2) = plt.subplots(2,1,sharex=True)
            
        frequencies = self.features.numpy().squeeze()[1:,:]
        energies    = self.features.numpy().squeeze()[0:1,:]
        g1 = sns.heatmap(frequencies, cmap="RdBu_r", 
                    cbar_kws={'label': 'MFCC Amplitude'}, ax=ax1)
        

        g2=sns.heatmap(energies, cmap="RdBu_r", 
                    cbar_kws={'label': 'MFCC Amplitude'}, ax=ax2)
        
        g2.set_xlabel("Time")        
        g1.set_ylabel("Component")
        g2.set_ylabel("Component")
        
