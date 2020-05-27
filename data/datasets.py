# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:36:37 2020

@author: vwahlstr
"""

from torchaudio.datasets import LIBRISPEECH
#from torch.util.data import Dataset
import torch
from audio.audio_sample import AudioSample
import os

class AudioDataset(LIBRISPEECH):
    
    def __getitem__(self, idx):
        
        waveform,\
        sample_rate,\
        utterance,\
        speaker_id,\
        chapter_id,\
        utterance_id = super(AudioDataset, self).__getitem__(idx)
    
        file_dest = self._walker[idx].replace("-", os.sep)
        file_dest = file_dest.split(os.sep)[0:2]
        file_path = os.path.join(self._path, *file_dest, self._walker[idx]+".flac")
        
        return AudioSample(file_path, waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
    
    