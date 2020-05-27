#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:29:34 2020

@author: vws
"""
from data.utils import get_dataloader
from data.Alphabet import Alphabet


# %%
alphabet = Alphabet("data/english_alphabet.txt")

dataloader = get_dataloader("librispeech", 
                            batch_size=8, 
                            use_cuda=False, 
                            alphabet=alphabet,
                            n_features=128,
                            split="train")

# %%
X, y, X_lens, y_lens = next(iter(dataloader))

# %%
import matplotlib.pyplot as plt
from data.utils import get_dataset
data_dir = "librispeech"
dataset = get_dataset(data_dir, download=True, split="train", as_audiodataset=True)

# %%
sample = dataset[0]

sample.plot(kind="mfcc", n_features=12)
plt.show()
#sample.plot(kind="waveform")
#plt.show()

# %%
sample.featurize(5)
features = sample.features.numpy().squeeze()