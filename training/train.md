# Instructions for training
## Loading data for training
For training you may use PyTorch's data loaders, which automatically handle batching, shuffling and end formatting of the data. See docs on [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). 

We have prepared automated dataloaders for the LibriSpeech dataset, [accessible similarly to the dataset](data/data.md#Data_loading). You can also access the dataset itself from the dataloader afterwards.

```python
from data.utils import get_dataloader
from data.Alphabet import Alphabet

# Path to where the the data will be stored locally
data_dir = "DEST_DIR_FOR_DATA_DOWNLOAD" 
batch_size = 8
use_cuda   = False # Cuda recommended for training
n_features = 16

# Class to encode the writing to numerical indices.
alphabet = Alphabet("data/english_alphabet.txt")

dataloader = get_dataloader(data_dir, 
                            batch_size=batch_size, 
                            use_cuda=use_cuda, 
                            alphabet=alphabet,
                            n_features=n_features,
                            split="train")
```