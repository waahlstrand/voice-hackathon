# Data format and loading
The goal for the text-to-voice and voice-to-text projects is to produce an application capable of transcribing and synthesizing Arabic and Polish text and voice.

However, for ease of evaluation and interpretability, the Hackathon will be concentrated on transcribing and synthesizing English text and voice. There are plenty of good sources of English language data.

For the Hackathon we have prepared the [LibriSpeech corpus on English](http://www.openslr.org/12), which contains 1000 hours of audio data and transcriptions. The data is easily downloadable via PyTorch and Tensorflow.


## Data loading
We have prepared easy loading and formatting of the LibriSpeech data in PyTorch, using its standard datasets, see docs on [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

To inspect the data, the data must first be downloaded.

``` python
from data.utils import get_dataset

# Path to where the the data will be stored locally
data_dir = "DEST_DIR_FOR_DATA_DOWNLOAD" 

# Download and load LibriSpeech dataset
dataset  = get_dataset(data_dir, 
                       split="train", 
                       download=True, 
                       as_audiodataset=False)
```
This automatically downloads the data locally.


## Data format
The LibriSpeech data consists of .flac audio files, collected at 16 kHz from almost 10,000 speakers reciting passages from almost 45,000 books. 

The data is organized in folders by speakers, denoted by *speaker_id*, containing the chapter folders denoted by *chapter_id*. These contain the utterances/passages read by the speakers as .flac files. Each utterance is mapped by its *utterance_id* to a transcription file.

```
data_dir/LibriSpeech
    .- train-clean-100/
        |
        .- speaker_id/
            |
            .- chapter_id/
            |    |
            |    .- speaker_id-chapter_id.trans.txt
            |    |    
            |    .- speaker_id-chapter_id-utterance_id_1.flac
            |    |
            |    .- speaker_id-chapter_id-utterance_id_2.flac
            |    |
            |    ...
            |
            .- chapter_id_2/
                | ...
```
A sample of the data returns the waveform as PyTorch tensor, the sample rate (always 16 kHz) the transcribed utterance, as well as the corresponding IDs. 

```python
waveform,\
sample_rate,\
utterance,\
speaker_id,\
chapter_id,\
utterance_id = dataset[i]
```

An example of the data may look like the following:
```python
>> dataset[0]

(tensor([[ 0.0037,  0.0028,  0.0025,  ..., -0.0015, -0.0010, -0.0002]]),
 16000,
 'ALL OF THIS IS KNOWN BY EVERYBODY TO BE A NECESSARY AND UNIVERSAL ADJUNCT OF THE HOTEL BUSINESS THE INSPIRATION OF THE BOOK',
 460,
 172357,
 23)
```

### AudioSample
The data may also be accessed through the convenience AudioSample objects. These wrap the information of the audio file and add some convenient utilities. Load a dataset as AudioSamples by writing
``` python
# Download and load LibriSpeech dataset
dataset  = get_dataset(data_dir, 
                       split="train", 
                       download=True, 
                       as_audiodataset=True)
```
The AudioSamples contain utilities for plotting the raw waveform of the utterance for inspection, as well as the full path of the audiofile.

```python
>> sample = dataset[0]

>> sample.file_path
'data_dir/LibriSpeech/train-clean-100/460/172357/460-172357-0023.flac'

>> sample.utterance
'ALL OF THIS IS KNOWN BY EVERYBODY TO BE A NECESSARY AND UNIVERSAL ADJUNCT OF THE HOTEL BUSINESS THE INSPIRATION OF THE BOOK'

>> sample.plot(kind="waveform")
```
![Waveform example](/examples/waveform_ex.png)

These features are useful for data exploration, and gives a better overview of the data. A bonus utility is the ability to play the sample in the computer speakers, using

```python
>> sample.play()
```
Use this to confirm that the transcription seems correct if you have problems with your model or data.
