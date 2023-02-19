import lmu  # https://github.com/hrshtv/pytorch-lmu
import torch
import torchaudio
from langIdentifier import audioData, languageIdentifier
from torch.utils.data import random_split, dataset, DataLoader
import sys 
from os.path import exists
import pandas 
from pathlib import Path

def main():
    if len(sys.argv) == 1:
        datapath = Path.cwd()/'data'
        metadata = Path.cwd()/'audio_data.csv'
    elif len(sys.argv) == 2:
        datapath = sys.argv[1]
        metadata = Path.cwd()/'audio_data.csv'
    else:
        datapath = sys.argv[1]
        metadata = sys.argv[2]
    
    if not exists(datapath):
        print("Error: datapath not found")
        exit()

    if not exists(metadata):
        print("Error: metadata not found")
        exit()

    dataFrame = pandas.read_csv(metadata)
    dataFrame.head()

    dataset = audioData(dataFrame, "./data/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    audio_train, audio_val = random_split(dataset, [train_size, val_size])

    dl_train = DataLoader(audio_train, batch_size = N_b, shuffle = True, num_workers = 2)
    dl_val   = DataLoader(audio_val, batch_size = N_b, shuffle = True, num_workers = 2)
        
if __name__ == "__main__":
    main()