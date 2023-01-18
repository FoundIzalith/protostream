import lmu  # https://github.com/hrshtv/pytorch-lmu
import torch
import torchaudio
from langIdentifier import audioData, languageIdentifier
from torch.utils.data import random_split
import sys 
from os.path import exists

def main():
    if len(sys.argv) == 1:
        datapath = "./data/"
    else:
        datapath = sys.argv[1]

    if not exists(datapath):
        print("Error: datapath not found")
        exit()

    dataset = audioData(dataFrame, "./data/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    audio_train, audio_val = random_split(dataset, [train_size, val_size])

    dl_train = DataLoader(audio_train, batch_size = N_b, shuffle = True, num_workers = 2)
    dl_val   = DataLoader(audio_val, batch_size = N_b, shuffle = True, num_workers = 2)
        
if __name__ == "__main__":
    main()