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

    dataset= audioData(dataFrame, "./data/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    audio_train, audio_val = random_split(dataset, [train_size, val_size])

    perm = torch.load("permutation.pt").long() # created using torch.randperm(784)
    ds_train = psMNIST(audio_train, perm)
    ds_val   = psMNIST(audio_val, perm) 

    dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 2)
    dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = True, num_workers = 2)
        
if __name__ == "__main__":
    main()