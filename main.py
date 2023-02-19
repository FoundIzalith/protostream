import lmu  # https://github.com/hrshtv/pytorch-lmu
import torch
import torchaudio
from langIdentifier import audioData, languageIdentifier
from torch.utils.data import random_split, dataset, DataLoader
import sys 
from os.path import exists
import pandas 
from pathlib import Path

N_x = 2 # Dimension of input
N_c = 176 # Number of classes, 176 for 176 languages
N_h = 212 # hidden layer size
N_m = 256 # memory size
batch = 100 # batch size
N_epochs = 10

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

    dl_train = DataLoader(audio_train, batch_size = batch, shuffle = True, num_workers = 2)
    dl_val   = DataLoader(audio_val, batch_size = batch, shuffle = True, num_workers = 2)

    model = Model(
        input_size = N_x #Dimensioon of input
        output_size = N_c #Number of classes # 176 for 176 languages 
        hidden_size = N_h #Dimension of hidden state
        memory_size =  N_m #dimension of memory
        learn_a = true
        learn_b = true    
    )
        
if __name__ == "__main__":
    main()