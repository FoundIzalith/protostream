import lmu  # https://github.com/hrshtv/pytorch-lmu
import torch
import torchaudio
from torch import optim, nn
from langIdentifier import audioData, languageIdentifier
from torch.utils.data import random_split, dataset, DataLoader
import sys 
from os.path import exists
import pandas 
from pathlib import Path

N_x = 860 # Dimension of input
N_c = 176 # Number of classes, 176 for 176 languages
N_h = 212 # hidden layer size
N_m = 256 # memory size
N_t = 784 # Number of time steps
batch = 64 # batch size
N_epochs = 10

def main():
    if len(sys.argv) == 1:
        datapath = 'data'
        metadata = 'audio_data.csv'
    elif len(sys.argv) == 2:
        datapath = sys.argv[1]
        metadata = 'audio_data.csv'
    else:
        datapath = sys.argv[1]
        metadata = sys.argv[2]
    
    if not exists(datapath):
        print("Error: datapath not found")
        exit()

    if not exists(metadata):
        print("Error: metadata not found")
        exit()

    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.cuda.empty_cache()
    else:
        DEVICE = "cpu"

    dataFrame = pandas.read_csv(metadata)
    language = dataFrame['Language'].unique()
    dataFrame.head()

    languages = []
    # Convert language name into an int id
    for entry in language:
        id = ord(entry[0]) * 100000
        for char in entry: 
            id += ord(char)
        languages.append(id)

    languages = torch.tensor(languages)

    dataset = audioData(dataFrame, datapath)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    audio_train, audio_val = random_split(dataset, [train_size, val_size])

    dl_train = DataLoader(audio_train, batch_size = batch, shuffle = True, num_workers = 2)
    dl_val   = DataLoader(audio_val, batch_size = batch, shuffle = True, num_workers = 2)

    model = languageIdentifier(
        input_size = N_x, # Dimension of input
        output_size = N_c, #Number of classes # 176 for 176 languages 
        hidden_size = N_h, #Dimension of hidden state
        memory_size =  N_m, #dimension of memory
        theta = N_t,
        learn_a = True,
        learn_b = True    
    )
    model = model.to(DEVICE)

    languageIdentifier.countParameters(model)
    optimizer = optim.Adam(params = model.parameters())

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(N_epochs):
        print("Epoch ", epoch)

        train_loss, train_acc = languageIdentifier.train(model, dl_train, optimizer, criterion, DEVICE, languages)
        val_loss, val_acc = languageIdentifier.validate(model, dl_val, criterion, DEVICE, languages)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc*100:.2f}%")
        print(f"Validation Loss: {val_loss:.3f} |  Validation Accuracy: {val_acc*100:.2f}%")
        print()

    torch.save(model.state_dict(), "trained_model.pt")

    print("All done!")
        
if __name__ == "__main__":
    main()