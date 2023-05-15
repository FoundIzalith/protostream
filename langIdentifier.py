import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from lmu import LMUFFT
from tqdm.notebook import tqdm

class AudioUtil():
    #Load audio file, return as tensor
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        #Do nothing
        if (sig.shape[0] == new_channel):
            return aud

        # Convert from stereo to mono by selecting only the first channel
        if (new_channel == 1):
            resig = sig[:1, :]
        # Convert from mono to stereo by duplicating the first channel
        else:
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        #Nothing to do 
        if (sr == newsr):
            return aud

        # Resample first channel
        num_channels = sig.shape[0]
    
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
        # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)   

    @staticmethod
    def timeshift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def spectroaugment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

class audioData(Dataset):
    def __init__(self, dataFrame, path, language):
        self.path = path
        self.duration = 10000
        self.sampleRate = 44100
        self.channels = 2
        self.dataFrame = dataFrame
        self.languages = language.tolist()
        
    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        audioFile = self.path + self.dataFrame.loc[idx, 'Sample Filename']
        class_name = self.dataFrame.loc[idx, 'Language']
        class_id = self.languages.index(class_name)

        file = AudioUtil.open(audioFile)
        #Ensure consistent data
        resampled = AudioUtil.resample(file, self.sampleRate)
        rechanneled = AudioUtil.rechannel(resampled, self.channels)
        padded = AudioUtil.pad_trunc(rechanneled, self.duration)
        shifted = AudioUtil.timeshift(padded, 0.4)
        spectro = AudioUtil.spectrogram(shifted, n_mels=64, n_fft=1024, hop_len=None)
        spectrogram = AudioUtil.spectroaugment(spectro, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return spectrogram, class_id

class langIdentifierLMU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, seq_len, theta):
        super(langIdentifierLMU, self).__init__()
        #self.lmu = lmu.LMU(input_size, hidden_size, memory_size, theta, learn_a, learn_b)
        #self.classifier = nn.Linear(hidden_size, output_size)
        self.lmu_fft = LMUFFT(input_size, hidden_size, memory_size, seq_len, theta)
        self.dropout = nn.Dropout(p = 0.5)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lmu_fft(x) # [batch_size, hidden_size]
        x = self.dropout(h_n)
        output = self.classifier(x)
        return output # [batch_size, output_size]

    def countParameters(model):
        """ Counts and prints the number of trainable and non-trainable parameters of a model """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")

    def training(model, loader, optimizer, criterion, DEVICE):
        # Single training epoch

        epoch_loss = 0
        y_pred = []
        y_true = []
        
        model.train()
        for batch, labels in tqdm(loader):
            # Batch shape default: [classes, workers?, ?, input]
            print("Batch size: ", batch.size())
            batch =  batch[:, 0, :, :]

            torch.cuda.empty_cache()

            batch = batch.to(DEVICE)
            labels = labels.long().to(DEVICE)

            
            optimizer.zero_grad()

            print("Batch size: ", batch.size())

            output = model(batch)
            output = output[:, -1]
            #=print("Output: ", output.size())
            #print("Labels: ", labels.size())
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            preds = output.argmax(dim = 1)
            y_pred += preds.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()

        # Loss
        avg_epoch_loss = epoch_loss / len(loader)

        # Accuracy
        epoch_acc = accuracy_score(y_true, y_pred)
        
        return avg_epoch_loss, epoch_acc

    def validate(model, loader, criterion, DEVICE):
    # Single validation epoch

        epoch_loss = 0
        y_pred = []
        y_true = []
        
        i = 0

        model.eval()
        with torch.no_grad():
            for batch, labels in tqdm(loader):
                x, v, h, m = batch
                batch = [x, h, m]

                torch.cuda.empty_cache()

                batch = batch.to(DEVICE)
                labels = labels.long().to(DEVICE)

                output = model(batch)
                output = output[:, -1]
                loss = criterion(output, labels)
                
                preds  = output.argmax(dim = 1)
                y_pred += preds.tolist()
                y_true += labels.tolist()
                epoch_loss += loss.item()

                print(i)
                i = i + 1
                
        # Loss
        avg_epoch_loss = epoch_loss / len(loader)

        # Accuracy
        epoch_acc = accuracy_score(y_true, y_pred)

        return avg_epoch_loss, epoch_acc

class langIdentifierReLU(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=176)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def countParameters(model):
        """ Counts and prints the number of trainable and non-trainable parameters of a model """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")
 
    #Forward pass
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

    def training(model, loader, optimizer, criterion, DEVICE):
        #Single training epoch
        loss = 0.0
        y_true = 0
        y_pred = 0
        
        for i, data in enumerate(loader):
            inputs = data[0].to(DEVICE)
            labels = torch.tensor(data[1]).long().to(DEVICE)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            loss += loss.item()

            _, prediction = torch.max(output, 1)
            y_true += (prediction == labels).sum().item()
            y_pred += prediction.shape[0]
        
        train_loss = loss / len(loader)
        train_acc = y_true / y_pred

        return train_loss, train_acc

    def validate(model, loader, criterion, DEVICE):
    # Single validation epoch
        epoch_loss = 0.0
        y_pred = 0
        y_true = 0

        with torch.no_grad():
            for data in loader:

                inputs = data[0].to(DEVICE)
                labels = data[1].to(DEVICE)

                inputs_m = inputs.mean()
                inputs_s = inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                output = model(inputs)

                _, prediction = torch.max(output, 1)

                y_true += (prediction == labels).sum().item()
                y_pred += prediction.shape[0]

                loss = criterion(output, labels)
                epoch_loss += loss.item()
                
        # Loss
        avg_epoch_loss = epoch_loss / len(loader)

        # Accuracy
        epoch_acc = y_true / y_pred

        return avg_epoch_loss, epoch_acc