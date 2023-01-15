#https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
#The base of the code here is taken from this article
#it has been modified to use the LMU

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.nn import init
import torchaudio
import lmu

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
    def __init__(self, filename, path, duration, sampleRate, channels, dataFrame):
        self.filename = filename
        self.path = path
        self.duration = duration
        self.sampleRate = sampleRate
        self.channels = channels
        self.dataFrame = dataFrame
        
    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, id):
        audioFile = self.path + self.dataFrame.loc[id, 'relative_path']
        classID = self.dataFrame.loc[id, 'classID']
        file = AudioUtil.open(audioFile)
        #Ensure consistent data
        resampled = AudioUtil.resample(file, self.sampleRate)
        rechanneled = AudioUtil.rechannel(resample, self.channels)
        padded = AudioUtil.pad_trunc(rechanneled, self.duration)
        shifted = AudioUtil.timeshift(padded, 0.4)
        spectro = AudioUtil.spectrogram(shifted, n_mels=64, n_fft=1024, hop_len=None)
        spectrogram = AudioUtil.spectroaugment(spectro, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return spectrogram, classID

class languageIdentifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
        super(Model, self).__init__()
        self.lmu = LMU(input_size, hidden_size, memory_size, theta, learn_a, learn_b)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lmu(x) # [batch_size, hidden_size]
        output = self.classifier(h_n)
        return output # [batch_size, output_size]

    def countParameters(model):
        """ Counts and prints the number of trainable and non-trainable parameters of a model """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")

    def train(model, loader, optimizer, criterion):
        # Single training epoch

        epoch_loss = 0
        y_pred = []
        y_true = []
        
        model.train()
        for batch, labels in tqdm(loader):

            torch.cuda.empty_cache()

            batch = batch.to(DEVICE)
            labels = labels.long().to(DEVICE)

            optimizer.zero_grad()

            output = model(batch)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            preds  = output.argmax(dim = 1)
            y_pred += preds.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()

        # Loss
        avg_epoch_loss = epoch_loss / len(loader)

        # Accuracy
        epoch_acc = accuracy_score(y_true, y_pred)

        return avg_epoch_loss, epoch_acc

    def validate(model, loader, criterion):
    # Single validation epoch

    epoch_loss = 0
    y_pred = []
    y_true = []
    
    model.eval()
    with torch.no_grad():
        for batch, labels in tqdm(loader):

            torch.cuda.empty_cache()

            batch = batch.to(DEVICE)
            labels = labels.long().to(DEVICE)

            output = model(batch)
            loss = criterion(output, labels)
            
            preds  = output.argmax(dim = 1)
            y_pred += preds.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()
            
    # Loss
    avg_epoch_loss = epoch_loss / len(loader)

    # Accuracy
    epoch_acc = accuracy_score(y_true, y_pred)

    return avg_epoch_loss, epoch_acc
    

