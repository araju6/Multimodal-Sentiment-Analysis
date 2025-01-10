import wave
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'


class wavImgConverter:

    def __init__(self) -> None:
        self.file_paths = None

    def convert(self, file_paths):
        self.file_paths = file_paths
        batch = []
        for file_path in self.file_paths:
            with wave.open(file_path, 'rb') as wav_file:
                num_channels = wav_file.getnchannels()
            #print(num_channels)
            if num_channels > 1:
                data = data[:, 0]
            rate, data = scipy.io.wavfile.read(file_path)
            f, t, S = scipy.signal.spectrogram(data, rate)

            S = cv2.resize(S, (200, 128))
            S = (S - S.min()) / (S.max() - S.min())
            # S = np.stack([S] * 3)
            batch.append(S)
        batch_spectrograms = np.stack(batch, axis=0)
        return torch.FloatTensor(batch_spectrograms).to(device).unsqueeze(1)
    
# a = wavImgConverter(["Dataset/YAF_angry/YAF_bar_angry.wav", "Dataset/YAF_angry/YAF_hate_angry.wav"])
# Sxx = a.convert()
# print(Sxx.shape)