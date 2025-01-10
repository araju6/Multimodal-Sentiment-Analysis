import speech_recognition as sr
import torch
import whisper
import warnings
warnings.filterwarnings("ignore")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = whisper.load_model("base")

class wavTextConverter:
    def __init__(self) -> None:
        self.file_paths = None

    def convert(self, file_paths):
        self.file_paths = file_paths
        batch = []
        r = sr.Recognizer()
        
        for file_path in self.file_paths:
            text = model.transcribe(file_path)["text"]
            # print(text)
            text_numbers = [ord(c) for c in text]
            batch.append(text_numbers)

        max_len = max(len(seq) for seq in batch)
        padded_batch = [seq + [0]*(25 - len(seq)) for seq in batch]
        
        return torch.tensor(padded_batch, dtype=torch.float32).to(device=device)
        
# a = wavTextConverter(["Dataset/YAF_happy/YAF_back_happy.wav", "Dataset/YAF_angry/YAF_bar_angry.wav"])
# Sxx = a.convert()
# print(Sxx.shape)