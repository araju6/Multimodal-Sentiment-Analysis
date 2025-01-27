import speech_recognition as sr
import torch
import whisper
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

whisper_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
senti_bert = AutoModel.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis").to(device)

class wavTextConverter:
    def __init__(self):
        self.file_paths = None

    def convert(self, file_paths):
        self.file_paths = file_paths
        batch = []

        for file_path in self.file_paths:
            text = whisper_model.transcribe(file_path)["text"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
            with torch.no_grad():
                outputs = senti_bert(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            batch.append(embeddings)

        batch_tensor = torch.cat(batch, dim=0)

        return batch_tensor
        
# a = wavTextConverter(["Dataset/YAF_happy/YAF_back_happy.wav", "Dataset/YAF_angry/YAF_bar_angry.wav"])
# Sxx = a.convert()
# print(Sxx.shape)