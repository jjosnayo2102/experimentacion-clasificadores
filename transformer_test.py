import torch
import torch.nn as nn
import math
from torchtext.data.utils import get_tokenizer 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))  
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, text):
        padding_mask = (text == vocab["<pad>"])  
        x = self.embedding(text) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1) 
        return self.fc(x)

print("Cargando vocabulario y modelo...")

vocab = torch.load("vocab_transformer.pth", weights_only=False) 
tokenizer = get_tokenizer("basic_english")

VOCAB_SIZE = len(vocab)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
NUM_CLASSES = 4
model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES)
model.load_state_dict(torch.load("transformer_agnews.pth", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

print("Modelo Transformer listo.")

def predict(text):
    tokens = tokenizer(text)
    idxs = [vocab[token] for token in tokens]
    if not idxs:
        return "Texto vacío o sin palabras conocidas"
    tensor = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(1).item() 
    clases = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    return clases[prediction]

sample_news = [
    "Fed Signals Rate Cuts Ahead (Bloomberg) - Inflation is cooling faster than expected...",
    "New AI Chip Crushes Benchmarks (TechCrunch) - The latest silicon from Silicon Valley...",
    "United Stuns City in Derby Thriller (BBC Sport) - A last-minute goal secured victory...",
    "Mars Rover Sends High-Res Panoramas (NASA) - The perseverance rover has beamed back images..."
]

for news in sample_news:
    resultado = predict(news)
    print(f"Categoría: {resultado} | Noticia: {news[:70]}...")