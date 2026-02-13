import torch
import torch.nn as nn
from gensim.utils import simple_preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
OUTPUT_DIM = 4 
N_LAYERS = 2
DROPOUT = 0.3

class LSTMClassifier(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1,:,:]
        return self.fc(last_hidden)

print("Cargando vocabulario y matriz...")
vocab_data = torch.load("vocab_data.pth", weights_only=False)
word2idx = vocab_data['word2idx']
weights_matrix = vocab_data['weights_matrix']

model = LSTMClassifier(weights_matrix, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)

model.load_state_dict(torch.load("lstm_agnews.pth", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("Modelo listo.")

def predict(text):
    tokens = simple_preprocess(text)
    idxs = [word2idx.get(t, 1) for t in tokens]
    if not idxs:
        return "Texto vacío o sin palabras conocidas"
    tensor = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(1).item()   
    clases = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    return clases[prediction]

sample_news = [
    # Class: Business
    "Fed Signals Rate Cuts Ahead (Bloomberg) Bloomberg - Federal Reserve officials hinted that inflation is cooling faster than expected, prompting markets to rally as traders bet on interest rate cuts by early next quarter.",

    # Class: Sci/Tech
    "New AI Chip Crushes Benchmarks (TechCrunch) TechCrunch - The latest silicon from Silicon Valley promises to double processing speeds for large language models while consuming 40 percent less energy, setting a new standard for data centers.",
    
    # Class: Sports
    "United Stuns City in Derby Thriller (BBC Sport) BBC - A last-minute goal secured a dramatic victory for Manchester United over their local rivals, shaking up the league table and leaving fans in absolute delirium.",
    
    # Class: World
    "Peace Talks Resume in Geneva (AP) Associated Press - Diplomats from three continents have gathered in Switzerland hoping to broker a ceasefire, though tensions remain high following the breakdown of last month's negotiations.",
    
    # Class: Sci/Tech
    "Mars Rover Sends High-Res Panoramas (NASA) NASA - The perseverance rover has beamed back the most detailed images of the Martian surface to date, revealing potential ancient riverbeds that excite astrobiologists.",
    
    # Class: Business
    "Oil Prices Surge Amid Supply Fears (Reuters) Reuters - Crude oil futures jumped 3 percent on Monday as geopolitical tensions threatened to disrupt key supply routes in the Middle East, worrying global importers."
]

for news in sample_news:
    resultado = predict(news)
    print(f"Noticia: {news}")
    print(f"Categoría: {resultado}")
    print("-" * 80)