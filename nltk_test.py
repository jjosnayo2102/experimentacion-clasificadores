import torch
import torch.nn as nn
import joblib
import numpy as np

class SentimentMLP(nn.Module):
    def __init__(self, input_dim):
        super(SentimentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

vectorizer = joblib.load("tfidf_vectorizer.pkl")
input_dim = len(vectorizer.vocabulary_)
model = SentimentMLP(input_dim)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()
texts = [
    "this movie was absolutely wonderful and inspiring",
    "the film was boring and a complete waste of time",
    "an average movie with some good moments",
    "terrible acting and bad story",
    "excellent plot and great performances"
]
X_np = vectorizer.transform(texts).toarray()
X_t = torch.FloatTensor(X_np)

with torch.no_grad():
    logits = model(X_t)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
labels = {0: "NEGATIVE", 1: "POSITIVE"}
for text, pred, prob in zip(texts, predictions, probabilities):
    print(f"\nTexto: {text}")
    print(f"Predicción: {labels[pred.item()]}")
    print(f"Probabilidades: Neg={prob[0]:.3f}, Pos={prob[1]:.3f}")
