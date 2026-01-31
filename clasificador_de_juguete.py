import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sentences = [
    "the processor is fast", "gpu for deep learning", "new software update", 
    "programming in python", "web development framework", "cloud computing power",
    "the team won the match", "football players on the field", "amazing goal scored",
    "basketball tournament season", "tennis racket and ball", "athlete training hard"
] * 20 
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 20

vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(sentences).toarray()
y = np.array(labels)
input_dim = X.shape[1] 
print(f"Dimensiones de entrada reales: {input_dim}") 
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=8, shuffle=True)

class SimpleNLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleNLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNLP(input_size=input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print("Iniciando entrenamiento...")

for epoch in range(15):
    model.train()
    for texts, targets in loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Época {epoch+1} - Loss: {loss.item():.4f}")

def quick_test(phrase):
    model.eval()
    with torch.no_grad():
        vec = vectorizer.transform([phrase]).toarray()
        vec_t = torch.FloatTensor(vec)
        res = model(vec_t)
        clase = torch.argmax(res).item()
        return "Tecnología" if clase == 0 else "Deportes"

print("-" * 30)
print(f"Test: 'Python coding' -> {quick_test('Python coding')}")
print(f"Test: 'Football match' -> {quick_test('Football match')}")