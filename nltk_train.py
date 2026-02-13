import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import movie_reviews
import joblib
from sklearn.metrics import classification_report

nltk.download('movie_reviews')
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid) 
        text = " ".join(words)
        documents.append((text, category))

X_raw = [doc[0] for doc in documents]
y_raw = [1 if doc[1] == 'pos' else 0 for doc in documents]

#vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
vectorizer = joblib.load("tfidf_vectorizer.pkl")
#X_np = vectorizer.fit_transform(X_raw).toarray()
X_np = vectorizer.transform(X_raw).toarray()
y_np = torch.tensor(y_raw, dtype=torch.int64)

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

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

input_dim = X_train_t.shape[1]
model = SentimentMLP(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Iniciando entrenamiento...")
for epoch in range(15):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "sentiment_model.pth")
#joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    predictions = predictions.numpy()
    print(classification_report(y_test, predictions, target_names=['NEGATIVE', 'POSITIVE']))
    print(f"Precisión en el set de prueba: {accuracy:.2%}")