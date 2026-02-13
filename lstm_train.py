import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from datasets import load_dataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

print("Cargando datos con Hugging Face...")
dataset = load_dataset("ag_news")
train_data = dataset['train']
test_data = dataset['test']

def tokenizer(text):
    return simple_preprocess(text)

print("Entrenando Word2Vec...")
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item['text'])

tokens_list = list(yield_tokens(train_data)) 

EMBEDDING_DIM = 100
w2v_model = Word2Vec(sentences=tokens_list, 
                     vector_size=EMBEDDING_DIM, 
                     window=5, 
                     min_count=1, 
                     workers=4)

vocab_gensim = w2v_model.wv.key_to_index
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

weights_matrix = np.zeros((len(vocab_gensim) + 2, EMBEDDING_DIM))
word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

for word, i in vocab_gensim.items():
    idx_new = i + 2 
    weights_matrix[idx_new] = w2v_model.wv[word]
    word2idx[word] = idx_new

weights_matrix = torch.tensor(weights_matrix, dtype=torch.float32)
print(f"Vocabulario creado. Tamaño: {len(word2idx)}")

text_pipeline = lambda x: [word2idx.get(token, word2idx[UNK_TOKEN]) for token in tokenizer(x)]
label_pipeline = lambda x: int(x) 

def collate_batch(batch):
    label_list, text_list = [], []
    for item in batch:
        _label = item['label']
        _text = item['text']  
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    return label_list.to(device), text_list.to(device)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)

class LSTMClassifier(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.shape 
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1,:,:]
        return self.fc(last_hidden)

HIDDEN_DIM = 256
OUTPUT_DIM = 4 
N_LAYERS = 2
DROPOUT = 0.3

model = LSTMClassifier(weights_matrix, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for labels, text in iterator:
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        correct = (predictions.argmax(1) == labels).float()
        acc = correct.sum() / len(correct)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()     
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for labels, text in iterator:
            predictions = model(text)
            loss = criterion(predictions, labels)
            correct = (predictions.argmax(1) == labels).float()
            acc = correct.sum() / len(correct)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 3
print(f"\nIniciando entrenamiento en {device} por {N_EPOCHS} épocas...")

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion) 
    print(f'Época: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

torch.save(model.state_dict(), "lstm_agnews.pth")
print("Modelo guardado correctamente.")

vocab_data = {
    'word2idx': word2idx,
    'weights_matrix': weights_matrix
}
torch.save(vocab_data, "vocab_data.pth")
print("Datos de vocabulario guardados en vocab_data.pth")