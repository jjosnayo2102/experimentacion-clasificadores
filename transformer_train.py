import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("ag_news")
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item['text'])

vocab = build_vocab_from_iterator(yield_tokens(dataset['train']), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

def collate_batch(batch):
    label_list, text_list = [], []
    for item in batch:
        label_list.append(label_pipeline(item['label']))
        processed_text = torch.tensor(text_pipeline(item['text']), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return label_list.to(device), text_list.to(device)

train_dataloader = DataLoader(dataset['train'], batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(dataset['test'], batch_size=64, shuffle=False, collate_fn=collate_batch)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=vocab["<pad>"])
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model)) 
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)  
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = dropout

    def forward(self, text):
        padding_mask = (text == vocab["<pad>"])
        x = self.embedding(text) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1) 
        return self.fc(x)

VOCAB_SIZE = len(vocab)
model = TransformerClassifier(VOCAB_SIZE, d_model=128, nhead=8, num_layers=3, num_classes=4).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001) 
criterion = nn.CrossEntropyLoss()
EPOCHS = 10

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for labels, text in iterator:
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(1) == labels).float().mean()
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
            acc = (predictions.argmax(1) == labels).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

print(f"\nModelo inicializado con {sum(p.numel() for p in model.parameters())} parámetros.")
print("Iniciando entrenamiento...")

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

torch.save(model.state_dict(), "transformer_agnews.pth")
torch.save(vocab, "vocab_transformer.pth")