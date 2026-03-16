import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttentionManual(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q_in, K_in, V_in, mask=None):
        batch_size = Q_in.size(0)
        seq_len = Q_in.size(1)
        
        Q = self.W_q(Q_in).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K_in).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V_in).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)   
        return output, attn_weights 

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttentionManual(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask) 
        x = self.norm1(x + self.dropout(attn_output)) 
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output)) 
        return x

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        padding_mask = (x != 0)
        if mask is None:
            # [batch, 1, 1, seq_len]
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = mask
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        # [batch, seq_len, 1]
        input_mask_float = padding_mask.float().unsqueeze(-1)
        x_masked = x * input_mask_float
        sum_vectors = x_masked.sum(dim=1) # [batch, d_model]
        count_vectors = input_mask_float.sum(dim=1)
        mean_pooled = sum_vectors / count_vectors 
        output_logits = self.classifier(self.dropout(mean_pooled)) 
        return output_logits

D_MODEL = 64
VOCAB_SIZE = 10000 
MAX_SEQ_LEN = 200  
N_HEADS = 4
D_FF = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

tokenizer = lambda x: x.lower().split() 
vocab = {'<pad>': 0, '<unk>': 1}

def build_vocab(dataset):
    print("Construyendo vocabulario...")
    counter = Counter()
    for text, _ in dataset:
        counter.update(tokenizer(text))

    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        if word not in vocab:
            vocab[word] = len(vocab)
    print(f"Vocabulario construido: {len(vocab)} tokens.")

def text_pipeline(text):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(text)]

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if processed_text.size(0) > MAX_SEQ_LEN:
            processed_text = processed_text[:MAX_SEQ_LEN]
        text_list.append(processed_text)     
    label_list = torch.tensor(label_list, dtype=torch.long)
    # Pad sequence rellena con 0s para que todos tengan la misma longitud
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return text_list.to(device), label_list.to(device)

print("Cargando IMDB dataset desde Hugging Face...")
dataset = load_dataset("imdb")
# El dataset viene como un DatasetDict: {'train': ..., 'test': ..., 'unsupervised': ...}
hf_train = dataset['train']
hf_test = dataset['test']
print("Convirtiendo formato para el DataLoader...")
train_data = [(item['text'], item['label']) for item in hf_train]
test_data = [(item['text'], item['label']) for item in hf_test]
print(f"Datos cargados: {len(train_data)} entrenamiento, {len(test_data)} test")

build_vocab(train_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

model = SimpleTransformerClassifier(len(vocab), D_MODEL, N_HEADS, D_FF, NUM_LAYERS, MAX_SEQ_LEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\nIniciando entrenamiento...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0   
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

print("Entrenamiento finalizado. Iniciando evaluación en Test set...")

model.eval()
correct = 0
total = 0
sample_reviews = [
    ("this movie was absolutely excellent and i loved it", 1),
    ("terrible waste of time boring and bad acting", 0)
]

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()     
    print(f"--> Accuracy Final en Test Set: {100 * correct / total:.2f}%") 
    print("\n--- Pruebas Manuales ---")
    for text, label in sample_reviews:
        pipeline_tensor = torch.tensor(text_pipeline(text), dtype=torch.long).unsqueeze(0).to(device)
        output = model(pipeline_tensor)
        pred = torch.argmax(output, dim=1).item()
        sentiment = "Positivo" if pred == 1 else "Negativo"
        print(f"Review: '{text}' \nPredicción: {sentiment} (Real: {'Positivo' if label==1 else 'Negativo'})\n")