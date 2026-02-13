import torch
import torch.nn as nn
import torch.optim as optim
import random

text = "el robot mueve el brazo"
tokens = text.split()

vocab = list(set(tokens))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
V = len(vocab)

window_size = 2
pairs = []

for i, word in enumerate(tokens):
    for j in range(-window_size, window_size + 1):
        if j != 0 and 0 <= i + j < len(tokens):
            context = tokens[i + j]
            pairs.append((word, context))

print(pairs)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, negative):
        v = self.in_embed(center) 
        u_pos = self.out_embed(context)  
        u_neg = self.out_embed(negative) 
        pos_score = torch.sum(v * u_pos, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(1)
        return - (pos_loss + neg_loss).mean()

embedding_dim = 10
model = SkipGram(V, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def get_negative_samples(k):
    return torch.tensor([random.randint(0, V-1) for _ in range(k)])

for epoch in range(300):
    total_loss = 0
    for center_word, context_word in pairs:
        center = torch.tensor([word_to_ix[center_word]])
        context = torch.tensor([word_to_ix[context_word]])
        negative = get_negative_samples(4).unsqueeze(0)
        loss = model(center, context, negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

embeddings = model.in_embed.weight.data
for word in vocab:
    print(word, embeddings[word_to_ix[word]])