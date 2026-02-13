import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
from nltk.corpus import movie_reviews
import re
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sentences = []
labels = []

for fileid in movie_reviews.fileids():
    words = movie_reviews.raw(fileid)
    sentences.append(words)
    labels.append(1 if movie_reviews.categories(fileid)[0] == 'pos' else 0)

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

tokenized_sentences = [simple_tokenize(s) for s in sentences]

w2v_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1, 
    negative=10   
)

def sentence_vector(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

X = np.array([sentence_vector(tokens) for tokens in tokenized_sentences])
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

def predict_sentiment(text):
    tokens = simple_tokenize(text.lower())
    vec = sentence_vector(tokens).reshape(1, -1)
    pred = clf.predict(vec)[0]
    return "Positivo" if pred == 1 else "Negativo"

print(predict_sentiment("the movie was amazing and emotional"))
print(predict_sentiment("this film was boring and terrible"))

