#%%
# Required Imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# Define stop words
stop_words = set(stopwords.words('english'))
# Read the CSV file
column_names = ['Sentiment', 'Text']
data = pd.read_csv("all-data.csv", header=None, names=column_names, encoding='ISO-8859-1')
# Preprocessing Function
def preprocess(text):
    text = text.lower()  # Lowercasing
    words = word_tokenize(text)  # Tokenize into words
    words = [w for w in words if w not in stop_words and w not in string.punctuation]  # Remove stopwords and punctuation
    return words
def word_frequencies(texts):
    word_freq = Counter()
    for text in texts:
        word_freq.update(text)
    total_words = sum(word_freq.values())
    word_freq = {word: count / total_words for word, count in word_freq.items()}
    return word_freq
def sif_embedding(model, word_freq, sentence, alpha=1e-3):
    sentence = preprocess(sentence)
    embedding = sum([model.wv[word] * alpha / (alpha + word_freq.get(word, 0)) for word in sentence if word in model.wv])
    return embedding / len(sentence) if len(sentence) > 0 else np.zeros(model.vector_size)
def pca_reduction(sentence_embeddings):
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=100)
    pca.fit(np.array(sentence_embeddings))
    sentence_embeddings = pca.transform(sentence_embeddings)
    return sentence_embeddings
def cos_similarity(sentence_embeddings):
    # Calculate the embedding for the entire document (for simplicity, we take the mean of sentence embeddings)
    document_embedding = np.mean(sentence_embeddings, axis=0)
    # Calculate cosine similarity
    similarity_scores = [cosine_similarity([emb], [document_embedding])[0][0] for emb in sentence_embeddings]
    # Using Nearest Neighbors to smooth the scores
    n_neighbors = 5  # Example value
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(sentence_embeddings)
    for i, emb in enumerate(sentence_embeddings):
        neighbors = knn.kneighbors([emb], return_distance=False)
        similarity_scores[i] = np.mean([similarity_scores[n] for n in neighbors[0]])
    return similarity_scores
#%%
texts = [preprocess(article) for article in data['Text']]
model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=4)
word_freq = word_frequencies(texts)
sentence_embeddings = np.array([sif_embedding(model, word_freq, article) for article in data['Text']])
sentence_embeddings = pca_reduction(sentence_embeddings)
similarity_scores = cos_similarity(sentence_embeddings)
#%%
# Extract top-k sentences
k = 5  # Number of sentences in summary
top_k_indices = np.argsort(similarity_scores)[-k:]
summary_sentences = [data['Text'].iloc[i] for i in sorted(top_k_indices)]
summary = ' '.join(summary_sentences)