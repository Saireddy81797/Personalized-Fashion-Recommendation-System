import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


class ContentRecommender:
def __init__(self, products_df, text_cols=['name','brand','category','description']):
self.products = products_df.copy().reset_index(drop=True)
self.text_cols = text_cols
self.tfidf = None
self.tfidf_matrix = None


def build(self):
self.products['meta'] = self.products[self.text_cols].fillna('').agg(' '.join, axis=1)
self.tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
self.tfidf_matrix = self.tfidf.fit_transform(self.products['meta'])


def recommend_by_product(self, product_id, top_k=10):
idxs = self.products.index[self.products['product_id'] == product_id].tolist()
if not idxs:
return pd.DataFrame(columns=['product_id','name','category','score'])
idx = int(idxs[0])
sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
best = np.argsort(-sims)[1: top_k+1]
return self.products.iloc[best][['product_id','name','category']].assign(score=sims[best]).reset_index(drop=True)


def recommend_for_user_profile(self, user_profile_text, top_k=10):
vec = self.tfidf.transform([user_profile_text])
sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
best = np.argsort(-sims)[:top_k]
return self.products.iloc[best][['product_id','name','category']].assign(score=sims[best]).reset_index(drop=True)


def save(self, path):
with open(path, 'wb') as f:
pickle.dump({'tfidf':self.tfidf,'products':self.products}, f)


@staticmethod
def load(path):
with open(path,'rb') as f:
state = pickle.load(f)
rec = ContentRecommender(state['products'])
rec.tfidf = state['tfidf']
rec.tfidf_matrix = rec.tfidf.transform(rec.products['meta'])
return rec
