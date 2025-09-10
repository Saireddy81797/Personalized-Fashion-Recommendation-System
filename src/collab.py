import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle


class CollabRecommender:
def __init__(self, interactions_df, n_components=64):
self.interactions = interactions_df
self.n_components = n_components
self.user_map = {}
self.item_map = {}
self.user_factors = None
self.item_factors = None


def build_matrix(self):
users = self.interactions['user_id'].unique()
items = self.interactions['product_id'].unique()
self.user_map = {u:i for i,u in enumerate(users)}
self.item_map = {p:i for i,p in enumerate(items)}
rows = self.interactions['user_id'].map(self.user_map)
cols = self.interactions['product_id'].map(self.item_map)
vals = self.interactions['event_weight']
mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
return mat


def fit(self):
mat = self.build_matrix()
svd = TruncatedSVD(n_components=self.n_components, random_state=42)
self.user_factors = svd.fit_transform(mat)
self.item_factors = svd.components_.T
self.user_factors = normalize(self.user_factors)
self.item_factors = normalize(self.item_factors)


def recommend_for_user(self, user_id, top_k=10):
if user_id not in self.user_map:
return []
uidx = self.user_map[user_id]
scores = self.user_factors[uidx].dot(self.item_factors.T)
top = np.argsort(-scores)[:top_k]
inv_item_map = {v:k for k,v in self.item_map.items()}
return [(inv_item_map[i], float(scores[i])) for i in top]


def save(self, path):
with open(path,'wb') as f:
pickle.dump(self, f)


@staticmethod
def load(path):
with open(path,'rb') as f:
return pickle.load(f)
