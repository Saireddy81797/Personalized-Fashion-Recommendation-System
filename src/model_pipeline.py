# Orchestrates training both content and collab recommenders
import os
from pathlib import Path
import pickle
from src.data_loader import load_products, load_interactions
from src.content_based import ContentRecommender
from src.collab import CollabRecommender


MODELS_DIR = Path(__file__).parents[1] / 'models'
MODELS_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
products = load_products()
interactions = load_interactions()


# Content model
content = ContentRecommender(products)
content.build()
pickle.dump(content, open(MODELS_DIR / 'content_rec.pkl','wb'))


# Collaborative model
collab = CollabRecommender(interactions)
collab.fit()
pickle.dump(collab, open(MODELS_DIR / 'collab_rec.pkl','wb'))


print("Models trained and saved to models/")
