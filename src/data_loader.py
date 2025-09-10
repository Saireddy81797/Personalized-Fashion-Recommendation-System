import pandas as pd
from pathlib import Path


DATA_RAW = Path(__file__).parents[1] / 'data' / 'raw'


def load_products(path=None):
path = Path(path) if path else DATA_RAW / 'products.csv'
return pd.read_csv(path)


def load_interactions(path=None):
path = Path(path) if path else DATA_RAW / 'interactions.csv'
df = pd.read_csv(path)
weight_map = {'view':1, 'click':1, 'add_to_cart':3, 'purchase':5}
df['event_weight'] = df['event_type'].map(weight_map).fillna(1)
return df


if __name__ == '__main__':
print(load_products().head())
print(load_interactions().head())
