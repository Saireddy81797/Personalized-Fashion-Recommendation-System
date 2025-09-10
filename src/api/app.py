from flask import Flask, jsonify
import pickle
from pathlib import Path


app = Flask(__name__)


MODELS_DIR = Path(__file__).parents[1] / 'models'
CONTENT_PATH = MODELS_DIR / 'content_rec.pkl'
COLLAB_PATH = MODELS_DIR / 'collab_rec.pkl'


content = None
collab = None


@app.before_first_request
def load_models():
global content, collab
if CONTENT_PATH.exists():
content = pickle.load(open(CONTENT_PATH,'rb'))
if COLLAB_PATH.exists():
collab = pickle.load(open(COLLAB_PATH,'rb'))


@app.route('/health')
def health():
return jsonify({'status':'ok'})


@app.route('/recommend/content/<int:product_id>')
def rec_by_product(product_id):
if not content:
return jsonify({'error':'content model not loaded'}), 500
res = content.recommend_by_product(product_id, top_k=10)
return res.to_json(orient='records')


@app.route('/recommend/user/<user_id>')
def rec_for_user(user_id):
if collab and user_id in collab.user_map:
recs = collab.recommend_for_user(user_id, top_k=10)
return jsonify([{'product_id':pid,'score':float(score)} for pid,score in recs])
return jsonify({'error':'no recs found for user'}), 404


if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000, debug=True)
