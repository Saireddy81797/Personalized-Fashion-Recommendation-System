import numpy as np


def precision_at_k(true_items, pred_items, k=10):
pred_k = pred_items[:k]
return len(set(true_items) & set(pred_k)) / k


def recall_at_k(true_items, pred_items, k=10):
pred_k = pred_items[:k]
return len(set(true_items) & set(pred_k)) / len(true_items) if len(true_items)>0 else 0.0


def mean_metric_over_users(metric_fn, user_truth, user_preds, k=10):
vals = []
for u in user_truth:
vals.append(metric_fn(user_truth[u], user_preds.get(u, []), k=k))
return np.mean(vals)
