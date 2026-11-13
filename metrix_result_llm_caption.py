import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

def metrics(ranks: np.ndarray, ks=(1, 5, 10, 50)) -> dict:
    ranks = ranks.astype(np.float64)
    gain = 1.0 / np.log2(ranks + 1)
    out = {}
    for k in ks:
        hit = ranks <= k
        recall = hit.mean()
        precision = recall / k
        f1 = 2 * precision * recall / (precision + recall) if recall > 0 else 0.0
        out[f"R@{k}"] = 100 * recall
        out[f"P@{k}"] = 100 * precision
        out[f"F1@{k}"] = 100 * f1
        out[f"Hits@{k}"] = hit.sum()
        out[f"DCG@{k}"] = np.where(hit, gain, 0.0).mean()
        out[f"NDCG@{k}"] = 100 * np.where(hit, gain, 0.0).mean()
    out["R_mean"] = np.mean([out[f"R@{k}"] for k in ks])
    out["MRR"] = 100 * (1.0 / ranks).mean()
    out["mAP"] = out["MRR"]
    out["R-Precision"] = out["R@1"]
    out["mean_rank"] = ranks.mean()
    out["median_rank"] = np.median(ranks)
    out = {name: float(v) for name, v in out.items()}
    return out

shoe_dir = Path("data/shoes")
llm_caption_emb_dir = Path("llm_captions_emb")
llm_caption_pair = Path("llm_caption_pair.json")
with open(llm_caption_pair) as f:
    pairs = json.load(f)

tgt = [p[1] for p in pairs]

result = []
orgs = os.listdir(llm_caption_emb_dir)
for org in orgs:
    models = os.listdir(llm_caption_emb_dir/org)
    for model in models:
        llm_emb = np.load(llm_caption_emb_dir/f"{org}/{model}/llm_captions.npy")
        catalog_emb = np.load(shoe_dir/f"{org}/{model}/catalog/embed.npy")
        with open("data/shoes/Marqo/marqo-fashionCLIP/catalog/img_idx.json")as f:
            catalog_idx = json.load(f)
        tgt_idx = [catalog_idx.index(t) for t in tgt]
        llm_emb /= np.linalg.norm(llm_emb, axis=1, keepdims=True)
        catalog_emb /= np.linalg.norm(catalog_emb, axis=1, keepdims=True)
        scores = llm_emb @ catalog_emb.T
        tgt_scores = scores[np.arange(scores.shape[0]), tgt_idx]
        ranks = 1 + np.count_nonzero(scores > tgt_scores[:, None], axis=1)
        res = metrics(ranks)
        res["org"] = org
        res["model"] = model
        result.append(res)
        
df = pd.DataFrame(result)
df.to_csv("llm_result.csv")