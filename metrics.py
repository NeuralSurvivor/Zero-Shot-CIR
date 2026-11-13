import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def metrics(path: str, ks=(1, 5, 10, 50)) -> dict:
    with open(path) as f:
        res = json.load(f)
    ranks = np.asarray(res["rank_of_tgt"], dtype=np.float64)
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
    out["alpha_used"] = res["alpha"]
    out["algo"] = res["algo"]
    out["normalize"] = res["normalize"]
    return out


parser = ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

folder = Path(args.folder)
result = []

orgs = os.listdir(folder)
for org in orgs:
    models = os.listdir(folder / org)
    for model in models:
        files = os.listdir(folder / f"{org}/{model}")
        for file in tqdm(files):
            try:
                row = {"org": org, "model": model}
                row.update(metrics(folder / f"{org}/{model}/{file}"))
                result.append(row)
            except:
                print(folder / f"{org}/{model}/{file}")
df = pd.DataFrame(result)
df.to_csv("result.csv")
