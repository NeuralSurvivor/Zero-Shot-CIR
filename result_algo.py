import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from algo import get_result

parser = ArgumentParser()
parser.add_argument("--test", required=True)
parser.add_argument("--ref", required=True)
args = parser.parse_args()

test_dataset_folder = Path(args.test)
ref_dataset_folder = Path(args.ref)

catalog_folder = test_dataset_folder / "catalog"
test_folder = test_dataset_folder / "test"
test_emb = np.load(test_folder / "test.npy")
with open(test_folder / "test_pair.json") as f:
    test_pair = json.load(f)

catalog_emb = np.load(catalog_folder / "embed.npy")
with open(catalog_folder / "img_idx.json") as f:
    catalog_idx = json.load(f)

ref_catalog_folder = ref_dataset_folder / "catalog"
ref_test_folder = ref_dataset_folder / "test"
ref_emb = np.load(ref_test_folder / "test.npy")
with open(ref_test_folder / "test_pair.json") as f:
    ref_pair = json.load(f)

ref_catalog_emb = np.load(ref_catalog_folder / "embed.npy")
with open(ref_catalog_folder / "img_idx.json") as f:
    ref_catalog_idx = json.load(f)

img_emb = test_emb[:, 0]
txt_emb = test_emb[:, 1]

tgt_idx = [catalog_idx.index(l[2]) for l in test_pair]

ref_img_emb = ref_emb[:, 0]
ref_txt_emb = ref_emb[:, 1]

ref_tgt_idx = [ref_catalog_idx.index(l[2]) for l in ref_pair]


alphas = [0.1, 0.25, 0.5, 0.75, 0.9, "auto"]
normalizes = [False, True]
both_list = [
    "weighted_sum",
    "fusion_min_max",
    "fusion_std",
    "fusion_z",
    "orth",
    "center",
    "center-cat",
    "orth-reverse",
]
alpha_only_list = ["slerp"]
normalize_only_list = ["product", "min"]
NOTA_list = []

folder = "/".join(str(test_dataset_folder).split("/")[-2:])
os.makedirs(f"results/{folder}", exist_ok=True)
result_folder = Path(f"results/{folder}/")

for algo in both_list:
    for alpha in alphas:
        for normalize in normalizes:
            if os.path.exists(result_folder / f"{algo}_{alpha}_{normalize}.json"):
                continue
            result = get_result(
                algo,
                img_emb,
                txt_emb,
                catalog_emb,
                alpha,
                normalize,
                tgt_idx,
                ref_img_emb,
                ref_txt_emb,
                ref_catalog_emb,
                ref_tgt_idx,
            )
            with open(result_folder / f"{algo}_{alpha}_{normalize}.json", "w") as f:
                json.dump(result, f)


for algo in alpha_only_list:
    normalize = False
    for alpha in alphas:
        if os.path.exists(result_folder / f"{algo}_{alpha}_{normalize}.json"):
            continue
        result = get_result(
            algo,
            img_emb,
            txt_emb,
            catalog_emb,
            alpha,
            normalize,
            tgt_idx,
            ref_img_emb,
            ref_txt_emb,
            ref_catalog_emb,
            ref_tgt_idx,
        )
        with open(result_folder / f"{algo}_{alpha}_{normalize}.json", "w") as f:
            json.dump(result, f)


for algo in normalize_only_list:
    alpha = 0.0
    for normalize in normalizes:
        if os.path.exists(result_folder / f"{algo}_{alpha}_{normalize}.json"):
            continue
        result = get_result(
            algo,
            img_emb,
            txt_emb,
            catalog_emb,
            alpha,
            normalize,
            tgt_idx,
            ref_img_emb,
            ref_txt_emb,
            ref_catalog_emb,
            ref_tgt_idx,
        )
        with open(result_folder / f"{algo}_{alpha}_{normalize}.json", "w") as f:
            json.dump(result, f)


for algo in NOTA_list:
    alpha = 0.0
    normalize = False
    if os.path.exists(result_folder / f"{algo}_{alpha}_{normalize}.json"):
        continue
    result = get_result(
        algo,
        img_emb,
        txt_emb,
        catalog_emb,
        alpha,
        normalize,
        tgt_idx,
        ref_img_emb,
        ref_txt_emb,
        ref_catalog_emb,
        ref_tgt_idx,
    )
    with open(result_folder / f"{algo}_{alpha}_{normalize}.json", "w") as f:
        json.dump(result, f)
