import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--folder", required=True)
parser.add_argument("--hf", action="store_true")
parser.add_argument("--pretrain", default=None)

args = parser.parse_args()

model_name = args.model
folder = Path(args.folder)
pretrained = args.pretrain
if args.hf:
    model_name_openclip = f"hf-hub:{model_name}"
else:
    model_name_openclip = model_name
    model_name = pretrained + "/" + model_name

image_folder = folder / "images"
json_folder = folder / "captions"
catalog_folder = folder / "image_splits"

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name_openclip, device="cuda:0", pretrained=pretrained
)
tokenizer = open_clip.get_tokenizer(model_name_openclip)
files = os.listdir(json_folder)
train_val_lists = []
test_lists = []
test_pairs = []

with torch.no_grad():
    for file in files:
        section = file.split(".")[-2]
        if section == "train":
            lists = train_val_lists
        elif section == "test":
            continue
        else:
            lists = test_lists

        with open(json_folder / file) as f:
            cap = json.load(f)

        for c in tqdm(cap):
            ref_img = c["candidate"] + ".jpg"
            txts = c["captions"]
            tgt_img = c["target"] + ".jpg"
            if section == "val":
                test_pairs.extend(
                    [
                        [c["candidate"], c["captions"][0], c["target"]],
                        [c["candidate"], c["captions"][1], c["target"]],
                    ]
                )
            images = [
                Image.open(image_folder / ref_img),
                Image.open(image_folder / tgt_img),
            ]
            img_ref = preprocess_val(images[0]).unsqueeze(0).to("cuda:0")
            img_tgt = preprocess_val(images[1]).unsqueeze(0).to("cuda:0")
            txts = tokenizer(txts).to("cuda:0")
            img_ref_emb = model.encode_image(img_ref).tolist()
            img_tgt_emb = model.encode_image(img_tgt).tolist()
            txt_embs = model.encode_text(txts).tolist()
            if section == "eval":
                lists.append([img_ref_emb[0], txt_embs[0]])
                lists.append([img_ref_emb[0], txt_embs[1]])
            else:
                lists.append([img_ref_emb[0], txt_embs[0], img_tgt_emb[0]])
                lists.append([img_ref_emb[0], txt_embs[1], img_tgt_emb[0]])

files = os.listdir(catalog_folder)
img_idx = []
catalog_embed = []
with torch.no_grad(), torch.amp.autocast("cuda"):
    for file in files:
        section = file.split(".")[-2]
        if section == "train":
            continue
        elif section == "test":
            continue

        with open(catalog_folder / file) as f:
            cap = json.load(f)

        for c in tqdm(cap):
            img_name = c + ".jpg"
            img_emb = (
                preprocess_val(Image.open(image_folder / img_name))
                .unsqueeze(0)
                .to("cuda:0")
            )
            img_emb = model.encode_image(img_emb).tolist()
            catalog_embed.append(img_emb[0])
            img_idx.append(c)

train_val_lists = np.array(train_val_lists)
test_lists = np.array(test_lists)
catalog_embed = np.array(catalog_embed)

train_lists, val_lists = train_test_split(
    train_val_lists, test_size=0.2, random_state=42
)
data_path = Path(f"./data/fashion/{model_name}")
os.makedirs(data_path / "training", exist_ok=True)
os.makedirs(data_path / "test", exist_ok=True)
os.makedirs(data_path / "catalog", exist_ok=True)

with open(data_path / "training/train.npy", "wb") as f:
    np.save(f, train_lists)
with open(data_path / "training/val.npy", "wb") as f:
    np.save(f, val_lists)
with open(data_path / "test/test.npy", "wb") as f:
    np.save(f, test_lists)
with open(data_path / "test/test_pair.json", "w") as f:
    json.dump(test_pairs, f, indent=2)
with open(data_path / "catalog/embed.npy", "wb") as f:
    np.save(f, catalog_embed)
with open(data_path / "catalog/img_idx.json", "w") as f:
    json.dump(img_idx, f, indent=2)
