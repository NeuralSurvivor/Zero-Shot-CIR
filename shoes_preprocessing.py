import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
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


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name_openclip, device="cuda:0", pretrained=pretrained
)
tokenizer = open_clip.get_tokenizer(model_name_openclip)

with open(folder / "relative_pairs_test.npy", "rb") as f:
    pairs = list(np.load(f, allow_pickle=True))

catalog_imgs = []
test_lists = []
test_pairs = []
with torch.no_grad():
    for p in tqdm(pairs):
        ref = p["source"]
        tgt = p["target"]
        txt = p["mod"]
        ref = "/".join(ref.split("/")[-3:])
        tgt = "/".join(tgt.split("/")[-3:])
        catalog_imgs.extend([ref, tgt])
        test_pairs.append([ref, txt, tgt])
        ref = preprocess_val(Image.open(folder / ref)).unsqueeze(0).to("cuda:0")
        txt = tokenizer([txt]).to("cuda:0")
        img_ref_emb = model.encode_image(ref).tolist()[0]
        txt_emb = model.encode_text(txt).tolist()[0]
        test_lists.append([img_ref_emb, txt_emb])

catalog_imgs = list(np.unique(catalog_imgs))
catalog_embed = []

with torch.no_grad():
    for img in tqdm(catalog_imgs):
        ref = preprocess_val(Image.open(folder / img)).unsqueeze(0).to("cuda:0")
        img_emb = model.encode_image(ref).tolist()[0]
        catalog_embed.append(img_emb)
test_lists = np.array(test_lists)
data_path = Path(f"./data/shoes/{model_name}")
os.makedirs(data_path / "test", exist_ok=True)
os.makedirs(data_path / "catalog", exist_ok=True)

with open(data_path / "test/test.npy", "wb") as f:
    np.save(f, test_lists)
with open(data_path / "test/test_pair.json", "w") as f:
    json.dump(test_pairs, f, indent=2)
with open(data_path / "catalog/embed.npy", "wb") as f:
    np.save(f, catalog_embed)
with open(data_path / "catalog/img_idx.json", "w") as f:
    json.dump(catalog_imgs, f, indent=2)
