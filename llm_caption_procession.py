import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--file", required=True)
parser.add_argument("--hf", action="store_true")
parser.add_argument("--pretrain", default=None)

args = parser.parse_args()

model_name = args.model
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

json_file = Path(args.file)
with open(json_file) as f:
    pairs = json.load(f)

txts = [p[0] for p in pairs]

txt_embs = []
for txt in tqdm(txts):
    txt = tokenizer([txt]).to("cuda:0")
    txt_emb = model.encode_text(txt).tolist()[0]
    txt_embs.append(txt_emb)

txt_embs = np.array(txt_embs)
os.makedirs(f"llm_captions_emb/{model_name}/", exist_ok=True)
np.save(f"llm_captions_emb/{model_name}/llm_captions.npy", txt_embs)
