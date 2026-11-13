import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

SYSTEM_PROMPT = """You are the query-generation module of a composed image retrieval system.

INPUT: a reference image and a modification instruction describing how the TARGET image
differs from the reference (e.g. "same but red", "at night", "remove the people",
"two of them", "more formal").

TASK: Write a caption of the TARGET image, i.e. what the reference image shows with the
modification applied. The caption will be embedded with CLIP to retrieve the target.

RULES
1. First observe the reference: subject(s) and count, color, material, pattern, shape,
   clothing/parts, pose/action, spatial layout, background, style.
2. Apply the modification fully. Replaced attributes must be overwritten, not appended
   (blue dress + "red" -> a red dress, never "blue and red").
3. Keep every detail of the reference that the modification does not contradict.
4. Resolve relative or elliptical instructions ("more", "same but", "darker", "bigger")
   into concrete descriptions.
5. Convert removals/negations into positive descriptions. Never use "no", "without",
   "not", "instead of" (CLIP handles negation badly). "without sleeves" -> "sleeveless".
6. If the modification changes the scene or style wholesale ("at the beach", "make it a
   painting"), rewrite the caption in that new context but keep the subject.
7. Describe only what is visible or implied by the edit. Do not invent unrelated details.
8. The caption must be standalone and natural: never mention the reference image, the
   user, or the edit ("changed", "now", "instead").
9. At most 45 words. Put the subject and the changed attributes first.

OUTPUT: strictly one JSON object, no markdown, no extra text:
{"target_caption": "<caption>"}"""

model_id = "Qwen/Qwen3-VL-8B-Instruct"

parser = ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--out", default=".", help="where to save output json files")
parser.add_argument("--max_side", type=int, default=896)
args = parser.parse_args()

pair_file = Path(args.file)
dataset = Path(args.dataset)

processor = AutoProcessor.from_pretrained(model_id)
load_kwargs = dict(device_map="cuda:0")
load_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval()


def parse_output(raw):
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0))["target_caption"].strip()
    except Exception:
        return raw.strip().strip('"')


@torch.no_grad()
def generate_target_text(image_path, modification, max_new_tokens=120):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((args.max_side, args.max_side))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f'Modification: "{modification}"\nReturn the JSON.',
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(
        qwen.device
    )
    out = qwen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    return parse_output(raw.strip())


out_file = "llm_caption_pair.json"

with open(pair_file) as f:
    test_pairs = json.load(f)

output = []
for img_path, txt, ref_id in tqdm(test_pairs):
    caption = generate_target_text(dataset / img_path, txt)
    output.append([caption, ref_id])

with open(out_file, "w") as f:
    json.dump(output, f)
print(f"saved {out_file}")
