# Zero-Shot Composed Image Retrieval on the Shoes Dataset: A Pre-Trained Embedding Merging Study

**Authors**

- Arvind Kumar Sharma (26AI60R08)
- Asim Aakash (26AI60R13)

---

## Overview

Composed Image Retrieval (CIR) retrieves a target image given a **reference image** and a **text modification** (e.g. *"same shoe but with a higher heel"*). This project studies **zero-shot CIR**: all vision and text encoders stay frozen, and we only study how their embeddings can be combined to form a query.

We evaluate on the **Shoes** dataset, whose captions describe fine-grained attribute changes. To our knowledge there is no published zero-shot CIR baseline on it, so this repo aims to provide one.

The questions we want to answer:

1. How far can frozen, pre-trained embeddings go on Shoes with no domain-specific training?
2. Which training-free merging strategy works best, and how sensitive is it to the image/text weight `alpha`?
3. Is zero-shot good enough for practical use, or is end-to-end fine-tuning needed?

## What is in this repo

Training-free baselines (implemented and runnable):

| Approach | Description |
|---|---|
| **Embedding-merging algorithms** | 11 ways of combining the reference-image embedding and the text embedding (see [Algorithms](#merging-algorithms)). |
| **`alpha` selection** | Fixed values (0.1, 0.25, 0.5, 0.75, 0.9) or `auto`: tuned with Optuna on a *separate* dataset (FashionIQ val), then applied unchanged to Shoes. |
| **LLM-caption baseline** | A vision-language model (Qwen3-VL-8B-Instruct, 4-bit) rewrites the modification into a full caption of the target image; the caption is then embedded and used for text-to-image retrieval. |

## Pipeline

```
 Shoes / FashionIQ images + captions
            │
            ▼
 shoes_preprocessing.py / fashionIQ_preprocessing.py     (frozen encoder → .npy embeddings)
            │
            ├──────────────► result_algo.py ──► results/<org>/<model>/*.json ──► metrics.py ──► result.csv
            │                  (merging algorithms,
            │                   alpha grid + Optuna)
            │
            └─ test_pair.json ─► llm.py ─► llm_caption_pair.json
                                    │
                                    ▼
                         llm_caption_procession.py ─► llm_captions_emb/<org>/<model>/llm_captions.npy
                                    │
                                    ▼
                         metrix_result_llm_caption.py ─► llm_result.csv
```

## Repository structure

| File | Purpose |
|---|---|
| `shoes_preprocessing.py` | Encodes Shoes test pairs (reference image + modification text) and the gallery/catalog with a frozen CLIP-style model. |
| `fashionIQ_preprocessing.py` | Same for FashionIQ. Its val split is used as the **reference set for tuning `alpha`**. |
| `algo.py` | All merging algorithms, the Optuna `alpha` search, rank computation and result serialisation. |
| `result_algo.py` | Runs the full algorithm × alpha × normalize grid and writes one JSON per configuration. |
| `metrics.py` | Aggregates all result JSONs into `result.csv`. |
| `llm.py` | Generates target-image captions with Qwen3-VL-8B-Instruct (4-bit NF4). |
| `llm_caption_procession.py` | Embeds the generated captions with a chosen text encoder. |
| `metrix_result_llm_caption.py` | Evaluates the LLM-caption baseline and writes `llm_result.csv`. |

## Setup

**Requirements**

- Python 3.10+ (the code uses `float | str` type syntax)
- A CUDA GPU (the scripts hard-code `cuda:0`)

```bash
pip install numpy pandas scikit-learn optuna tqdm pillow \
            torch open_clip_torch \
            transformers bitsandbytes accelerate   # last three only for llm.py
```

**Data**

Both datasets are available on Hugging Face: [NeuralSurvivor/FashionIQAndShoe](https://huggingface.co/datasets/NeuralSurvivor/FashionIQAndShoe) (about 828 MB).

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NeuralSurvivor/FashionIQAndShoe",
    repo_type="dataset",
    local_dir="./datasets",
)
```

Point the `--folder` argument of the preprocessing scripts at the downloaded Shoes and FashionIQ folders.

- **Shoes**: needs `relative_pairs_test.npy` and the images, resolved relative to the dataset folder you pass in. Each pair is a dict with `source`, `target` and `mod` keys.
- **FashionIQ**: expects `images/`, `captions/` and `image_splits/` folders in the standard layout.

**Models**

The encoders evaluated so far (from the OpenCLIP / Hugging Face hubs):

| Model | Source | Flags |
|---|---|---|
| `ViT-B-32` | OpenAI, via OpenCLIP | `--pretrain openai` |
| `ViT-L-14` | OpenAI, via OpenCLIP | `--pretrain openai` |
| `ViT-B-32-SigLIP2-256` | WebLI, via OpenCLIP | `--pretrain webli` |
| `EVA02-B-16` | `merged2b_s8b_b131k`, via OpenCLIP | `--pretrain merged2b_s8b_b131k` |
| `Marqo/marqo-fashionCLIP` | Hugging Face | `--hf` |
| `Marqo/marqo-fashionSigLIP` | Hugging Face | `--hf` |

## Usage

### 1. Extract embeddings

```bash
# Shoes (test set + catalog)
python shoes_preprocessing.py --model ViT-B-32 --pretrain openai --folder /path/to/shoes
python shoes_preprocessing.py --model Marqo/marqo-fashionCLIP --hf --folder /path/to/shoes

# FashionIQ (val set for alpha tuning + catalog)
python fashionIQ_preprocessing.py --model ViT-B-32 --pretrain openai --folder /path/to/fashionIQ
python fashionIQ_preprocessing.py --model Marqo/marqo-fashionCLIP --hf --folder /path/to/fashionIQ
```

Outputs are written under `data/<dataset>/<org>/<model>/` (for OpenCLIP models `<org>` is the `--pretrain` name, e.g. `data/shoes/openai/ViT-B-32/`):

```
test/test.npy          # [reference-image emb, text emb] per query
test/test_pair.json    # [reference image, modification text, target image]
catalog/embed.npy      # gallery embeddings
catalog/img_idx.json   # gallery image ids (row order of embed.npy)
training/{train,val}.npy   # FashionIQ only
```

### 2. Run the merging algorithms

```bash
python result_algo.py \
    --test data/shoes/openai/ViT-B-32 \
    --ref  data/fashion/openai/ViT-B-32
```

`--test` is the evaluation set (Shoes). `--ref` is the set used only to tune `alpha` when `alpha="auto"` (FashionIQ), so **no Shoes labels are used for tuning**. Results go to `results/<org>/<model>/<algo>_<alpha>_<normalize>.json`; existing files are skipped, so the run can be resumed.

### 3. Compute metrics

```bash
python metrics.py --folder results     # → result.csv
```

### 4. LLM-caption baseline (optional)

```bash
# a) Generate target captions (needs the Shoes images and test_pair.json)
python llm.py --file data/shoes/openai/ViT-B-32/test/test_pair.json --dataset /path/to/shoes

# b) Embed captions with each text encoder
python llm_caption_procession.py --model ViT-B-32 --pretrain openai --file llm_caption_pair.json

# c) Evaluate against the Shoes catalogs
python metrix_result_llm_caption.py     # → llm_result.csv
```

## Merging algorithms

Let `i` be the reference-image embedding, `t` the text embedding and `C` the catalog. With `normalize=True`, embeddings are L2-normalised first.

| Name | Idea | What `alpha` does |
|---|---|---|
| `weighted_sum` | `alpha·i + (1-alpha)·t`, cosine against `C` | `alpha` = weight on **image** |
| `slerp` | Spherical interpolation between `i` and `t` (always normalised) | `alpha` moves from image (0) toward **text** (1) |
| `orth` | `i + alpha·t⊥`, where `t⊥` is `t` with its component along `i` removed | `alpha` = strength of the **text** residual |
| `orth-reverse` | `t + alpha·i⊥` (the mirror of `orth`) | `alpha` = strength of the **image** residual |
| `center` | Mean-centre query-side `i` and `t` (over the query set), then `(1-alpha)·i + alpha·t` | `alpha` = weight on **text** |
| `center-cat` | Same, but centre everything with the **catalog** mean | `alpha` = weight on **text** |
| `fusion_min_max` | Min-max normalise the two score vectors per query, then `alpha·s_i + (1-alpha)·s_t` | `alpha` = weight on **image** score |
| `fusion_std` | Divide each score vector by its std, then weighted sum | `alpha` = weight on **image** score |
| `fusion_z` | Z-score each score vector, then weighted sum | `alpha` = weight on **image** score |
| `product` | `s_i · s_t` (scores mapped to [0, 1]) | none |
| `min` | `min(s_i, s_t)` (scores mapped to [0, 1]) | none |

> **Careful with `alpha`:** its direction is *not* the same across algorithms (see the last column). Compare results per algorithm, or account for the flip when plotting `alpha` sweeps.

The grid in `result_algo.py` is `alpha ∈ {0.1, 0.25, 0.5, 0.75, 0.9, auto}` × `normalize ∈ {False, True}`. `slerp` runs with `normalize=False` (it normalises internally), and `product`/`min` ignore `alpha`. `auto` runs 100 Optuna trials (step 0.001) minimising the mean target rank on the reference set.

## Tuning process (`alpha="auto"`)

All encoders are frozen, so the only fitted quantity is the image/text weight `alpha`. It is selected on a separate reference dataset (FashionIQ val) and then applied unchanged to Shoes. No Shoes labels are used for tuning.

**Step by step**

1. **Reference set.** `fashionIQ_preprocessing.py` encodes the FashionIQ val split with the same frozen model as Shoes. Each val triplet contributes two queries (one per human caption), and the catalog is the set of val gallery images. These are stored under `data/fashion/<org>/<model>/` and passed to `result_algo.py` as `--ref`.
2. **Search.** For every (algorithm, normalize) configuration, `find_best_alpha` in `algo.py` creates an [Optuna](https://optuna.org) study and runs **100 trials**. Each trial samples `alpha ∈ [0, 1]` in steps of 0.001 (Optuna's default TPE sampler).
3. **Objective.** For a trial, the algorithm builds query scores against the reference catalog, and the trial value is the **mean rank of the ground-truth target** over all reference queries (`get_ranks`: rank = 1 + number of gallery items scoring strictly higher than the target). The study **minimises** this value.
4. **Apply.** The best `alpha` is used to score the Shoes test queries with the same algorithm and `normalize` setting, and the result is written to `results/<org>/<model>/<algo>_auto_<normalize>.json`.
5. **Record.** The chosen value is stored in the result JSON (`alpha`) and appears as `alpha_used` in `result.csv`, so you can compare the tuned value against the fixed-alpha runs (0.1 to 0.9).

**Details**

- Tuning runs separately for each model, algorithm and `normalize` setting, so every row of `result.csv` has its own `alpha`.
- Mean rank is used as the objective because it gives a smoother signal than a thresholded recall.
- `product` and `min` have no `alpha`, so they skip tuning.
- Trials run sequentially (`n_jobs=1`). The Optuna sampler is not seeded; for exactly reproducible `alpha` values, create the study with `sampler=optuna.samplers.TPESampler(seed=<int>)` in `find_best_alpha`.
- Finished configurations are detected by their result file and skipped, so an interrupted run can be restarted without repeating tuning.

## Evaluation

Each query has a single ground-truth target in the catalog; the catalog is the set of unique reference and target images in the test split. `metrics.py` reports, per configuration:

- **Recall@K** for K = 1, 5, 10, 50 (the headline metric) and `R_mean`, their average
- Precision@K, F1@K, Hits@K, DCG@K, NDCG@K
- MRR, mean rank, median rank

Because there is exactly one relevant item per query, `mAP` equals `MRR` and `R-Precision` equals `R@1` in the output.

## Notes

- The scripts run on a single GPU (`cuda:0`).
- `metrix_result_llm_caption.py` reads the catalog id order from `data/shoes/Marqo/marqo-fashionCLIP/catalog/img_idx.json`. The catalog order is the same for every model, so run the fashionCLIP Shoes preprocessing first.
- `fashionIQ_preprocessing.py` uses the official **val** split as the tuning/reference set.

## Demo

A demo notebook is available on Kaggle: [vc-demo](https://www.kaggle.com/code/arvindsharma126/vc-demo).

## Citation / acknowledgements

If you use this code, please cite this repository. Encoders are provided by [OpenCLIP](https://github.com/mlfoundations/open_clip) and [Marqo](https://huggingface.co/Marqo); caption generation uses [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

## License

_To be added._
