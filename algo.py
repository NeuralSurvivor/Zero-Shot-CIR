import numpy as np
import optuna


def find_best_alpha(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    tgt_idx: np.ndarray,
    normalize: bool,
    algo: str,
    merge_func,
):
    def opt_fn(trial: optuna.Trial):
        alpha = trial.suggest_float("alpha", 0, 1, step=0.001)
        scores = merge_func(img_emb, txt_emb, catalog_emb, alpha, normalize)
        return np.mean(get_ranks(scores, tgt_idx))

    study = optuna.create_study(
        study_name=algo,
        direction="minimize",
    )
    study.optimize(opt_fn, 100, n_jobs=1)
    return study.best_params["alpha"]


def normalized(arr: np.ndarray):
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8)


def get_ranks(scores: np.ndarray, tgt_idx: np.ndarray):
    tgt_scores = scores[np.arange(scores.shape[0]), tgt_idx]
    return 1 + np.count_nonzero(scores > tgt_scores[:, None], axis=1)


def process_scores(
    scores: np.ndarray, tgt_idx: np.ndarray, alpha: float, algo: str, normalize: bool
):
    ranked_idx = np.argsort(-scores, axis=1)
    Q, N = scores.shape
    q_range = np.arange(Q)
    tgt_score = scores[q_range, tgt_idx]
    rank_of_each = np.empty_like(ranked_idx)
    np.put_along_axis(
        rank_of_each, ranked_idx, np.broadcast_to(np.arange(1, N + 1), (Q, N)), axis=1
    )
    rank_of_tgt = rank_of_each[np.arange(Q), tgt_idx]
    return {
        "scores": scores.tolist(),
        "tgt_score": tgt_score.tolist(),
        "rank_of_each": rank_of_each.tolist(),
        "rank_of_tgt": rank_of_tgt.tolist(),
        "alpha": alpha,
        "algo": algo,
        "normalize": normalize,
    }


def weighted_sum(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
    query_emb = alpha * img_emb + (1 - alpha) * txt_emb
    if normalize:
        query_emb = normalized(query_emb)
        catalog_emb = normalized(catalog_emb)
    return query_emb @ catalog_emb.T


def fusion_score_min_max(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    score_i = img_emb @ catalog_emb.T
    score_t = txt_emb @ catalog_emb.T
    score_i = (score_i - score_i.min(axis=1, keepdims=True)) / (
        score_i.max(axis=1, keepdims=True) - score_i.min(axis=1, keepdims=True) + 1e-8
    )
    score_t = (score_t - score_t.min(axis=1, keepdims=True)) / (
        score_t.max(axis=1, keepdims=True) - score_t.min(axis=1, keepdims=True) + 1e-8
    )
    return alpha * score_i + (1 - alpha) * score_t


def fusion_score_std(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    score_i = img_emb @ catalog_emb.T
    score_t = txt_emb @ catalog_emb.T
    score_i = score_i / (score_i.std(axis=1, keepdims=True) + 1e-8)
    score_t = score_t / (score_t.std(axis=1, keepdims=True) + 1e-8)
    return alpha * score_i + (1 - alpha) * score_t


def fusion_z_score(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    score_i = img_emb @ catalog_emb.T
    score_t = txt_emb @ catalog_emb.T
    score_i = (score_i - score_i.mean(axis=1, keepdims=True)) / (
        score_i.std(axis=1, keepdims=True) + 1e-8
    )
    score_t = (score_t - score_t.mean(axis=1, keepdims=True)) / (
        score_t.std(axis=1, keepdims=True) + 1e-8
    )
    return alpha * score_i + (1 - alpha) * score_t


def slerp(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    img_emb = normalized(img_emb)
    txt_emb = normalized(txt_emb)
    catalog_emb = normalized(catalog_emb)
    omega = np.arccos(
        np.clip((img_emb * txt_emb).sum(-1, keepdims=True), -1 + 1e-7, 1 - 1e-7)
    )
    so = np.sin(omega)
    query_emb = (np.sin((1 - alpha) * omega) / so) * img_emb + (
        np.sin(alpha * omega) / so
    ) * txt_emb
    query_emb = normalized(query_emb)
    return query_emb @ catalog_emb.T


def product_fusion(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    score_i = (img_emb @ catalog_emb.T + 1) / 2
    score_t = (txt_emb @ catalog_emb.T + 1) / 2
    return score_i * score_t


def min_fusion(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    score_i = (img_emb @ catalog_emb.T + 1) / 2
    score_t = (txt_emb @ catalog_emb.T + 1) / 2
    return np.minimum(score_i, score_t)


def orthogonal_text_injection(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        txt_emb = normalized(txt_emb)
    img_emb = normalized(img_emb)
    catalog_emb = normalized(catalog_emb)
    txt_perp = txt_emb - (txt_emb * img_emb).sum(axis=-1, keepdims=True) * img_emb
    query_emb = img_emb + alpha * txt_perp
    query_emb = normalized(query_emb)
    return query_emb @ catalog_emb.T


def centered_sum(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
    catalog_emb = normalized(catalog_emb)
    img_emb = img_emb - img_emb.mean(0, keepdims=True)
    txt_emb = txt_emb - txt_emb.mean(0, keepdims=True)
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
    query = normalized((1 - alpha) * img_emb + alpha * txt_emb)
    return query @ catalog_emb.T


def catalog_centered_sum(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
        catalog_emb = normalized(catalog_emb)
    mu = catalog_emb.mean(0, keepdims=True)
    img_emb = img_emb - mu
    txt_emb = txt_emb - mu
    catalog_emb = catalog_emb - mu
    if normalize:
        img_emb = normalized(img_emb)
        txt_emb = normalized(txt_emb)
    catalog_emb = normalized(catalog_emb)
    query = normalized((1 - alpha) * img_emb + alpha * txt_emb)
    return query @ catalog_emb.T


def orth_reverse(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float,
    normalize: bool = True,
):
    if normalize:
        txt_emb = normalized(txt_emb)
    img_emb = normalized(img_emb)
    catalog_emb = normalized(catalog_emb)
    img_perp = img_emb - (img_emb * txt_emb).sum(-1, keepdims=True) * txt_emb
    query = normalized(txt_emb + alpha * img_perp)
    return query @ catalog_emb.T


_algo = {
    "weighted_sum": weighted_sum,
    "fusion_min_max": fusion_score_min_max,
    "fusion_std": fusion_score_std,
    "fusion_z": fusion_z_score,
    "slerp": slerp,
    "product": product_fusion,
    "min": min_fusion,
    "orth": orthogonal_text_injection,
    "center": centered_sum,
    "center-cat": catalog_centered_sum,
    "orth-reverse": orth_reverse,
}


def get_algo_list():
    return list(_algo.keys())


def get_result(
    algo: str,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    catalog_emb: np.ndarray,
    alpha: float | str,
    normalize: bool,
    tgt_idx: np.ndarray,
    ref_img_emb: np.ndarray,
    ref_txt_emb: np.ndarray,
    ref_catalog_emb: np.ndarray,
    ref_tgt_idx,
):
    func = _algo[algo]
    if isinstance(alpha, str) and alpha == "auto":
        alpha = find_best_alpha(
            ref_img_emb,
            ref_txt_emb,
            ref_catalog_emb,
            ref_tgt_idx,
            normalize,
            algo,
            func,
        )
    scores = func(img_emb, txt_emb, catalog_emb, alpha, normalize)
    result = process_scores(scores, tgt_idx, alpha, algo, normalize)
    return result
