from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np


@dataclass
class CausalTargetBuilderConfig:
    cache_dir: str = "./cache_encdec"
    feature_dims_file: str = "feature_dims.pkl"

    x_filename: str = "encdec_X.f32"
    m_filename: str = "encdec_M.u8"

    max_lag: int = 1
    min_valid_points: int = 10
    ridge_alpha: float = 1e-4

    output_path: str = "./cache_encdec/A_target.npy"
    zero_diagonal: bool = True

    standardize_per_subject: bool = False
    use_only_temporal_features: bool = True
    verbose: bool = True

    threshold: float = 0.0
    keep_topk_per_target: int = 0
    symmetrize: bool = False

    save_improve_matrix: bool = True
    save_valid_count_matrix: bool = True
    save_metadata: bool = True


def load_feature_dims(cfg: CausalTargetBuilderConfig) -> dict:
    path = os.path.join(cfg.cache_dir, cfg.feature_dims_file)
    if not os.path.exists(path):
        path = cfg.feature_dims_file
    with open(path, "rb") as f:
        return pickle.load(f)


def open_memmaps(cfg: CausalTargetBuilderConfig, feature_dims: dict) -> Tuple[np.memmap, np.memmap]:
    x_path = os.path.join(cfg.cache_dir, cfg.x_filename)
    m_path = os.path.join(cfg.cache_dir, cfg.m_filename)

    if not (os.path.exists(x_path) and os.path.exists(m_path)):
        raise FileNotFoundError(f"memmap 文件不存在:\n{x_path}\n{m_path}")

    n = int(feature_dims["num_subjects"])
    t = int(feature_dims["sequence_length"])
    d = int(feature_dims["input_dim"])

    x = np.memmap(x_path, dtype="float32", mode="r", shape=(n, t, d))
    m = np.memmap(m_path, dtype="uint8", mode="r", shape=(n, t, d))
    return x, m


def _normalize_slice_spec(obj) -> List[Tuple[str, List[int]]]:
    groups: List[Tuple[str, List[int]]] = []

    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return groups

    for key, value in items:
        name = str(key)

        if isinstance(value, dict):
            if "indices" in value:
                idxs = list(map(int, value["indices"]))
            elif "slice" in value:
                s, e = value["slice"]
                idxs = list(range(int(s), int(e)))
            elif "start" in value and "end" in value:
                idxs = list(range(int(value["start"]), int(value["end"])))
            else:
                continue
        elif isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, (int, np.integer)) for v in value):
            s, e = value
            idxs = list(range(int(s), int(e)))
        elif isinstance(value, (tuple, list)):
            idxs = list(map(int, value))
        else:
            continue

        idxs = [i for i in idxs if i >= 0]
        if idxs:
            groups.append((name, idxs))

    return groups


def discover_feature_groups(feature_dims: dict, input_dim: int, use_only_temporal_features: bool) -> List[Tuple[str, List[int]]]:
    candidate_keys = [
        "graph_feature_slices",
        "feature_slices",
        "raw_feature_slices",
        "original_feature_slices",
        "feature_groups",
    ]

    groups: List[Tuple[str, List[int]]] = []
    for key in candidate_keys:
        if key in feature_dims:
            groups = _normalize_slice_spec(feature_dims[key])
            if groups:
                break

    if use_only_temporal_features:
        temporal_keys = ["temporal_feature_slices", "temporal_groups"]
        temporal_groups: List[Tuple[str, List[int]]] = []
        for key in temporal_keys:
            if key in feature_dims:
                temporal_groups = _normalize_slice_spec(feature_dims[key])
                if temporal_groups:
                    groups = temporal_groups
                    break

    if not groups:
        num_graph_features = int(feature_dims.get("num_total_features", input_dim))
        if num_graph_features > input_dim:
            raise ValueError(
                f"num_total_features={num_graph_features} > input_dim={input_dim}，且 feature_dims 中未提供 feature_slices。"
            )

        if num_graph_features == input_dim:
            groups = [(f"f{i}", [i]) for i in range(input_dim)]
        else:
            bounds = np.linspace(0, input_dim, num_graph_features + 1, dtype=int)
            for i in range(num_graph_features):
                s = int(bounds[i])
                e = int(bounds[i + 1])
                if e <= s:
                    e = min(input_dim, s + 1)
                groups.append((f"f{i}", list(range(s, e))))

    return groups


def aggregate_group_series(
    x: np.ndarray,
    m: np.ndarray,
    groups: Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n, t, _ = x.shape
    f = len(groups)

    node_x = np.zeros((n, t, f), dtype=np.float32)
    node_m = np.zeros((n, t, f), dtype=np.uint8)
    names: List[str] = []

    for g_idx, (name, idxs) in enumerate(groups):
        idxs_arr = np.asarray(list(idxs), dtype=np.int64)
        xg = x[:, :, idxs_arr]
        mg = m[:, :, idxs_arr].astype(np.float32, copy=False)

        valid_count = mg.sum(axis=2)
        summed = (xg * mg).sum(axis=2)
        avg = summed / np.clip(valid_count, 1.0, None)

        node_x[:, :, g_idx] = avg.astype(np.float32, copy=False)
        node_m[:, :, g_idx] = (valid_count > 0).astype(np.uint8, copy=False)
        names.append(name)

    return node_x, node_m, names


def standardize_all_subjects(node_x: np.ndarray, node_m: np.ndarray) -> np.ndarray:
    valid = node_m.astype(bool)
    valid_f = valid.astype(np.float32)

    counts = valid_f.sum(axis=1)
    sums = (node_x * valid_f).sum(axis=1)
    means = sums / np.clip(counts, 1.0, None)

    centered = node_x - means[:, None, :]
    vars_ = ((centered ** 2) * valid_f).sum(axis=1) / np.clip(counts, 1.0, None)
    stds = np.sqrt(vars_)

    too_few = counts < 2
    stds = np.where((stds < 1e-8) | too_few, 1.0, stds)

    node_x_std = np.where(valid, centered / stds[:, None, :], 0.0).astype(np.float32, copy=False)
    return node_x_std


def prepare_lagged_views(
    node_x: np.ndarray,
    node_m: np.ndarray,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_lag <= 0:
        raise ValueError("max_lag 必须 >= 1")

    n, t, f = node_x.shape
    if max_lag >= t:
        raise ValueError(f"max_lag={max_lag} 必须小于序列长度 T={t}")

    y = node_x[:, max_lag:, :]
    y_valid = node_m[:, max_lag:, :].astype(bool)

    hist_list = []
    hist_valid_list = []
    for lag in range(1, max_lag + 1):
        hist_list.append(node_x[:, max_lag - lag:t - lag, :])
        hist_valid_list.append(node_m[:, max_lag - lag:t - lag, :].astype(bool))

    hist = np.stack(hist_list, axis=-1)
    hist_valid = np.stack(hist_valid_list, axis=-1)
    hist_all_valid = hist_valid.all(axis=-1)

    s = y.shape[0] * y.shape[1]
    y_flat = y.reshape(s, f)
    y_valid_flat = y_valid.reshape(s, f)
    hist_flat = hist.reshape(s, f, max_lag)
    hist_all_valid_flat = hist_all_valid.reshape(s, f)

    return y_flat, y_valid_flat, hist_flat, hist_all_valid_flat


def ridge_sse(y: np.ndarray, x: np.ndarray, ridge_alpha: float) -> float:
    n = x.shape[0]
    x1 = np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1)
    p = x1.shape[1]

    xtx = x1.T @ x1
    reg = ridge_alpha * np.eye(p, dtype=np.float64)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtx + reg, x1.T @ y)
    resid = y - x1 @ beta
    return float((resid ** 2).sum())


def compute_pair_stats_from_views(
    yj: np.ndarray,
    xr_j: np.ndarray,
    base_valid_j: np.ndarray,
    cause_hist_i: np.ndarray,
    cause_valid_i: np.ndarray,
    min_valid_points: int,
    ridge_alpha: float,
) -> Tuple[float, float, int]:
    valid = base_valid_j & cause_valid_i
    num_valid = int(valid.sum())
    if num_valid < min_valid_points:
        return 0.0, 0.0, num_valid

    y = yj[valid].astype(np.float64, copy=False)
    xr = xr_j[valid].astype(np.float64, copy=False)
    xc = cause_hist_i[valid].astype(np.float64, copy=False)
    xf = np.concatenate([xr, xc], axis=1)

    sse_r = ridge_sse(y, xr, ridge_alpha)
    sse_f = ridge_sse(y, xf, ridge_alpha)

    improve = max(0.0, (sse_r - sse_f) / (sse_r + 1e-12))
    score = 1.0 - np.exp(-5.0 * improve)
    return float(np.clip(score, 0.0, 1.0)), float(improve), num_valid


def apply_postprocessing(
    a: np.ndarray,
    *,
    threshold: float,
    keep_topk_per_target: int,
    zero_diagonal: bool,
    symmetrize: bool,
) -> np.ndarray:
    out = a.copy()

    if threshold > 0:
        out = np.where(out >= threshold, out, 0.0)

    if keep_topk_per_target > 0:
        f = out.shape[0]
        for target_j in range(f):
            col = out[:, target_j].copy()
            if zero_diagonal:
                col[target_j] = 0.0

            nz = np.flatnonzero(col > 0)
            if len(nz) > keep_topk_per_target:
                top_idx = np.argpartition(col, -keep_topk_per_target)[-keep_topk_per_target:]
                mask = np.zeros_like(col, dtype=bool)
                mask[top_idx] = True
                col = np.where(mask, col, 0.0)

            out[:, target_j] = col

    if symmetrize:
        out = np.maximum(out, out.T)

    if zero_diagonal:
        np.fill_diagonal(out, 0.0)

    return out.astype(np.float32, copy=False)


def summarize_matrix(name: str, a: np.ndarray) -> str:
    nz = int((a > 0).sum())
    total = int(a.size)
    vals = a[a > 0]
    if vals.size == 0:
        return f"{name}: all zero | shape={a.shape}"
    q = np.quantile(vals, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return (
        f"{name}: shape={a.shape} min={float(a.min()):.6f} max={float(a.max()):.6f} "
        f"mean={float(a.mean()):.6f} nonzero={nz}/{total} | "
        f"nz_q25={q[0]:.6f} nz_q50={q[1]:.6f} nz_q75={q[2]:.6f} "
        f"nz_q90={q[3]:.6f} nz_q95={q[4]:.6f} nz_q99={q[5]:.6f}"
    )


def build_global_causal_target(cfg: CausalTargetBuilderConfig) -> np.ndarray:
    feature_dims = load_feature_dims(cfg)
    x_mm, m_mm = open_memmaps(cfg, feature_dims)

    input_dim = int(feature_dims["input_dim"])
    groups = discover_feature_groups(
        feature_dims,
        input_dim=input_dim,
        use_only_temporal_features=cfg.use_only_temporal_features,
    )

    node_x, node_m, names = aggregate_group_series(x_mm, m_mm, groups)

    if cfg.standardize_per_subject:
        node_x = standardize_all_subjects(node_x, node_m)

    y_flat, y_valid_flat, hist_flat, hist_all_valid_flat = prepare_lagged_views(
        node_x=node_x,
        node_m=node_m,
        max_lag=cfg.max_lag,
    )

    f = y_flat.shape[1]
    a = np.zeros((f, f), dtype=np.float32)
    improve_mat = np.zeros((f, f), dtype=np.float32)
    valid_count_mat = np.zeros((f, f), dtype=np.int32)

    if cfg.verbose:
        num_samples = y_flat.shape[0]
        print(
            f"Building A_target | nodes={f} max_lag={cfg.max_lag} "
            f"min_valid_points={cfg.min_valid_points} flattened_samples={num_samples} "
            f"use_only_temporal_features={cfg.use_only_temporal_features} "
            f"standardize_per_subject={cfg.standardize_per_subject}"
        )

    for target_j in range(f):
        yj = y_flat[:, target_j]
        xr_j = hist_flat[:, target_j, :]
        base_valid_j = y_valid_flat[:, target_j] & hist_all_valid_flat[:, target_j]

        for cause_i in range(f):
            if cause_i == target_j:
                continue

            score, improve, num_valid = compute_pair_stats_from_views(
                yj=yj,
                xr_j=xr_j,
                base_valid_j=base_valid_j,
                cause_hist_i=hist_flat[:, cause_i, :],
                cause_valid_i=hist_all_valid_flat[:, cause_i],
                min_valid_points=cfg.min_valid_points,
                ridge_alpha=cfg.ridge_alpha,
            )
            a[cause_i, target_j] = score
            improve_mat[cause_i, target_j] = improve
            valid_count_mat[cause_i, target_j] = num_valid

    if cfg.zero_diagonal:
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(improve_mat, 0.0)
        np.fill_diagonal(valid_count_mat, 0)

    a_post = apply_postprocessing(
        a,
        threshold=cfg.threshold,
        keep_topk_per_target=cfg.keep_topk_per_target,
        zero_diagonal=cfg.zero_diagonal,
        symmetrize=cfg.symmetrize,
    )

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    np.save(cfg.output_path, a_post)

    base = os.path.splitext(cfg.output_path)[0]
    names_path = base + "_names.pkl"
    with open(names_path, "wb") as fobj:
        pickle.dump(names, fobj)

    if cfg.save_improve_matrix:
        np.save(base + "_improve.npy", improve_mat)

    if cfg.save_valid_count_matrix:
        np.save(base + "_valid_count.npy", valid_count_mat)

    if cfg.save_metadata:
        meta: Dict[str, Any] = {
            "names": names,
            "config": cfg.__dict__.copy(),
            "summary_pre": summarize_matrix("A_target_pre", a),
            "summary_post": summarize_matrix("A_target_post", a_post),
            "valid_count_nonzero_mean": float(valid_count_mat[valid_count_mat > 0].mean()) if np.any(valid_count_mat > 0) else 0.0,
            "valid_count_nonzero_min": int(valid_count_mat[valid_count_mat > 0].min()) if np.any(valid_count_mat > 0) else 0,
            "valid_count_nonzero_max": int(valid_count_mat.max()),
        }
        with open(base + "_meta.pkl", "wb") as fobj:
            pickle.dump(meta, fobj)

    if cfg.verbose:
        print(f"Saved A_target -> {cfg.output_path}, shape={a_post.shape}")
        print(f"Saved node names -> {names_path}")
        if cfg.save_improve_matrix:
            print(f"Saved improve matrix -> {base + '_improve.npy'}")
        if cfg.save_valid_count_matrix:
            print(f"Saved valid count matrix -> {base + '_valid_count.npy'}")
        if cfg.save_metadata:
            print(f"Saved metadata -> {base + '_meta.pkl'}")
        print(summarize_matrix("A_target_pre", a))
        print(summarize_matrix("A_target_post", a_post))

    return a_post


if __name__ == "__main__":
    cfg = CausalTargetBuilderConfig()
    build_global_causal_target(cfg)
