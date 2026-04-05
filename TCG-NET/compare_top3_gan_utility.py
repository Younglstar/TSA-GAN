#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

from config import DataConfig


class TwoStageUtilityComparator:
    RANDOM_STATE = 42

    def __init__(
        self,
        real_csv_path: str,
        synthetic_csvs: List[str],
        names: Optional[List[str]],
        target_cols: List[str],
        out_dir: str,
        max_rows_train: int = 200_000,
        max_rows_test: int = 100_000,
        max_rows_synth: int = 200_000,
        positive_threshold: float = 1e-8,
        on_weight: float = 0.5,
        pos_weight: float = 0.5,
        log1p_positive_regression: bool = False,
    ):
        self.real_csv_path = real_csv_path
        self.synthetic_csvs = synthetic_csvs
        self.names = names if names is not None else [
            Path(p).parent.parent.name if "csv_outputs" in p else Path(p).stem
            for p in synthetic_csvs
        ]
        self.target_cols = target_cols
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.max_rows_train = int(max_rows_train)
        self.max_rows_test = int(max_rows_test)
        self.max_rows_synth = int(max_rows_synth)
        self.positive_threshold = float(positive_threshold)
        self.on_weight = float(on_weight)
        self.pos_weight = float(pos_weight)
        self.log1p_positive_regression = bool(log1p_positive_regression)

        self.data_conf = DataConfig()
        self.subject_col = self.data_conf.SUBJECT_ID_COL
        self.seq_col = "SEQ_POS"
        self.time_cols = list(getattr(self.data_conf, "TIMEDATA_COLS", []))

        print(f"[数据加载] 正在使用 pyarrow 引擎极速读取真实宽表数据: {self.real_csv_path}")
        # 【优化】使用 pyarrow 处理 2GB 级别的宽表
        self.real_df = pd.read_csv(self.real_csv_path, engine="pyarrow")
        print(f"[数据加载] 真实数据读取完毕，形状: {self.real_df.shape}，正在处理序列...")
        
        self.real_df = self._add_seq_pos_if_needed(self.real_df)

        print("[数据加载] 正在划分训练集/测试集...")
        self.real_train_df, self.real_holdout_df = self._subject_split(
            self.real_df, test_size=0.2, seed=self.RANDOM_STATE
        )
        print("[数据加载] 真实数据准备完毕！")

        self.exclude_base_cols = set(
            [self.subject_col] + [c for c in self.time_cols if c in self.real_df.columns]
        )
        
        self.real_data_cache = {} # 存放预计算特征

    def _add_seq_pos_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.seq_col in df.columns:
            return df
        sort_cols = [self.subject_col] + [c for c in self.time_cols if c in df.columns]
        if len(sort_cols) > 1:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        else:
            df = df.sort_values([self.subject_col]).reset_index(drop=True)
        df[self.seq_col] = df.groupby(self.subject_col).cumcount()
        return df

    def _align_schema(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
        common_cols = [c for c in real_df.columns if c in synth_df.columns]
        return synth_df[common_cols].copy()

    def _subject_split(self, df: pd.DataFrame, test_size=0.2, seed=42):
        subjects = df[self.subject_col].drop_duplicates().to_numpy()
        tr_subj, te_subj = train_test_split(subjects, test_size=test_size, random_state=seed)
        return df[df[self.subject_col].isin(tr_subj)].copy(), df[df[self.subject_col].isin(te_subj)].copy()

    def _build_xy(self, df: pd.DataFrame, target_col: str):
        cols_to_drop = list(self.exclude_base_cols.union({target_col}))
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        # 处理宽表 get_dummies
        X = pd.get_dummies(df.drop(columns=cols_to_drop), dummy_na=True)
        y = pd.to_numeric(df[target_col], errors="coerce")
        valid = np.isfinite(y.to_numpy(dtype=np.float64))
        X = X.loc[valid].copy()
        y = y.loc[valid].copy()
        
        # 删除全 NaN 列
        X = X.loc[:, X.notna().any(axis=0)]
        # 删除常数列
        X = X.loc[:, X.nunique(dropna=True) > 1]
        X = pd.DataFrame(X)
        return X, y

    def _precompute_real_data_features(self):
        """预先计算并缓存真实数据特征，避免在宽表上反复执行 get_dummies"""
        print("[特征工程] 正在预计算真实数据的特征矩阵...")
        for target in self.target_cols:
            if target not in self.real_train_df.columns:
                continue
            X_train, y_train = self._build_xy(self.real_train_df, target)
            X_test, y_test = self._build_xy(self.real_holdout_df, target)
            self.real_data_cache[target] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test
            }
        print("[特征工程] 真实数据特征缓存完成！")

    def _align_train_test_synth(self, X_real_train: pd.DataFrame, X_real_test: pd.DataFrame, X_synth_train: pd.DataFrame):
        cols = X_real_train.columns
        X_real_test = X_real_test.reindex(columns=cols, fill_value=0)
        X_synth_train = X_synth_train.reindex(columns=cols, fill_value=0)

        X_real_train = X_real_train.astype(float)
        X_real_test = X_real_test.astype(float)
        X_synth_train = X_synth_train.astype(float)

        return X_real_train, X_real_test, X_synth_train

    def _sample_rows(self, X: pd.DataFrame, y: pd.Series, max_n: int, stratify: Optional[pd.Series] = None):
        if len(X) <= max_n:
            return X, y
        if stratify is not None and stratify.nunique(dropna=True) >= 2:
            idx = train_test_split(
                np.arange(len(X)),
                train_size=max_n,
                random_state=self.RANDOM_STATE,
                stratify=stratify.to_numpy(),
            )[0]
        else:
            rng = np.random.default_rng(self.RANDOM_STATE)
            idx = rng.choice(len(X), size=max_n, replace=False)
        idx = np.sort(idx)
        return X.iloc[idx].copy(), y.iloc[idx].copy()

    def _make_clf_models(self):
        # 【优化】开启 n_jobs=-1 利用多 CPU，替换宽表杀手原生 SVC
        models = {
            "LogisticRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1, # 利用多核
                    random_state=self.RANDOM_STATE,
                )),
            ]),
            "RandomForest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    random_state=self.RANDOM_STATE,
                    n_estimators=100, # 宽表可适当降低树的数量防止 OOM，保持效果
                    n_jobs=-1,
                )),
            ]),
            "LinearSVC_Calibrated": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CalibratedClassifierCV(
                    LinearSVC(random_state=self.RANDOM_STATE, dual=False, max_iter=2000), 
                    cv=3,
                    n_jobs=-1 # 并行校准
                )),
            ]),
        }
        if HAS_LIGHTGBM:
            models["LightGBM"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMClassifier(
                    random_state=self.RANDOM_STATE, 
                    n_jobs=-1, 
                    verbose=-1
                )),
            ])
        return models

    def _make_reg_models(self):
        models = {
            "LinearRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression(n_jobs=-1)),
            ]),
            "RandomForestRegressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    random_state=self.RANDOM_STATE,
                    n_estimators=100,
                    n_jobs=-1,
                )),
            ]),
        }
        if HAS_LIGHTGBM:
            models["LightGBMRegressor"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(
                    random_state=self.RANDOM_STATE, 
                    n_jobs=-1, 
                    verbose=-1
                )),
            ])
        return models

    def _run_onoff_stage(self, X_real_train, y_real_train, X_real_test, y_real_test, X_synth_train, y_synth_train):
        pos_rate_train = float((y_real_train > self.positive_threshold).mean())

        if pos_rate_train >= 0.98 or pos_rate_train <= 0.02:
            return {
                "status": "skip_almost_constant",
                "real_pos_rate": pos_rate_train,
            }

        y_on_real_train = (y_real_train > self.positive_threshold).astype(int)
        y_on_real_test = (y_real_test > self.positive_threshold).astype(int)
        y_on_synth_train = (y_synth_train > self.positive_threshold).astype(int)

        if y_on_real_train.nunique() < 2 or y_on_real_test.nunique() < 2 or y_on_synth_train.nunique() < 2:
            return {
                "status": "skip_single_class",
                "real_pos_rate": pos_rate_train,
            }

        X_real_train_s, y_on_real_train_s = self._sample_rows(
            X_real_train, y_on_real_train, self.max_rows_train, stratify=y_on_real_train
        )
        X_real_test_s, y_on_real_test_s = self._sample_rows(
            X_real_test, y_on_real_test, self.max_rows_test, stratify=y_on_real_test
        )
        X_synth_train_s, y_on_synth_train_s = self._sample_rows(
            X_synth_train, y_on_synth_train, self.max_rows_synth, stratify=y_on_synth_train
        )

        rows = []
        for model_name, model in self._make_clf_models().items():
            try:
                model_real = model.fit(X_real_train_s, y_on_real_train_s)
                prob_real = model_real.predict_proba(X_real_test_s)[:, 1]
                pred_real = (prob_real >= 0.5).astype(int)

                model_synth = model.fit(X_synth_train_s, y_on_synth_train_s)
                prob_synth = model_synth.predict_proba(X_real_test_s)[:, 1]
                pred_synth = (prob_synth >= 0.5).astype(int)

                auc_real = roc_auc_score(y_on_real_test_s, prob_real)
                auc_synth = roc_auc_score(y_on_real_test_s, prob_synth)
                f1_real = f1_score(y_on_real_test_s, pred_real, zero_division=0)
                f1_synth = f1_score(y_on_real_test_s, pred_synth, zero_division=0)
                bacc_real = balanced_accuracy_score(y_on_real_test_s, pred_real)
                bacc_synth = balanced_accuracy_score(y_on_real_test_s, pred_synth)

                utility_ratio = np.nan
                if np.isfinite(auc_real) and auc_real > 0:
                    utility_ratio = auc_synth / auc_real

                rows.append({
                    "model": model_name,
                    "real_auc": auc_real,
                    "synth_auc": auc_synth,
                    "real_f1": f1_real,
                    "synth_f1": f1_synth,
                    "real_bacc": bacc_real,
                    "synth_bacc": bacc_synth,
                    "utility_ratio": utility_ratio,
                    "utility_clipped": max(0.0, utility_ratio) if pd.notna(utility_ratio) else np.nan,
                    "status": "ok",
                })
            except Exception as e:
                rows.append({"model": model_name, "status": f"error: {e}"})

        rows_df = pd.DataFrame(rows)
        ok_rows = rows_df[rows_df["status"] == "ok"].copy()
        if len(ok_rows) == 0:
            return {"status": "error_all_models_failed"}

        best_row = ok_rows.sort_values("utility_clipped", ascending=False, na_position="last").iloc[0].to_dict()
        return {
            "status": "ok",
            "real_pos_rate": pos_rate_train,
            "best_model": best_row["model"],
            "best_real_auc": float(best_row["real_auc"]),
            "best_synth_auc": float(best_row["synth_auc"]),
            "best_utility_ratio": float(best_row["utility_ratio"]) if pd.notna(best_row["utility_ratio"]) else np.nan,
            "best_utility_clipped": float(best_row["utility_clipped"]) if pd.notna(best_row["utility_clipped"]) else np.nan,
        }

    def _run_positive_reg_stage(self, X_real_train, y_real_train, X_real_test, y_real_test, X_synth_train, y_synth_train):
        mask_real_train = y_real_train > self.positive_threshold
        mask_real_test = y_real_test > self.positive_threshold
        mask_synth_train = y_synth_train > self.positive_threshold

        if mask_real_train.sum() < 100 or mask_real_test.sum() < 50 or mask_synth_train.sum() < 100:
            return {"status": "skip_too_few_positive_samples"}

        X_real_train_p = X_real_train.loc[mask_real_train].copy()
        y_real_train_p = y_real_train.loc[mask_real_train].copy()
        X_real_test_p = X_real_test.loc[mask_real_test].copy()
        y_real_test_p = y_real_test.loc[mask_real_test].copy()
        X_synth_train_p = X_synth_train.loc[mask_synth_train].copy()
        y_synth_train_p = y_synth_train.loc[mask_synth_train].copy()

        X_real_train_p, y_real_train_p = self._sample_rows(X_real_train_p, y_real_train_p, self.max_rows_train)
        X_real_test_p, y_real_test_p = self._sample_rows(X_real_test_p, y_real_test_p, self.max_rows_test)
        X_synth_train_p, y_synth_train_p = self._sample_rows(X_synth_train_p, y_synth_train_p, self.max_rows_synth)

        if self.log1p_positive_regression:
            y_real_train_fit = np.log1p(y_real_train_p.to_numpy())
            y_synth_train_fit = np.log1p(y_synth_train_p.to_numpy())
            inverse_pred = np.expm1
        else:
            y_real_train_fit = y_real_train_p.to_numpy()
            y_synth_train_fit = y_synth_train_p.to_numpy()
            inverse_pred = lambda x: x

        y_true = y_real_test_p.to_numpy()

        rows = []
        for model_name, model in self._make_reg_models().items():
            try:
                model_real = model.fit(X_real_train_p, y_real_train_fit)
                pred_real = inverse_pred(model_real.predict(X_real_test_p))

                model_synth = model.fit(X_synth_train_p, y_synth_train_fit)
                pred_synth = inverse_pred(model_synth.predict(X_real_test_p))

                r2_real = r2_score(y_true, pred_real)
                r2_synth = r2_score(y_true, pred_synth)
                mae_real = mean_absolute_error(y_true, pred_real)
                mae_synth = mean_absolute_error(y_true, pred_synth)

                utility_ratio = np.nan
                if np.isfinite(r2_real) and r2_real > 0:
                    utility_ratio = r2_synth / r2_real

                rows.append({
                    "model": model_name,
                    "real_r2": r2_real,
                    "synth_r2": r2_synth,
                    "real_mae": mae_real,
                    "synth_mae": mae_synth,
                    "utility_ratio": utility_ratio,
                    "utility_clipped": max(0.0, utility_ratio) if pd.notna(utility_ratio) else np.nan,
                    "status": "ok",
                })
            except Exception as e:
                rows.append({"model": model_name, "status": f"error: {e}"})

        rows_df = pd.DataFrame(rows)
        ok_rows = rows_df[rows_df["status"] == "ok"].copy()
        if len(ok_rows) == 0:
            return {"status": "error_all_models_failed"}

        best_row = ok_rows.sort_values("utility_clipped", ascending=False, na_position="last").iloc[0].to_dict()
        return {
            "status": "ok",
            "best_model": best_row["model"],
            "best_real_r2": float(best_row["real_r2"]),
            "best_synth_r2": float(best_row["synth_r2"]),
            "best_real_mae": float(best_row["real_mae"]),
            "best_synth_mae": float(best_row["synth_mae"]),
            "best_utility_ratio": float(best_row["utility_ratio"]) if pd.notna(best_row["utility_ratio"]) else np.nan,
            "best_utility_clipped": float(best_row["utility_clipped"]) if pd.notna(best_row["utility_clipped"]) else np.nan,
        }

    def _run_one_target_for_one_synth(self, synth_df: pd.DataFrame, target_col: str) -> Dict:
        if target_col not in synth_df.columns:
            return {"target_col": target_col, "status": "missing_target"}

        # 【优化】直接从缓存提取特征，避免宽表重复运算
        cached_real = self.real_data_cache.get(target_col)
        if not cached_real:
            return {"target_col": target_col, "status": "missing_target_in_real"}

        X_real_train = cached_real["X_train"].copy()
        y_real_train = cached_real["y_train"].copy()
        X_real_test = cached_real["X_test"].copy()
        y_real_test = cached_real["y_test"].copy()

        X_synth_train, y_synth_train = self._build_xy(synth_df, target_col)

        X_real_train, X_real_test, X_synth_train = self._align_train_test_synth(
            X_real_train, X_real_test, X_synth_train
        )

        on_res = self._run_onoff_stage(X_real_train, y_real_train, X_real_test, y_real_test, X_synth_train, y_synth_train)
        pos_res = self._run_positive_reg_stage(X_real_train, y_real_train, X_real_test, y_real_test, X_synth_train, y_synth_train)

        on_score = float(on_res.get("best_utility_clipped", np.nan)) if on_res.get("status") == "ok" else np.nan
        pos_score = float(pos_res.get("best_utility_clipped", np.nan)) if pos_res.get("status") == "ok" else np.nan

        if np.isnan(on_score) and np.isnan(pos_score):
            final_score = np.nan
        elif np.isnan(on_score):
            final_score = pos_score
        elif np.isnan(pos_score):
            final_score = on_score
        else:
            final_score = self.on_weight * on_score + self.pos_weight * pos_score

        return {
            "target_col": target_col,
            "onoff_status": on_res.get("status"),
            "positive_reg_status": pos_res.get("status"),
            "onoff_best_model": on_res.get("best_model"),
            "positive_reg_best_model": pos_res.get("best_model"),
            "onoff_utility_clipped": on_score,
            "positive_reg_utility_clipped": pos_score,
            "combined_utility_score": final_score,
            "onoff_real_auc": on_res.get("best_real_auc"),
            "onoff_synth_auc": on_res.get("best_synth_auc"),
            "positive_reg_real_r2": pos_res.get("best_real_r2"),
            "positive_reg_synth_r2": pos_res.get("best_synth_r2"),
            "positive_reg_real_mae": pos_res.get("best_real_mae"),
            "positive_reg_synth_mae": pos_res.get("best_synth_mae"),
        }

    def run(self):
        detail_rows = []
        summary_rows = []

        # 执行预计算
        self._precompute_real_data_features()

        for name, synth_path in zip(self.names, self.synthetic_csvs):
            print(f"\n===== two-stage utility compare: {name} =====")
            
            # 【优化】流式加载合成表，用完即释放
            print(f"  正在读取 {synth_path}...")
            synth_df = pd.read_csv(synth_path, engine="pyarrow")
            synth_df = self._add_seq_pos_if_needed(synth_df)
            synth_df = self._align_schema(self.real_df, synth_df)

            candidate_scores = []

            for target in self.target_cols:
                result = self._run_one_target_for_one_synth(synth_df, target)
                row = {
                    "candidate_name": name,
                    "candidate_csv": synth_path,
                    **result,
                }
                detail_rows.append(row)

                if pd.notna(result.get("combined_utility_score", np.nan)):
                    candidate_scores.append(float(result["combined_utility_score"]))

                print(
                    f"  [{target}] combined={result.get('combined_utility_score')} | "
                    f"onoff={result.get('onoff_utility_clipped')} ({result.get('onoff_status')}) | "
                    f"posreg={result.get('positive_reg_utility_clipped')} ({result.get('positive_reg_status')})"
                )

            summary_rows.append({
                "candidate_name": name,
                "candidate_csv": synth_path,
                "num_targets_total": len(self.target_cols),
                "num_targets_success": len(candidate_scores),
                "mean_combined_utility": float(np.mean(candidate_scores)) if candidate_scores else np.nan,
                "median_combined_utility": float(np.median(candidate_scores)) if candidate_scores else np.nan,
            })

            # 【优化】彻底抹除内存中巨大的合成数据集宽表，防御 OOM
            del synth_df
            gc.collect()

        detail_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows).sort_values("mean_combined_utility", ascending=False, na_position="last")

        detail_path = self.out_dir / "twostage_utility_details.csv"
        summary_path = self.out_dir / "twostage_utility_summary.csv"
        best_json_path = self.out_dir / "best_candidate.json"

        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        best_row = None
        if len(summary_df) > 0 and pd.notna(summary_df.iloc[0]["mean_combined_utility"]):
            best_row = summary_df.iloc[0].to_dict()
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_row, f, ensure_ascii=False, indent=2)

        print("\n===== FINAL TWO-STAGE UTILITY RANKING =====")
        print(summary_df.to_string(index=False))
        if best_row is not None:
            print("\n[SELECTED BEST BY TWO-STAGE UTILITY]")
            print(best_row["candidate_name"])
            print(f"mean_combined_utility = {best_row['mean_combined_utility']:.6f}")

        return summary_df, detail_df


def build_argparser():
    parser = argparse.ArgumentParser(description="比较多个 top candidate synthetic_merged_long.csv 的两阶段 utility，选最终最优模型")
    parser.add_argument("--real_csv_path", type=str, required=True, help="真实 long CSV 路径")
    parser.add_argument("--candidate_csvs", nargs="+", required=True, help="多个 synthetic_merged_long.csv 路径")
    parser.add_argument("--candidate_names", nargs="*", default=None, help="候选名称，个数需与 candidate_csvs 对齐；可不填")
    parser.add_argument("--target_cols", nargs="+", default=["leg1v", "leg2v", "air1", "grid", "solar"], help="目标列")
    parser.add_argument("--out_dir", type=str, default="./gan_round3_utility_twostage", help="输出目录")
    parser.add_argument("--max_rows_train", type=int, default=200000)
    parser.add_argument("--max_rows_test", type=int, default=100000)
    parser.add_argument("--max_rows_synth", type=int, default=200000)
    parser.add_argument("--positive_threshold", type=float, default=1e-8)
    parser.add_argument("--on_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=0.5)
    parser.add_argument("--log1p_positive_regression", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()

    if args.candidate_names is not None and len(args.candidate_names) not in (0, len(args.candidate_csvs)):
        raise ValueError("candidate_names 个数必须为 0 或与 candidate_csvs 相同")
    names = None if (args.candidate_names is None or len(args.candidate_names) == 0) else args.candidate_names

    runner = TwoStageUtilityComparator(
        real_csv_path=args.real_csv_path,
        synthetic_csvs=args.candidate_csvs,
        names=names,
        target_cols=args.target_cols,
        out_dir=args.out_dir,
        max_rows_train=args.max_rows_train,
        max_rows_test=args.max_rows_test,
        max_rows_synth=args.max_rows_synth,
        positive_threshold=args.positive_threshold,
        on_weight=args.on_weight,
        pos_weight=args.pos_weight,
        log1p_positive_regression=args.log1p_positive_regression,
    )
    runner.run()


if __name__ == "__main__":
    main()