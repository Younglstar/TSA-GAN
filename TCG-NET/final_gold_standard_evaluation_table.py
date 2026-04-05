# file: final_gold_standard_evaluation_table_multitarget_nanfixed.py
from __future__ import annotations

import os
import json
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from tqdm import tqdm

from config import DataConfig

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False


class GoldStandardComparator:
    SYNTHETIC_MERGED_PATH = "./cache_encdec/final_synthetic_output/synthetic_merged_long.csv"
    REAL_SOURCE_PATH = None

    UTILITY_TARGET_COL = None
    AUTO_UTILITY_SCAN = True
    UTILITY_MAX_TARGETS = None
    UTILITY_MIN_NON_NULL_RATIO = 0.20
    UTILITY_MAX_UNIQUE_RATIO = 0.95
    UTILITY_MIN_CLASS_COUNT = 20
    UTILITY_REG_MAX_SAMPLES = 20000
    UTILITY_REPORT_TOPK = 30

    MAX_TSNE_SAMPLES = 4000
    MAX_PRIVACY_SAMPLES = 50000
    RANDOM_STATE = 42

    def __init__(self):
        self.output_dir = "final_gold_standard_report_table"
        os.makedirs(os.path.join(self.output_dir, "1_fidelity"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "2_utility"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "3_privacy"), exist_ok=True)

        self.report_content = "--- Synthetic Data Gold Standard Report (Table-space) ---\n\n"

        self.data_conf = DataConfig()
        self.subject_col = self.data_conf.SUBJECT_ID_COL
        self.seq_col = "SEQ_POS"
        self.time_cols = list(getattr(self.data_conf, "TIMEDATA_COLS", []))

        print("📊 正在加载表空间评估所需的数据...")
        real_path = self.REAL_SOURCE_PATH if self.REAL_SOURCE_PATH else self.data_conf.INPUT_FILE
        synth_path = self.SYNTHETIC_MERGED_PATH

        if not os.path.exists(real_path):
            raise FileNotFoundError(f"未找到真实源表: {real_path}")
        if not os.path.exists(synth_path):
            raise FileNotFoundError(f"未找到合成长表: {synth_path}")

        self.real_df = pd.read_csv(real_path)
        self.synthetic_df = pd.read_csv(synth_path)

        print(f"  - real raw shape      : {self.real_df.shape}")
        print(f"  - synthetic raw shape : {self.synthetic_df.shape}")

        self.real_df = self._add_seq_pos_if_needed(self.real_df)
        self.real_df, self.synthetic_df = self._align_schema(self.real_df, self.synthetic_df)

        self.exclude_cols = {self.subject_col, self.seq_col}
        self.exclude_cols.update([c for c in self.time_cols if c in self.real_df.columns])

        self.numeric_cols = [
            c for c in self.real_df.columns
            if c not in self.exclude_cols and pd.api.types.is_numeric_dtype(self.real_df[c])
        ]
        self.categorical_cols = [
            c for c in self.real_df.columns
            if c not in self.exclude_cols and c not in self.numeric_cols
        ]

        self.static_num_cols = [c for c in getattr(self.data_conf, "STATIC_NUMERICAL_FEATURES", []) if c in self.real_df.columns]
        self.static_cat_cols = [c for c in getattr(self.data_conf, "STATIC_CATEGORICAL_FEATURES", []) if c in self.real_df.columns]
        self.temporal_num_cols = [c for c in getattr(self.data_conf, "TEMPORAL_NUMERICAL_FEATURES", []) if c in self.real_df.columns]
        self.temporal_cat_cols = [c for c in getattr(self.data_conf, "TEMPORAL_CATEGORICAL_FEATURES", []) if c in self.real_df.columns]

        self.real_train_df, self.real_holdout_df = self._subject_split(self.real_df, test_size=0.2, seed=self.RANDOM_STATE)

        print("数据加载完成！")
        print(f"  - real aligned shape      : {self.real_df.shape}")
        print(f"  - synthetic aligned shape : {self.synthetic_df.shape}")
        print(f"  - numeric cols            : {len(self.numeric_cols)}")
        print(f"  - categorical cols        : {len(self.categorical_cols)}")
        print(f"  - real train shape        : {self.real_train_df.shape}")
        print(f"  - real holdout shape      : {self.real_holdout_df.shape}")

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

    def _align_schema(self, real_df: pd.DataFrame, synth_df: pd.DataFrame):
        common_cols = [c for c in real_df.columns if c in synth_df.columns]
        return real_df[common_cols].copy(), synth_df[common_cols].copy()

    def _subject_split(self, df: pd.DataFrame, test_size=0.2, seed=42):
        subjects = df[self.subject_col].drop_duplicates().to_numpy()
        tr_subj, te_subj = train_test_split(subjects, test_size=test_size, random_state=seed)
        return df[df[self.subject_col].isin(tr_subj)].copy(), df[df[self.subject_col].isin(te_subj)].copy()

    def _get_subject_level_repr(self, df: pd.DataFrame, exclude_target: str | None = None) -> pd.DataFrame:
        parts = []
        base = pd.DataFrame({self.subject_col: df[self.subject_col].drop_duplicates().sort_values()})
        parts.append(base.set_index(self.subject_col))

        static_cols = [c for c in (self.static_num_cols + self.static_cat_cols) if c in df.columns]
        if exclude_target is not None:
            static_cols = [c for c in static_cols if c != exclude_target]
        if static_cols:
            parts.append(df.groupby(self.subject_col)[static_cols].first())

        temp_num = [c for c in self.temporal_num_cols if c in df.columns]
        if exclude_target is not None:
            temp_num = [c for c in temp_num if c != exclude_target]
        if temp_num:
            agg_num = df.groupby(self.subject_col)[temp_num].agg(["mean", "std", "min", "max"])
            agg_num.columns = [f"{c}_{stat}" for c, stat in agg_num.columns]
            parts.append(agg_num)

        temp_cat = [c for c in self.temporal_cat_cols if c in df.columns]
        if exclude_target is not None:
            temp_cat = [c for c in temp_cat if c != exclude_target]
        for col in temp_cat:
            mode_series = df.groupby(self.subject_col)[col].agg(
                lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan
            )
            parts.append(mode_series.rename(f"{col}_mode").to_frame())

        return pd.concat(parts, axis=1).reset_index()

    def _numeric_subject_repr(self, df: pd.DataFrame) -> np.ndarray:
        subj_df = self._get_subject_level_repr(df)
        num_cols = [c for c in subj_df.columns if c != self.subject_col and pd.api.types.is_numeric_dtype(subj_df[c])]
        x = subj_df[num_cols].copy().replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True))
        return x.to_numpy(dtype=np.float32)

    def _plot_and_save(self, fig, path: str):
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _safe_auc(self, y_true, y_score):
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_score)
        except Exception:
            return np.nan

    def _build_subject_level_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if target_col not in df.columns:
            raise KeyError(f"target_col 不在 df 中: {target_col}")
        if target_col in self.static_num_cols or target_col in self.static_cat_cols:
            out = df.groupby(self.subject_col)[target_col].first().rename(target_col).reset_index()
        elif target_col in self.temporal_num_cols:
            out = df.groupby(self.subject_col)[target_col].mean().rename(target_col).reset_index()
        elif target_col in self.temporal_cat_cols:
            out = df.groupby(self.subject_col)[target_col].agg(
                lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan
            ).rename(target_col).reset_index()
        else:
            out = df.groupby(self.subject_col)[target_col].first().rename(target_col).reset_index()
        return out

    def _prepare_subject_level_task_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        subj_df = self._get_subject_level_repr(df, exclude_target=target_col)
        target_by_subj = self._build_subject_level_target(df, target_col)
        subj_df = subj_df.merge(target_by_subj, on=self.subject_col, how="left")
        return subj_df

    def _get_utility_candidate_cols(self) -> List[str]:
        candidates = []
        for col in self.real_df.columns:
            if col in self.exclude_cols:
                continue
            s = self.real_df[col]
            non_null_ratio = 1.0 - s.isna().mean()
            if non_null_ratio < self.UTILITY_MIN_NON_NULL_RATIO:
                continue
            nunique = s.nunique(dropna=True)
            if nunique < 2:
                continue
            unique_ratio = nunique / max(len(s.dropna()), 1)
            if unique_ratio > self.UTILITY_MAX_UNIQUE_RATIO:
                continue
            candidates.append(col)

        preferred = []
        preferred += [c for c in self.static_cat_cols if c in candidates]
        preferred += [c for c in self.static_num_cols if c in candidates]
        preferred += [c for c in self.temporal_cat_cols if c in candidates]
        preferred += [c for c in self.temporal_num_cols if c in candidates]
        preferred += [c for c in candidates if c not in preferred]

        if self.UTILITY_MAX_TARGETS is not None:
            preferred = preferred[: self.UTILITY_MAX_TARGETS]
        return preferred

    def _infer_task_type(self, y: pd.Series) -> str:
        y = pd.Series(y).dropna()
        if pd.api.types.is_numeric_dtype(y):
            nunique = y.nunique()
            if nunique <= 2:
                return "binary_clf"
            if nunique <= 20 and np.allclose(y, y.round()):
                return "multiclass_clf"
            return "regression"
        else:
            return "binary_clf" if y.nunique() <= 2 else "multiclass_clf"

    def _build_feature_matrices(self, real_train_subj, real_test_subj, synth_subj, target_col):
        drop_cols = [self.subject_col, target_col]
        X_real_train = pd.get_dummies(real_train_subj.drop(columns=[c for c in drop_cols if c in real_train_subj.columns]), dummy_na=True)
        X_real_test = pd.get_dummies(real_test_subj.drop(columns=[c for c in drop_cols if c in real_test_subj.columns]), dummy_na=True)
        X_synth_train = pd.get_dummies(synth_subj.drop(columns=[c for c in drop_cols if c in synth_subj.columns]), dummy_na=True)

        X_real_train, X_real_test = X_real_train.align(X_real_test, join="outer", axis=1, fill_value=0)
        X_real_train, X_synth_train = X_real_train.align(X_synth_train, join="outer", axis=1, fill_value=0)
        X_real_test = X_real_test.reindex(columns=X_real_train.columns, fill_value=0)
        X_synth_train = X_synth_train.reindex(columns=X_real_train.columns, fill_value=0)
        return X_real_train, X_real_test, X_synth_train

    def _make_regression_models(self):
        models = {
            "LinearRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]),
            "RandomForestRegressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=self.RANDOM_STATE, n_estimators=200, n_jobs=-1)),
            ]),
        }
        if HAS_LIGHTGBM:
            models["LightGBMRegressor"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(random_state=self.RANDOM_STATE, verbose=-1)),
            ])
        return models

    def _make_classification_models(self, task_type: str):
        if task_type == "binary_clf":
            return {
                "LogisticRegression": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LogisticRegression(max_iter=2000, random_state=self.RANDOM_STATE)),
                ]),
                "RandomForest": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(random_state=self.RANDOM_STATE)),
                ]),
                "SVM": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", SVC(probability=True, random_state=self.RANDOM_STATE)),
                ]),
            }

        models = {
            "RandomForest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=self.RANDOM_STATE)),
            ]),
        }
        if HAS_LIGHTGBM:
            models["LightGBM"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMClassifier(random_state=self.RANDOM_STATE, verbose=-1)),
            ])
        return models

    def _run_single_utility_task(self, target_col: str):
        real_train_subj = self._prepare_subject_level_task_data(self.real_train_df, target_col)
        real_test_subj = self._prepare_subject_level_task_data(self.real_holdout_df, target_col)
        synth_subj = self._prepare_subject_level_task_data(self.synthetic_df, target_col)

        real_train_subj = real_train_subj[~pd.isna(real_train_subj[target_col])].copy()
        real_test_subj = real_test_subj[~pd.isna(real_test_subj[target_col])].copy()
        synth_subj = synth_subj[~pd.isna(synth_subj[target_col])].copy()

        if len(real_train_subj) < 100 or len(real_test_subj) < 50 or len(synth_subj) < 100:
            return {"target_col": target_col, "status": "skip_too_few_samples"}

        task_type = self._infer_task_type(real_train_subj[target_col])
        X_real_train, X_real_test, X_synth_train = self._build_feature_matrices(real_train_subj, real_test_subj, synth_subj, target_col)

        result = {
            "target_col": target_col,
            "task_type": task_type,
            "n_real_train": int(len(real_train_subj)),
            "n_real_test": int(len(real_test_subj)),
            "n_synth_train": int(len(synth_subj)),
            "nan_real_train": int(X_real_train.isna().sum().sum()),
            "nan_real_test": int(X_real_test.isna().sum().sum()),
            "nan_synth_train": int(X_synth_train.isna().sum().sum()),
            "status": "ok",
        }

        if task_type in ("binary_clf", "multiclass_clf"):
            y_real_train = real_train_subj[target_col].astype("category")
            y_real_test = real_test_subj[target_col].astype("category")
            y_synth_train = synth_subj[target_col].astype("category")

            all_cats = pd.Index(sorted(set(y_real_train.astype(str)) | set(y_real_test.astype(str)) | set(y_synth_train.astype(str))))
            cat_map = {c: i for i, c in enumerate(all_cats)}

            y_real_train = y_real_train.astype(str).map(cat_map)
            y_real_test = y_real_test.astype(str).map(cat_map)
            y_synth_train = y_synth_train.astype(str).map(cat_map)

            vc = pd.Series(y_real_train).value_counts()
            if (vc < self.UTILITY_MIN_CLASS_COUNT).any():
                result["status"] = "skip_small_class"
                result["min_class_count_real_train"] = int(vc.min())
                return result

            models = self._make_classification_models(task_type)
            rows = []
            for model_name, model in models.items():
                try:
                    if task_type == "binary_clf":
                        model_real = model.fit(X_real_train, y_real_train)
                        prob_real = model_real.predict_proba(X_real_test)[:, 1]
                        pred_real = (prob_real >= 0.5).astype(int)

                        model_synth = model.fit(X_synth_train, y_synth_train)
                        prob_synth = model_synth.predict_proba(X_real_test)[:, 1]
                        pred_synth = (prob_synth >= 0.5).astype(int)

                        auc_real = self._safe_auc(y_real_test, prob_real)
                        auc_synth = self._safe_auc(y_real_test, prob_synth)
                        f1_real = f1_score(y_real_test, pred_real, zero_division=0)
                        f1_synth = f1_score(y_real_test, pred_synth, zero_division=0)
                        acc_real = accuracy_score(y_real_test, pred_real)
                        acc_synth = accuracy_score(y_real_test, pred_synth)

                        utility_score = np.nan
                        if np.isfinite(auc_real) and auc_real > 0:
                            utility_score = 100.0 * auc_synth / auc_real

                        rows.append({
                            "model": model_name, "primary_metric": "auroc",
                            "real_score": auc_real, "synth_score": auc_synth,
                            "utility_score_pct": utility_score,
                            "real_f1": f1_real, "synth_f1": f1_synth,
                            "real_acc": acc_real, "synth_acc": acc_synth,
                            "model_status": "ok",
                        })
                    else:
                        model_real = model.fit(X_real_train, y_real_train)
                        pred_real = model_real.predict(X_real_test)

                        model_synth = model.fit(X_synth_train, y_synth_train)
                        pred_synth = model_synth.predict(X_real_test)

                        f1_real = f1_score(y_real_test, pred_real, average="macro", zero_division=0)
                        f1_synth = f1_score(y_real_test, pred_synth, average="macro", zero_division=0)
                        bacc_real = balanced_accuracy_score(y_real_test, pred_real)
                        bacc_synth = balanced_accuracy_score(y_real_test, pred_synth)

                        utility_score = np.nan
                        if np.isfinite(f1_real) and f1_real > 0:
                            utility_score = 100.0 * f1_synth / f1_real

                        rows.append({
                            "model": model_name, "primary_metric": "macro_f1",
                            "real_score": f1_real, "synth_score": f1_synth,
                            "utility_score_pct": utility_score,
                            "real_bacc": bacc_real, "synth_bacc": bacc_synth,
                            "model_status": "ok",
                        })
                except Exception as e:
                    rows.append({"model": model_name, "model_status": f"error: {e}"})

            rows_df = pd.DataFrame(rows)
            ok_rows = rows_df[rows_df["model_status"] == "ok"].copy()
            if len(ok_rows) == 0:
                result["status"] = "error_all_models_failed"
                result["details"] = rows_df.to_dict(orient="records")
                return result

            best_row = ok_rows.sort_values("utility_score_pct", ascending=False, na_position="last").iloc[0].to_dict()
            result.update({
                "details": rows_df.to_dict(orient="records"),
                "best_model": best_row["model"],
                "primary_metric": best_row["primary_metric"],
                "best_real_score": float(best_row["real_score"]),
                "best_synth_score": float(best_row["synth_score"]),
                "best_utility_score_pct": float(best_row["utility_score_pct"]) if pd.notna(best_row["utility_score_pct"]) else np.nan,
            })
            return result

        # regression
        if len(real_train_subj) > self.UTILITY_REG_MAX_SAMPLES:
            real_train_subj = real_train_subj.sample(self.UTILITY_REG_MAX_SAMPLES, random_state=self.RANDOM_STATE)
        if len(real_test_subj) > self.UTILITY_REG_MAX_SAMPLES:
            real_test_subj = real_test_subj.sample(self.UTILITY_REG_MAX_SAMPLES, random_state=self.RANDOM_STATE)
        if len(synth_subj) > self.UTILITY_REG_MAX_SAMPLES:
            synth_subj = synth_subj.sample(self.UTILITY_REG_MAX_SAMPLES, random_state=self.RANDOM_STATE)

        X_real_train, X_real_test, X_synth_train = self._build_feature_matrices(real_train_subj, real_test_subj, synth_subj, target_col)
        result["nan_real_train"] = int(X_real_train.isna().sum().sum())
        result["nan_real_test"] = int(X_real_test.isna().sum().sum())
        result["nan_synth_train"] = int(X_synth_train.isna().sum().sum())

        y_real_train = pd.to_numeric(real_train_subj[target_col], errors="coerce")
        y_real_test = pd.to_numeric(real_test_subj[target_col], errors="coerce")
        y_synth_train = pd.to_numeric(synth_subj[target_col], errors="coerce")

        valid_real_train = np.isfinite(y_real_train)
        valid_real_test = np.isfinite(y_real_test)
        valid_synth_train = np.isfinite(y_synth_train)

        X_real_train = X_real_train.loc[valid_real_train]
        y_real_train = y_real_train.loc[valid_real_train]
        X_real_test = X_real_test.loc[valid_real_test]
        y_real_test = y_real_test.loc[valid_real_test]
        X_synth_train = X_synth_train.loc[valid_synth_train]
        y_synth_train = y_synth_train.loc[valid_synth_train]

        if len(y_real_train) < 100 or len(y_real_test) < 50 or len(y_synth_train) < 100:
            result["status"] = "skip_too_few_valid_reg_samples"
            return result

        models = self._make_regression_models()
        rows = []
        for model_name, model in models.items():
            try:
                model_real = model.fit(X_real_train, y_real_train)
                pred_real = model_real.predict(X_real_test)

                model_synth = model.fit(X_synth_train, y_synth_train)
                pred_synth = model_synth.predict(X_real_test)

                r2_real = r2_score(y_real_test, pred_real)
                r2_synth = r2_score(y_real_test, pred_synth)
                mae_real = mean_absolute_error(y_real_test, pred_real)
                mae_synth = mean_absolute_error(y_real_test, pred_synth)

                utility_score = np.nan
                if np.isfinite(r2_real) and r2_real > 0:
                    utility_score = 100.0 * r2_synth / r2_real

                rows.append({
                    "model": model_name, "primary_metric": "r2",
                    "real_score": r2_real, "synth_score": r2_synth,
                    "utility_score_pct": utility_score,
                    "real_mae": mae_real, "synth_mae": mae_synth,
                    "model_status": "ok",
                })
            except Exception as e:
                rows.append({"model": model_name, "model_status": f"error: {e}"})

        rows_df = pd.DataFrame(rows)
        ok_rows = rows_df[rows_df["model_status"] == "ok"].copy()
        if len(ok_rows) == 0:
            result["status"] = "error_all_models_failed"
            result["details"] = rows_df.to_dict(orient="records")
            return result

        best_row = ok_rows.sort_values("synth_score", ascending=False, na_position="last").iloc[0].to_dict()
        result.update({
            "details": rows_df.to_dict(orient="records"),
            "best_model": best_row["model"],
            "primary_metric": best_row["primary_metric"],
            "best_real_score": float(best_row["real_score"]),
            "best_synth_score": float(best_row["synth_score"]),
            "best_utility_score_pct": float(best_row["utility_score_pct"]) if pd.notna(best_row["utility_score_pct"]) else np.nan,
        })
        return result

    def analyze_fidelity(self):
        print("\n--- 1. 开始数据保真度 (Fidelity) 分析 ---")
        save_dir = os.path.join(self.output_dir, "1_fidelity")
        self.plot_feature_distributions(save_dir)
        self.plot_categorical_distributions(save_dir)
        self.plot_missingness_analysis(save_dir)
        self.plot_statistical_moments(save_dir)
        self.plot_correlation_analysis(save_dir)
        self.plot_dimensionality_reduction_analysis(save_dir)
        self.plot_temporal_trajectory_analysis(save_dir)
        self.generate_summary_report(save_dir)
        print("✅ 数据保真度分析完成。")

    def plot_feature_distributions(self, save_dir):
        print("  -> 正在生成数值特征分布图...")
        if not self.numeric_cols:
            return
        n_features = len(self.numeric_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = np.array(axes).ravel()
        for i, col in enumerate(self.numeric_cols):
            real_x = self.real_df[col].dropna().to_numpy()
            synth_x = self.synthetic_df[col].dropna().to_numpy()
            ax = axes[i]
            try:
                _, p_value = ks_2samp(real_x, synth_x)
                sns.kdeplot(real_x, ax=ax, label="Real", color="blue", fill=True, alpha=0.35)
                sns.kdeplot(synth_x, ax=ax, label="Synthetic", color="red", fill=True, alpha=0.35)
                ax.set_title(f"{col}\nKS p={p_value:.3e}")
                ax.legend()
            except Exception as e:
                ax.set_title(f"{col}\nplot failed: {e}")
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        self._plot_and_save(fig, os.path.join(save_dir, "feature_distributions_numeric.png"))

    def plot_categorical_distributions(self, save_dir):
        print("  -> 正在生成类别特征分布图...")
        if not self.categorical_cols:
            return
        out_dir = os.path.join(save_dir, "categorical_distributions")
        os.makedirs(out_dir, exist_ok=True)
        for col in self.categorical_cols:
            try:
                real_dist = self.real_df[col].astype("object").fillna("__NAN__").value_counts(normalize=True)
                synth_dist = self.synthetic_df[col].astype("object").fillna("__NAN__").value_counts(normalize=True)
                compare = pd.concat([real_dist, synth_dist], axis=1).fillna(0.0)
                compare.columns = ["Real", "Synthetic"]
                compare = compare.sort_values("Real", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(10, 4))
                compare.plot(kind="bar", ax=ax)
                ax.set_title(f"Categorical Distribution: {col}")
                ax.set_ylabel("Probability")
                self._plot_and_save(fig, os.path.join(out_dir, f"{col}.png"))
            except Exception as e:
                print(f"    - 绘制类别分布 {col} 出错: {e}")

    def plot_missingness_analysis(self, save_dir):
        print("  -> 正在分析缺失率...")
        miss_real = self.real_df.isna().mean()
        miss_synth = self.synthetic_df.isna().mean()
        miss_df = pd.DataFrame({"Real": miss_real, "Synthetic": miss_synth})
        miss_df["AbsDiff"] = (miss_df["Real"] - miss_df["Synthetic"]).abs()
        miss_df.to_csv(os.path.join(save_dir, "missingness_summary.csv"))
        fig, ax = plt.subplots(figsize=(12, 6))
        miss_df[["Real", "Synthetic"]].sort_values("Real", ascending=False).head(40).plot(kind="bar", ax=ax)
        ax.set_title("Missingness Comparison (Top 40 by Real Missing Rate)")
        ax.set_ylabel("Missing Rate")
        self._plot_and_save(fig, os.path.join(save_dir, "missingness_comparison.png"))

    def plot_statistical_moments(self, save_dir):
        print("  -> 正在分析统计矩...")
        if not self.numeric_cols:
            return
        real_num = self.real_df[self.numeric_cols]
        synth_num = self.synthetic_df[self.numeric_cols]
        moments = {
            "mean": (real_num.mean(axis=0), synth_num.mean(axis=0)),
            "std": (real_num.std(axis=0), synth_num.std(axis=0)),
            "skew": (real_num.skew(axis=0), synth_num.skew(axis=0)),
            "kurtosis": (real_num.kurtosis(axis=0), synth_num.kurtosis(axis=0)),
        }
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        for i, (moment_name, (real_m, synth_m)) in enumerate(moments.items()):
            real_v = real_m.to_numpy(dtype=float)
            synth_v = synth_m.to_numpy(dtype=float)
            mask = np.isfinite(real_v) & np.isfinite(synth_v)
            real_v = real_v[mask]
            synth_v = synth_v[mask]
            axes[i].scatter(real_v, synth_v, alpha=0.6)
            if len(real_v) > 1:
                min_val = min(real_v.min(), synth_v.min())
                max_val = max(real_v.max(), synth_v.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], "r--")
                r = np.corrcoef(real_v, synth_v)[0, 1]
                r2 = r ** 2 if np.isfinite(r) else np.nan
            else:
                r2 = np.nan
            axes[i].set_title(f"{moment_name.capitalize()} Comparison\nR²={r2:.3f}")
            axes[i].set_xlabel("Real")
            axes[i].set_ylabel("Synthetic")
        self._plot_and_save(fig, os.path.join(save_dir, "statistical_moments.png"))

    def plot_correlation_analysis(self, save_dir):
        print("  -> 正在进行相关性分析...")
        if len(self.numeric_cols) < 2:
            return
        real_corr = self.real_df[self.numeric_cols].corr().to_numpy()
        synth_corr = self.synthetic_df[self.numeric_cols].corr().to_numpy()
        diff_corr = np.abs(real_corr - synth_corr)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", center=0, vmin=-1, vmax=1)
        axes[0].set_title("Real Correlations")
        sns.heatmap(synth_corr, ax=axes[1], cmap="coolwarm", center=0, vmin=-1, vmax=1)
        axes[1].set_title("Synthetic Correlations")
        sns.heatmap(diff_corr, ax=axes[2], cmap="YlOrRd", vmin=0)
        axes[2].set_title(f"Abs Corr Difference\nMean Diff={np.mean(diff_corr):.4f}")
        self._plot_and_save(fig, os.path.join(save_dir, "correlation_analysis.png"))

    def plot_dimensionality_reduction_analysis(self, save_dir):
        print("  -> 正在运行降维分析 (PCA & t-SNE)...")
        x_real = self._numeric_subject_repr(self.real_df)
        x_synth = self._numeric_subject_repr(self.synthetic_df)
        n_real = len(x_real)
        n_synth = len(x_synth)
        combined = np.vstack([x_real, x_synth])
        labels = np.array(["Real"] * n_real + ["Synthetic"] * n_synth)

        pca = PCA(n_components=min(3, combined.shape[1]))
        pca_result = pca.fit_transform(combined)
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        for label, color in zip(["Real", "Synthetic"], ["blue", "red"]):
            mask = labels == label
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], label=label, alpha=0.45, color=color, s=10)
        ax1.set_title("PCA (2D)")
        ax1.legend()
        self._plot_and_save(fig, os.path.join(save_dir, "pca_analysis.png"))

        x_real_s = x_real[: min(len(x_real), self.MAX_TSNE_SAMPLES // 2)]
        x_synth_s = x_synth[: min(len(x_synth), self.MAX_TSNE_SAMPLES // 2)]
        x_tsne = np.vstack([x_real_s, x_synth_s])
        y_tsne = np.array(["Real"] * len(x_real_s) + ["Synthetic"] * len(x_synth_s))
        if len(x_tsne) >= 100:
            tsne = TSNE(n_components=2, perplexity=30, random_state=self.RANDOM_STATE, init="pca")
            z = tsne.fit_transform(x_tsne)
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, color in zip(["Real", "Synthetic"], ["blue", "red"]):
                mask = y_tsne == label
                ax.scatter(z[mask, 0], z[mask, 1], label=label, alpha=0.5, color=color, s=10)
            ax.set_title("t-SNE Visualization")
            ax.legend()
            self._plot_and_save(fig, os.path.join(save_dir, "tsne_analysis.png"))

    def plot_temporal_trajectory_analysis(self, save_dir):
        print("  -> 正在分析 temporal trajectory...")
        temp_num = [c for c in self.temporal_num_cols if c in self.real_df.columns]
        if not temp_num:
            temp_num = self.numeric_cols[: min(4, len(self.numeric_cols))]
        out_dir = os.path.join(save_dir, "temporal_trajectories")
        os.makedirs(out_dir, exist_ok=True)
        for col in temp_num[:4]:
            try:
                real_curve = self.real_df.groupby(self.seq_col)[col].mean()
                synth_curve = self.synthetic_df.groupby(self.seq_col)[col].mean()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(real_curve.index, real_curve.values, label="Real", marker="o")
                ax.plot(synth_curve.index, synth_curve.values, label="Synthetic", marker="o")
                ax.set_title(f"Trajectory Mean by {self.seq_col}: {col}")
                ax.set_xlabel(self.seq_col)
                ax.set_ylabel(col)
                ax.legend()
                self._plot_and_save(fig, os.path.join(out_dir, f"{col}.png"))
            except Exception as e:
                print(f"    - temporal trajectory {col} 出错: {e}")

    def generate_summary_report(self, save_dir):
        print("  -> 正在生成保真度总结报告...")
        if not self.numeric_cols:
            with open(os.path.join(save_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
                f.write(self.report_content + "\nNo numeric columns available.\n")
            return
        real_num = self.real_df[self.numeric_cols]
        synth_num = self.synthetic_df[self.numeric_cols]
        stats_df = pd.DataFrame(index=self.numeric_cols)
        stats_df["Real_Mean"] = real_num.mean(axis=0)
        stats_df["Synthetic_Mean"] = synth_num.mean(axis=0)
        stats_df["Real_Std"] = real_num.std(axis=0)
        stats_df["Synthetic_Std"] = synth_num.std(axis=0)
        ks_stats, ks_pvals = [], []
        for col in self.numeric_cols:
            x = self.real_df[col].dropna().to_numpy()
            y = self.synthetic_df[col].dropna().to_numpy()
            ks_stat, p_val = ks_2samp(x, y)
            ks_stats.append(ks_stat)
            ks_pvals.append(p_val)
        stats_df["KS_Statistic"] = ks_stats
        stats_df["KS_P_Value"] = ks_pvals
        diff_stats = {
            "Mean_Abs_Diff_Mean": float(np.mean(np.abs(stats_df["Real_Mean"] - stats_df["Synthetic_Mean"]))),
            "Mean_Abs_Diff_Std": float(np.mean(np.abs(stats_df["Real_Std"] - stats_df["Synthetic_Std"]))),
            "Mean_KS_Stat": float(np.mean(ks_stats)),
            "Mean_KS_P_Value": float(np.mean(ks_pvals)),
        }
        report = self.report_content
        report += "--- Numeric Feature-wise Statistics ---\n"
        report += stats_df.to_string()
        report += "\n\n--- Overall Statistical Differences ---\n"
        for metric, value in diff_stats.items():
            report += f"{metric}: {value:.6f}\n"
        with open(os.path.join(save_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(save_dir, "summary_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(diff_stats, f, ensure_ascii=False, indent=2)

    def analyze_utility(self):
        print("\n--- 2. 开始数据可用性 (Utility) 分析 ---")
        save_dir = os.path.join(self.output_dir, "2_utility")
        self._run_table_tstr_evaluation(save_dir)
        print("✅ 数据可用性分析完成。")

    def _run_table_tstr_evaluation(self, save_dir):
        print("  -> 正在运行表空间 TSTR 评估...")
        if self.UTILITY_TARGET_COL is not None:
            target_cols = [self.UTILITY_TARGET_COL]
            auto_mode = False
        elif self.AUTO_UTILITY_SCAN:
            target_cols = self._get_utility_candidate_cols()
            auto_mode = True
        else:
            target_cols = []
            auto_mode = False

        if not target_cols:
            report = (
                "--- Table-space Utility (TSTR) Report ---\n\n"
                "未找到可评估的目标列。\n"
                "如果你有明确目标列，请设置 UTILITY_TARGET_COL；否则将 AUTO_UTILITY_SCAN=True。\n"
            )
            print(report)
            with open(os.path.join(save_dir, "utility_tstr_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
            return

        print(f"    - 目标列数量: {len(target_cols)}")
        all_results = []
        for target_col in tqdm(target_cols, desc="  -> Utility tasks"):
            try:
                all_results.append(self._run_single_utility_task(target_col))
            except Exception as e:
                all_results.append({"target_col": target_col, "status": f"error: {e}"})

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(save_dir, "utility_per_column.csv"), index=False)

        ok_df = results_df[results_df["status"] == "ok"].copy()
        report = "--- Table-space Utility (TSTR) Report ---\n\n"
        report += f"Auto scan mode: {auto_mode}\n"
        report += f"Num targets total: {len(results_df)}\n"
        report += f"Num targets success: {len(ok_df)}\n\n"

        if len(ok_df) == 0:
            report += "没有成功完成的 utility 任务。\n"
            print(report)
            with open(os.path.join(save_dir, "utility_tstr_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
            return

        top_df = ok_df.sort_values("best_utility_score_pct", ascending=False, na_position="last").head(self.UTILITY_REPORT_TOPK)
        show_cols = ["target_col", "task_type", "best_model", "primary_metric", "best_real_score", "best_synth_score", "best_utility_score_pct"]
        report += "--- Top Utility Targets ---\n"
        report += top_df[show_cols].to_string(index=False)
        report += "\n\nInterpretation:\n"
        report += " - utility_score_pct 越接近 100%，说明 synthetic 训练在该目标列上的效果越接近 real 训练。\n"
        report += " - classification: primary metric 为 AUROC 或 macro-F1。\n"
        report += " - regression: primary metric 为 R²；同时请结合 MAE。\n"

        print(report)
        with open(os.path.join(save_dir, "utility_tstr_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        fig, ax = plt.subplots(figsize=(12, max(6, len(top_df) * 0.35)))
        plot_df = top_df.sort_values("best_utility_score_pct", ascending=True)
        ax.barh(plot_df["target_col"], plot_df["best_utility_score_pct"])
        ax.set_xlabel("Utility Score (%)")
        ax.set_title("Top Utility Targets")
        self._plot_and_save(fig, os.path.join(save_dir, "utility_top_targets.png"))

    def analyze_privacy(self):
        print("\n--- 3. 开始隐私保护 (Privacy) 分析 ---")
        save_dir = os.path.join(self.output_dir, "3_privacy")
        real_train_repr = self._numeric_subject_repr(self.real_train_df)
        real_holdout_repr = self._numeric_subject_repr(self.real_holdout_df)
        synth_repr = self._numeric_subject_repr(self.synthetic_df)
        real_train_repr = real_train_repr[: min(len(real_train_repr), self.MAX_PRIVACY_SAMPLES)]
        real_holdout_repr = real_holdout_repr[: min(len(real_holdout_repr), self.MAX_PRIVACY_SAMPLES)]
        synth_repr = synth_repr[: min(len(synth_repr), self.MAX_PRIVACY_SAMPLES)]
        self._run_membership_inference_attack(real_train_repr, synth_repr, save_dir)
        self._run_attribute_inference_attack(save_dir)
        self._run_reidentification_attack(real_train_repr, real_holdout_repr, synth_repr, save_dir)
        print("✅ 隐私保护分析完成。")

    def _run_membership_inference_attack(self, real_train_repr, synth_repr, save_dir):
        print("  -> 正在进行成员推断攻击 (MIA) 模拟...")
        try:
            n = min(len(real_train_repr), len(synth_repr))
            X = np.concatenate([real_train_repr[:n], synth_repr[:n]], axis=0)
            y = np.concatenate([np.ones(n), np.zeros(n)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.RANDOM_STATE, stratify=y)
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=2000, random_state=self.RANDOM_STATE)),
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            report = "\n--- Membership Inference Attack (Table-space) ---\n\n"
            report += f"Attack model accuracy: {acc:.6f}\n"
            report += f"Attack model AUC: {auc:.6f}\n\n"
            report += "Interpretation: 越接近 0.5 越好。\n"
            print(report)
            with open(os.path.join(save_dir, "privacy_mia_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
        except Exception as e:
            print(f"    - MIA 出错: {e}")

    def _run_attribute_inference_attack(self, save_dir):
        print("  -> 正在进行属性推断攻击 (subject-level)...")
        sensitive_candidates = [c for c in self.static_cat_cols if c in self.real_df.columns]
        if not sensitive_candidates:
            report = "\n--- Attribute Inference Attack ---\n\n未找到静态类别敏感列，跳过属性推断攻击。\n"
            print(report)
            with open(os.path.join(save_dir, "privacy_attribute_inference_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
            return
        try:
            real_train_subj = self._get_subject_level_repr(self.real_train_df)
            synth_subj = self._get_subject_level_repr(self.synthetic_df)
            results = {}
            for target_col in tqdm(sensitive_candidates, desc="    - 攻击进度"):
                if target_col not in real_train_subj.columns or target_col not in synth_subj.columns:
                    continue
                X_synth = synth_subj.drop(columns=[self.subject_col, target_col], errors="ignore")
                y_synth = synth_subj[target_col]
                X_real = real_train_subj.drop(columns=[self.subject_col, target_col], errors="ignore")
                y_real = real_train_subj[target_col]
                if pd.Series(y_real).dropna().nunique() < 2 or pd.Series(y_synth).dropna().nunique() < 2:
                    continue
                X_synth = pd.get_dummies(X_synth, dummy_na=True)
                X_real = pd.get_dummies(X_real, dummy_na=True)
                X_synth, X_real = X_synth.align(X_real, join="outer", axis=1, fill_value=0)
                valid_real = ~pd.isna(y_real)
                valid_synth = ~pd.isna(y_synth)
                y_synth_codes = pd.Series(y_synth[valid_synth]).astype("category").cat.codes
                y_real_codes = pd.Series(y_real[valid_real]).astype("category").cat.codes
                base_model = RandomForestClassifier(random_state=self.RANDOM_STATE) if not HAS_LIGHTGBM else lgb.LGBMClassifier(random_state=self.RANDOM_STATE, verbose=-1)
                model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base_model)])
                model.fit(X_synth.loc[valid_synth], y_synth_codes)
                pred = model.predict(X_real.loc[valid_real])
                results[target_col] = accuracy_score(y_real_codes, pred)
            if not results:
                report = "\n--- Attribute Inference Attack ---\n\n没有可用的敏感类别列可用于攻击评估。\n"
            else:
                results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Attack Accuracy"])
                mean_acc = results_df["Attack Accuracy"].mean()
                report = "\n--- Attribute Inference Attack ---\n\n"
                report += results_df.to_string()
                report += f"\n\nMean attack accuracy: {mean_acc:.6f}\n"
                fig, ax = plt.subplots(figsize=(12, max(6, len(results_df) * 0.4)))
                results_df.sort_values("Attack Accuracy").plot(kind="barh", ax=ax)
                ax.axvline(x=0.5, color="r", linestyle="--", label="Random Guess")
                ax.set_xlabel("Attack Accuracy")
                ax.legend()
                self._plot_and_save(fig, os.path.join(save_dir, "privacy_attribute_inference_plot.png"))
            print(report)
            with open(os.path.join(save_dir, "privacy_attribute_inference_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
        except Exception as e:
            print(f"    - 属性推断攻击出错: {e}")

    def _calculate_min_dists_in_batches(self, source_data, target_data, batch_size=1024):
        print(f"    - 正在分块计算 {source_data.shape[0]} 个样本的最近邻距离...")
        source_data = source_data.astype(np.float32)
        target_data = target_data.astype(np.float32)
        all_min_dists = []
        num_batches = int(np.ceil(source_data.shape[0] / batch_size))
        for i in tqdm(range(num_batches), desc="  -> 计算距离"):
            s = i * batch_size
            e = min((i + 1) * batch_size, source_data.shape[0])
            batch = source_data[s:e]
            dist_chunk = cdist(batch, target_data, metric="euclidean")
            all_min_dists.append(dist_chunk.min(axis=1))
        return np.concatenate(all_min_dists)

    def _run_reidentification_attack(self, real_train_repr, real_holdout_repr, synth_repr, save_dir):
        print("  -> 正在进行重识别攻击 (最近邻) 模拟...")
        try:
            dists_train = self._calculate_min_dists_in_batches(real_train_repr, synth_repr)
            dists_holdout = self._calculate_min_dists_in_batches(real_holdout_repr, synth_repr)
            all_labels = np.concatenate([np.ones_like(dists_train), np.zeros_like(dists_holdout)])
            all_scores = -np.concatenate([dists_train, dists_holdout])
            auc_score = roc_auc_score(all_labels, all_scores)
            report = "\n--- Re-identification Attack Report ---\n\n"
            report += f"AUC: {auc_score:.6f}\n\n"
            report += "Interpretation: 越接近 0.5 越好。\n"
            print(report)
            with open(os.path.join(save_dir, "privacy_reidentification_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
        except Exception as e:
            print(f"    - 重识别攻击出错: {e}")

    def run_all_analyses(self):
        self.analyze_fidelity()
        self.analyze_utility()
        self.analyze_privacy()
        print(f"\n--- 🚀 表空间黄金标准评估完成！报告目录: {self.output_dir}/ ---")


def main():
    comparator = GoldStandardComparator()
    comparator.run_all_analyses()


if __name__ == "__main__":
    main()
