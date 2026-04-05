#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置画图风格
sns.set_theme(style="whitegrid")


def plot_positive_distributions(real_csv, synth_csv, target_cols, out_dir, positive_threshold=1e-8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在极速读取真实数据 (仅加载目标列): {real_csv}")
    # 针对宽表的内存优化：只读取 target_cols
    real_df = pd.read_csv(real_csv, engine="pyarrow", usecols=target_cols)

    print(f"🚀 正在极速读取合成数据 (仅加载目标列): {synth_csv}")
    synth_df = pd.read_csv(synth_csv, engine="pyarrow", usecols=target_cols)

    for col in target_cols:
        print(f"\n🎨 正在绘制特征: {col} ...")
        if col not in real_df.columns or col not in synth_df.columns:
            print(f"⚠️ 警告: 列 {col} 不在数据集中，已跳过。")
            continue

        # 核心逻辑：只过滤出大于阈值（正样本）的值，因为你的模型在这个阶段崩了
        real_pos = real_df[col][real_df[col] > positive_threshold].dropna()
        synth_pos = synth_df[col][synth_df[col] > positive_threshold].dropna()

        if len(real_pos) == 0 or len(synth_pos) == 0:
            print(f"⚠️ 警告: {col} 没有足够的正样本进行绘图。")
            continue

        # 创建一个包含两个子图的画布 (1行2列)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Distribution Comparison (Positive Values > 0): {col}", fontsize=16)

        # 构建用于画图的 DataFrame
        plot_df = pd.DataFrame({
            'Value': pd.concat([real_pos, synth_pos]),
            'Dataset': ['Real'] * len(real_pos) + ['Synthetic'] * len(synth_pos)
        })

        # --- 图 1: 直方图与核密度估计 (KDE) ---
        sns.histplot(
            data=plot_df, x='Value', hue='Dataset',
            kde=True, stat="density", common_norm=False,
            ax=axes[0], palette={"Real": "#1f77b4", "Synthetic": "#ff7f0e"},
            alpha=0.5, bins=50
        )
        axes[0].set_title("Histogram & KDE (Shape Comparison)")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")

        # --- 图 2: 箱型图 (检查极端异常值) ---
        sns.boxplot(
            data=plot_df, x='Dataset', y='Value',
            ax=axes[1], palette={"Real": "#1f77b4", "Synthetic": "#ff7f0e"}
        )
        axes[1].set_title("Boxplot (Outlier Detection)")
        axes[1].set_ylabel("Value")

        # 调整布局并保存
        plt.tight_layout()
        save_path = out_dir / f"dist_{col}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 图片已保存至: {save_path}")

    print("\n🎉 所有分布图绘制完毕！快去输出文件夹看看吧。")


def main():
    parser = argparse.ArgumentParser(description="可视化真实数据与合成数据中连续变量的分布 (仅大于0的部分)")
    parser.add_argument("--real_csv", type=str, required=True, help="真实数据 CSV 路径")
    parser.add_argument("--synth_csv", type=str, required=True, help="选出的最佳合成数据 CSV 路径")
    parser.add_argument("--target_cols", nargs="+", default=["leg1v", "leg2v", "air1", "grid", "solar"],
                        help="需要对比的列名")
    parser.add_argument("--out_dir", type=str, default="./visualizations", help="图片保存目录")

    args = parser.parse_args()

    plot_positive_distributions(
        real_csv=args.real_csv,
        synth_csv=args.synth_csv,
        target_cols=args.target_cols,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()