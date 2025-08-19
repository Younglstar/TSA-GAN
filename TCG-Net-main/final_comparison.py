# file: final_comprehensive_analysis.py (增强版)

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import ks_2samp
import os
from tqdm import tqdm
import lightgbm as lgb  # 新增：用于TSTR分析的强大模型
from statsmodels.graphics.tsaplots import plot_acf  # 新增：用于自相关性分析

# --- 导入我们所有更新后的配置文件 ---
# 假设这些配置文件与此脚本在同一目录下
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig


class ComprehensiveComparator:
    def __init__(self):
        """加载所有必需的数据和配置。"""
        # --- 初始化配置和目录 ---
        self.encdec_conf = EncoderDecoderConfig()
        self.gan_conf = GANConfig()
        self.output_dir = 'final_comparison_report'
        os.makedirs(self.output_dir, exist_ok=True)

        # 新增：用于存储总结报告的文本内容
        self.report_content = "--- Synthetic Data Quality Report ---\n\n"

        print("📊 正在加载所有分析所需的数据...")
        try:
            # 加载潜在空间数据
            with open(self.gan_conf.SYNTHETIC_DATA_PATH, 'rb') as f:
                self.synthetic_encoded = pickle.load(f)
            with open(self.encdec_conf.ENCODED_DATA_PATH, 'rb') as f:
                self.real_encoded = pickle.load(f)

            self.n_real = len(self.real_encoded)
            self.n_synthetic = len(self.synthetic_encoded)
            self.n_features = self.real_encoded.shape[1]
            print("数据加载完成！")
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}")
            print("  请确保您已按顺序成功运行了所有训练脚本。")
            raise

    # ==============================================================================
    # == 新增方法: 下游任务可用性评估 (TSTR) ==
    # ==============================================================================
    def run_tstr_analysis(self):
        """
        运行“训练合成，测试真实”(Train-Synthetic, Test-Real)分析。
        这是一个衡量合成数据实用性的黄金标准。
        我们将尝试用其他特征预测第一个特征，作为一个代理任务。
        """
        print("  -> 正在运行下游任务可用性分析 (TSTR)...")
        try:
            # 准备数据：X为所有特征，y为第一个特征
            X_real, y_real = self.real_encoded[:, 1:], self.real_encoded[:, 0]
            X_synth, y_synth = self.synthetic_encoded[:, 1:], self.synthetic_encoded[:, 0]

            # 从真实数据中划分出一个独立的测试集，这是我们评估的基准
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=0.3, random_state=42
            )

            # --- 场景1: 在真实数据上训练，在真实数据上测试 (性能基准) ---
            model_on_real = lgb.LGBMRegressor(random_state=42)
            model_on_real.fit(X_real_train, y_real_train)
            preds_real = model_on_real.predict(X_real_test)
            score_real = r2_score(y_real_test, preds_real)

            # --- 场景2: 在合成数据上训练，在真实数据上测试 (评估合成数据) ---
            model_on_synth = lgb.LGBMRegressor(random_state=42)
            model_on_synth.fit(X_synth, y_synth)
            preds_synth = model_on_synth.predict(X_real_test)
            score_synth = r2_score(y_real_test, preds_synth)

            # --- 可视化与报告 ---
            plt.figure(figsize=(8, 6))
            bars = plt.bar(['Train on Real (Benchmark)', 'Train on Synthetic'], [score_real, score_synth],
                           color=['blue', 'red'])
            plt.ylabel('R² Score on Real Test Set')
            plt.title('Train-Synthetic, Test-Real (TSTR) Utility Score')
            plt.ylim(min(0, score_real, score_synth) - 0.1, max(score_real, score_synth) + 0.1)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tstr_analysis.png'))
            plt.close()

            # 添加到总结报告
            self.report_content += "--- Downstream Task Utility (TSTR) ---\n"
            self.report_content += f"Benchmark Score (Train on Real): {score_real:.4f}\n"
            self.report_content += f"Synthetic Data Score (Train on Synthetic): {score_synth:.4f}\n"
            self.report_content += f"Utility Score (Synthetic / Real): {(score_synth / score_real) * 100 if score_real > 0 else 0:.2f}%\n\n"

        except Exception as e:
            print(f"    - TSTR 分析时出错: {e}")
            plt.close()

    # ==============================================================================
    # == 新增方法: 时序动态分析 (Autocorrelation) ==
    # ==============================================================================
    def plot_autocorrelation_analysis(self, n_features_to_plot=4, lags=40):
        """
        (新增) 比较真实数据和合成数据的自相关性，以评估时序动态。
        我们只选择前几个特征进行可视化。
        """
        print("  -> 正在分析时序自相关性...")
        try:
            n_plot = min(self.n_features, n_features_to_plot)
            fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot))
            if n_plot == 1: axes = [axes]

            for i in range(n_plot):
                # 绘制真实数据的ACF
                plot_acf(self.real_encoded[:, i], ax=axes[i], lags=lags, title=f'Autocorrelation for Feature {i + 1}',
                         label='Real ACF', color='blue', alpha=0.5)
                # 在同一子图上绘制合成数据的ACF
                plot_acf(self.synthetic_encoded[:, i], ax=axes[i], lags=lags, label='Synthetic ACF', color='red',
                         alpha=0.5)

                # 调整图例
                handles, labels = axes[i].get_legend_handles_labels()
                # 我们需要手动创建一个干净的图例
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color='blue', lw=4),
                                Line2D([0], [0], color='red', lw=4)]
                axes[i].legend(custom_lines, ['Real', 'Synthetic'])

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'autocorrelation_analysis.png'))
            plt.close()

        except Exception as e:
            print(f"    - 绘制自相关图时出错: {e}")
            plt.close()

    def plot_feature_distributions(self):
        """(源自第二个脚本) 绘制所有特征的分布图并进行统计检验"""
        print("  -> 正在生成特征分布图...")
        # ... (此方法的代码保持不变)
        try:
            n_cols = 4
            n_rows = (self.n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            if self.n_features == 1: axes = np.array([axes])
            axes = axes.ravel()
            for i in range(self.n_features):
                real_feature, synth_feature = self.real_encoded[:, i], self.synthetic_encoded[:, i]
                ks_stat, p_value = ks_2samp(real_feature, synth_feature)
                sns.kdeplot(real_feature, ax=axes[i], label='Real', color='blue', fill=True, alpha=0.5)
                sns.kdeplot(synth_feature, ax=axes[i], label='Synthetic', color='red', fill=True, alpha=0.5)
                axes[i].set_title(f'Feature {i + 1}\nKS p-value: {p_value:.3e}')
                axes[i].legend()
            for i in range(self.n_features, len(axes)): fig.delaxes(axes[i])
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'))
            plt.close()
        except Exception as e:
            print(f"    - 绘制特征分布图时出错: {e}")
            plt.close()

    def plot_statistical_moments(self):
        """(源自第二个脚本) 比较统计矩，并计算R²值"""
        print("  -> 正在分析统计矩...")
        # ... (此方法的代码保持不变)
        try:
            moments = {'mean': (np.mean(self.real_encoded, axis=0), np.mean(self.synthetic_encoded, axis=0)),
                       'std': (np.std(self.real_encoded, axis=0), np.std(self.synthetic_encoded, axis=0)),
                       'skew': (stats.skew(self.real_encoded, axis=0), stats.skew(self.synthetic_encoded, axis=0)),
                       'kurtosis': (
                           stats.kurtosis(self.real_encoded, axis=0), stats.kurtosis(self.synthetic_encoded, axis=0))}
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.ravel()
            for i, (moment_name, (real_moment, synth_moment)) in enumerate(moments.items()):
                axes[i].scatter(real_moment, synth_moment, alpha=0.5)
                min_val, max_val = min(np.nanmin(real_moment), np.nanmin(synth_moment)), max(np.nanmax(real_moment),
                                                                                             np.nanmax(synth_moment))
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
                r2 = np.corrcoef(real_moment, synth_moment)[0, 1] ** 2
                axes[i].set_title(f'{moment_name.capitalize()} Comparison\nR² = {r2:.3f}')
                axes[i].set_xlabel('Real Data');
                axes[i].set_ylabel('Synthetic Data')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_moments.png'))
            plt.close()
        except Exception as e:
            print(f"    - 绘制统计矩图时出错: {e}")
            plt.close()

    def plot_correlation_analysis(self):
        """(源自第二个脚本) 详细的相关性分析，并计算R²值"""
        print("  -> 正在进行相关性分析...")
        # ... (此方法的代码保持不变)
        try:
            real_corr, synth_corr = np.corrcoef(self.real_encoded.T), np.corrcoef(self.synthetic_encoded.T)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
            sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1).set_title(
                'Real Data Correlations')
            sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1).set_title(
                'Synthetic Data Correlations')
            diff_corr = np.abs(real_corr - synth_corr)
            sns.heatmap(diff_corr, ax=ax3, cmap='YlOrRd', vmin=0, vmax=1).set_title(
                f'Absolute Correlation Differences\nMean Diff: {np.mean(diff_corr):.3f}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_analysis.png'))
            plt.close()

            plt.figure(figsize=(8, 8))
            triu_indices = np.triu_indices(self.n_features, k=1)
            real_corr_triu, synth_corr_triu = real_corr[triu_indices], synth_corr[triu_indices]
            plt.scatter(real_corr_triu, synth_corr_triu, alpha=0.5)
            min_val, max_val = min(real_corr_triu.min(), synth_corr_triu.min()), max(real_corr_triu.max(),
                                                                                     synth_corr_triu.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            r2 = np.corrcoef(real_corr_triu, synth_corr_triu)[0, 1] ** 2
            plt.xlabel('Real Data Correlations');
            plt.ylabel('Synthetic Data Correlations')
            plt.title(f'Correlation Preservation\nR² = {r2:.3f}')
            plt.savefig(os.path.join(self.output_dir, 'correlation_preservation.png'))
            plt.close()
        except Exception as e:
            print(f"    - 进行相关性分析时出错: {e}")
            plt.close()

    def plot_dimensionality_reduction_analysis(self):
        """(源自第一个脚本) 增强的降维分析 (PCA & t-SNE)"""
        print("  -> 正在运行降维分析 (PCA & t-SNE)...")
        # ... (此方法的代码保持不变)
        try:
            combined_data = np.vstack([self.real_encoded, self.synthetic_encoded])
            labels = ['Real'] * self.n_real + ['Synthetic'] * self.n_synthetic

            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(combined_data)

            # 优化：对于大数据集，TSNE可能很慢，这里可以考虑采样
            sample_indices = np.random.permutation(len(combined_data))[:1000]  # 最多使用1000个点
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            tsne_result = tsne.fit_transform(combined_data[sample_indices])

            # 绘制 PCA
            fig = plt.figure(figsize=(16, 6))
            ax1 = fig.add_subplot(121)
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = np.array(labels) == label
                ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], label=label, alpha=0.5, color=color)
            ax1.set_title('PCA (2D)');
            ax1.legend()
            ax2 = fig.add_subplot(122, projection='3d')
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = np.array(labels) == label
                ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2], label=label, alpha=0.5,
                            color=color)
            ax2.set_title('PCA (3D)');
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pca_analysis.png'))
            plt.close()

            # 绘制 t-SNE (使用采样后的数据)
            plt.figure(figsize=(8, 6))
            sampled_labels = np.array(labels)[sample_indices]
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = sampled_labels == label
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=label, alpha=0.5, color=color)
            plt.title('t-SNE Visualization');
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tsne_analysis.png'))
            plt.close()
        except Exception as e:
            print(f"    - 进行降维分析时出错: {e}")
            plt.close()

    def generate_summary_report(self):
        """(源自第二个脚本) 生成并保存详细的总结报告"""
        print("  -> 正在生成总结报告...")
        try:
            stats_dict = {'Real_Mean': np.mean(self.real_encoded, axis=0),
                          'Synthetic_Mean': np.mean(self.synthetic_encoded, axis=0),
                          'Real_Std': np.std(self.real_encoded, axis=0),
                          'Synthetic_Std': np.std(self.synthetic_encoded, axis=0),
                          'Real_Skew': stats.skew(self.real_encoded, axis=0),
                          'Synthetic_Skew': stats.skew(self.synthetic_encoded, axis=0),
                          'Real_Kurtosis': stats.kurtosis(self.real_encoded, axis=0),
                          'Synthetic_Kurtosis': stats.kurtosis(self.synthetic_encoded, axis=0)}
            ks_stats, ks_pvals = [], []
            for i in range(self.n_features):
                ks_stat, p_val = ks_2samp(self.real_encoded[:, i], self.synthetic_encoded[:, i])
                ks_stats.append(ks_stat)
                ks_pvals.append(p_val)
            stats_dict['KS_Statistic'], stats_dict['KS_P_Value'] = ks_stats, ks_pvals
            stats_df = pd.DataFrame(stats_dict)

            diff_stats = {'Mean_Abs_Diff_Mean': np.mean(np.abs(stats_dict['Real_Mean'] - stats_dict['Synthetic_Mean'])),
                          'Mean_Abs_Diff_Std': np.mean(np.abs(stats_dict['Real_Std'] - stats_dict['Synthetic_Std'])),
                          'Mean_Abs_Diff_Skew': np.mean(np.abs(stats_dict['Real_Skew'] - stats_dict['Synthetic_Skew'])),
                          'Mean_Abs_Diff_Kurtosis': np.mean(
                              np.abs(stats_dict['Real_Kurtosis'] - stats_dict['Synthetic_Kurtosis'])),
                          'Mean_KS_Stat': np.mean(ks_stats), 'Mean_KS_P_Value': np.mean(ks_pvals)}

            self.report_content += "--- Feature-wise Statistics ---\n"
            self.report_content += stats_df.to_string()
            self.report_content += "\n\n--- Overall Statistical Differences ---\n"
            for metric, value in diff_stats.items():
                self.report_content += f"{metric}: {value:.4f}\n"

            # 最终将所有报告内容写入文件
            with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
                f.write(self.report_content)

            # 保存一个机器可读的pkl文件
            with open(os.path.join(self.output_dir, 'summary_statistics.pkl'), 'wb') as f:
                pickle.dump({'feature_stats': stats_df, 'overall_diffs': diff_stats}, f)
        except Exception as e:
            print(f"    - 生成总结报告时出错: {e}")

    def run_all_analyses(self):
        """运行所有对比分析"""
        print("\n--- 开始最终数据对比分析 ---")
        # 1. 首先评估下游任务可用性 (最重要的新指标)
        self.run_tstr_analysis()
        # 2. 然后评估时序特性
        self.plot_autocorrelation_analysis()
        # 3. 最后运行所有统计保真度分析
        self.plot_feature_distributions()
        self.plot_statistical_moments()
        self.plot_correlation_analysis()
        self.plot_dimensionality_reduction_analysis()
        # 4. 生成包含所有信息的最终报告
        self.generate_summary_report()
        print(f"\n--- ✅ 分析完成！所有结果已保存至: {self.output_dir}/ ---")


def main():
    try:
        comparator = ComprehensiveComparator()
        comparator.run_all_analyses()
    except Exception as e:
        print(f"主程序执行时出错: {e}")


if __name__ == "__main__":
    main()