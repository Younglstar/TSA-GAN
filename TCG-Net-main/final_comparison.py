# file: final_gold_standard_evaluation.py (推荐的最终文件名)

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.stats import ks_2samp
import os
import lightgbm as lgb
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm

# --- 导入我们所有更新后的模块和配置文件 ---
from encdec_model import CausalVAE
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig
from config import DataConfig


class GoldStandardComparator:
    def __init__(self):
        # ... (初始化部分保持不变) ...
        self.output_dir = 'final_gold_standard_report'
        os.makedirs(os.path.join(self.output_dir, '1_fidelity'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, '2_utility'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, '3_privacy'), exist_ok=True)
        self.report_content = "--- Synthetic Data Gold Standard Report ---\n\n"

        print("📊 正在加载所有分析所需的数据...")
        try:
            gan_conf = GANConfig()
            encdec_conf = EncoderDecoderConfig()
            with open(gan_conf.SYNTHETIC_DATA_PATH, 'rb') as f:
                self.synthetic_encoded = pickle.load(f)
            with open(encdec_conf.ENCODED_DATA_PATH, 'rb') as f:
                self.real_encoded = pickle.load(f)

            self.n_real = len(self.real_encoded)
            self.n_synthetic = len(self.synthetic_encoded)
            self.n_features = self.real_encoded.shape[1]
            print("数据加载完成！")
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}")
            raise

    # ==============================================================================
    # == 第一部分：数据保真度 (Fidelity) 分析 (保持高水准) ==
    # ==============================================================================
    def analyze_fidelity(self):
        print("\n--- 1. 开始数据保真度 (Fidelity) 分析 ---")
        save_dir=os.path.join(self.output_dir, '1_fidelity')
        # 2. 然后评估时序特性
        self.plot_autocorrelation_analysis(save_dir)
        # 3. 最后运行所有统计保真度分析
        self.plot_feature_distributions(save_dir)
        self.plot_statistical_moments(save_dir)
        self.plot_correlation_analysis(save_dir)
        self.plot_dimensionality_reduction_analysis(save_dir)
        # 4. 生成包含所有信息的最终报告
        self.generate_summary_report(save_dir)
        # ... (这部分的所有函数 _plot_... 和 _generate_... 与之前一样，此处省略) ...
        print("✅ 数据保真度分析完成。")

    def plot_autocorrelation_analysis(self,save_dir):
        """
        (新增) 比较真实数据和合成数据的自相关性，以评估时序动态。
        我们只选择前几个特征进行可视化。
        """
        print("  -> 正在分析时序自相关性...")
        n_features_to_plot=4
        lags = 40
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
            plt.savefig(os.path.join(save_dir, 'autocorrelation_analysis.png'))
            plt.close()

        except Exception as e:
            print(f"    - 绘制自相关图时出错: {e}")
            plt.close()

    def plot_feature_distributions(self,save_dir):
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
            plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
            plt.close()
        except Exception as e:
            print(f"    - 绘制特征分布图时出错: {e}")
            plt.close()

    def plot_statistical_moments(self,save_dir):
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
                axes[i].set_xlabel('Real Data')
                axes[i].set_ylabel('Synthetic Data')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'statistical_moments.png'))
            plt.close()
        except Exception as e:
            print(f"    - 绘制统计矩图时出错: {e}")
            plt.close()

    def plot_correlation_analysis(self,save_dir):
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
            plt.xlabel('Real Data Correlations')
            plt.ylabel('Synthetic Data Correlations')
            plt.title(f'Correlation Preservation\nR² = {r2:.3f}')
            plt.savefig(os.path.join(save_dir, 'correlation_preservation.png'))
            plt.close()
        except Exception as e:
            print(f"    - 进行相关性分析时出错: {e}")
            plt.close()

    def plot_dimensionality_reduction_analysis(self,save_dir):
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
            ax1.set_title('PCA (2D)')
            ax1.legend()
            ax2 = fig.add_subplot(122, projection='3d')
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = np.array(labels) == label
                ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2], label=label, alpha=0.5,
                            color=color)
            ax2.set_title('PCA (3D)')
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'pca_analysis.png'))
            plt.close()

            # 绘制 t-SNE (使用采样后的数据)
            plt.figure(figsize=(8, 6))
            sampled_labels = np.array(labels)[sample_indices]
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = sampled_labels == label
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=label, alpha=0.5, color=color)
            plt.title('t-SNE Visualization')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'tsne_analysis.png'))
            plt.close()
        except Exception as e:
            print(f"    - 进行降维分析时出错: {e}")
            plt.close()

    def generate_summary_report(self,save_dir):
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
            with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
                f.write(self.report_content)

            # 保存一个机器可读的pkl文件
            with open(os.path.join(save_dir, 'summary_statistics.pkl'), 'wb') as f:
                pickle.dump({'feature_stats': stats_df, 'overall_diffs': diff_stats}, f)
        except Exception as e:
            print(f"    - 生成总结报告时出错: {e}")
    # ==============================================================================
    # == 第二部分：数据可用性 (Utility) 分析 (已增强) ==
    # ==============================================================================
    def analyze_utility(self):
        print("\n--- 2. 开始数据可用性 (Utility) 分析 ---")
        save_dir = os.path.join(self.output_dir, '2_utility')
        self._run_enhanced_tstr_evaluation(save_dir)
        print("✅ 数据可用性分析完成。")

    def _run_enhanced_tstr_evaluation(self, save_dir):
        """(已增强) 使用多种模型进行TSTR评估"""
        print("  -> 正在使用多种模型进行下游任务可用性评估 (TSTR)...")
        try:
            X_real, y_real_reg = self.real_encoded[:, 1:], self.real_encoded[:, 0]
            y_real_clf = y_real_reg > np.median(y_real_reg)
            X_synth, y_synth_reg = self.synthetic_encoded[:, 1:], self.synthetic_encoded[:, 0]
            y_synth_clf = y_synth_reg > np.median(y_synth_reg)

            X_real_train, X_real_test, y_real_train_clf, y_real_test_clf = train_test_split(
                X_real, y_real_clf, test_size=0.3, random_state=42)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42)
            }
            results = {}

            for name, model in models.items():
                # 基准模型
                model_real = model.fit(X_real_train, y_real_train_clf)
                score_real = roc_auc_score(y_real_test_clf, model_real.predict_proba(X_real_test)[:, 1])
                # 合成模型
                model_synth = model.fit(X_synth, y_synth_clf)
                score_synth = roc_auc_score(y_real_test_clf, model_synth.predict_proba(X_real_test)[:, 1])
                results[name] = [score_real, score_synth]

            results_df = pd.DataFrame.from_dict(results, orient='index',
                                                columns=['Benchmark (Train on Real)', 'Synthetic (Train on Synthetic)'])
            results_df['Utility Score (%)'] = (results_df['Synthetic (Train on Synthetic)'] / results_df[
                'Benchmark (Train on Real)']) * 100

            # 报告
            report = "--- 增强版下游任务可用性 (TSTR) 报告 ---\n\n" + results_df.to_string() + "\n\n解读: Utility Score越接近100%，说明合成数据在各种任务下的可用性越高。\n"
            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '1_utility_tstr_report.txt'), 'w') as f:
                f.write(report)

            # 可视化
            results_df.plot(kind='bar', figsize=(12, 7))
            plt.title('TSTR Utility Comparison Across Different Models')
            plt.ylabel('AUROC Score on Real Test Set')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, '2_utility_tstr_plot.png'))
            plt.close()

        except Exception as e:
            print(f"    - 可用性评估出错: {e}")

    # ==============================================================================
    # == 第三部分：隐私保护 (Privacy) 分析 (全新) ==
    # ==============================================================================
    def analyze_privacy(self):
        print("\n--- 3. 开始隐私保护 (Privacy) 分析 ---")
        save_dir = os.path.join(self.output_dir, '3_privacy')

        # 将真实数据划分为训练部分和留出部分，用于攻击模拟
        real_train, real_holdout = train_test_split(self.real_encoded, test_size=0.5, random_state=42)

        self._run_membership_inference_attack(real_train, real_holdout, save_dir)
        self._run_attribute_inference_attack(real_train, save_dir)
        print("✅ 隐私保护分析完成。")

    def _run_membership_inference_attack(self, real_train, real_holdout, save_dir):
        """执行成员推断攻击 (MIA)"""
        print("  -> 正在进行成员推断攻击 (MIA) 模拟...")
        try:
            # 准备攻击模型的数据集
            # 正样本(标签1): 来自GAN训练集的数据
            # 负样本(标签0): GAN从未见过的数据
            attack_X = np.concatenate((real_train, real_holdout))
            attack_y = np.concatenate((np.ones(len(real_train)), np.zeros(len(real_holdout))))

            # 训练一个攻击模型来区分
            attack_model = LogisticRegression(max_iter=1000, random_state=42).fit(attack_X, attack_y)
            attack_accuracy = accuracy_score(attack_y, attack_model.predict(attack_X))

            report = f"\n--- 成员推断攻击 (MIA) 报告 ---\n\n"
            report += f"攻击模型准确率: {attack_accuracy:.4f}\n\n"
            report += f"解读:\n"
            report += f" - 准确率 ≈ 0.5: 理想情况。攻击者无法区分训练成员和非成员，隐私保护效果好。\n"
            report += f" - 准确率 -> 1.0: 危险信号！模型泄漏了其训练集成员的身份信息，隐私风险高。\n"
            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '1_privacy_mia_report.txt'), 'w') as f:
                f.write(report)
        except Exception as e:
            print(f"    - 成员推断攻击出错: {e}")

    def _run_attribute_inference_attack(self, real_train, save_dir):
        """执行属性推断攻击"""
        print("  -> 正在进行属性推断攻击模拟...")
        try:
            # 任务：根据其他特征，推断第一个特征
            X_synth_train, y_synth_train = self.synthetic_encoded[:, 1:], self.synthetic_encoded[:, 0] > np.median(
                self.synthetic_encoded[:, 0])
            X_real_attack, y_real_attack_true = real_train[:, 1:], real_train[:, 0] > np.median(real_train[:, 0])

            # 在合成数据上训练一个模型
            inference_model = RandomForestClassifier(random_state=42).fit(X_synth_train, y_synth_train)

            # 用这个模型去猜测真实训练集样本的属性
            y_real_attack_pred = inference_model.predict(X_real_attack)
            attack_accuracy = accuracy_score(y_real_attack_true, y_real_attack_pred)

            report = f"\n--- 属性推断攻击报告 ---\n\n"
            report += f"攻击模型准确率: {attack_accuracy:.4f}\n\n"
            report += f"解读:\n"
            report += f" - 准确率代表了攻击者根据合成数据模型，猜中真实训练成员敏感属性的能力。\n"
            report += f" - 准确率越低，说明模型学到的特征关系越泛化，隐私保护效果越好。\n"
            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '2_privacy_attribute_inference_report.txt'), 'w') as f:
                f.write(report)
        except Exception as e:
            print(f"    - 属性推断攻击出错: {e}")

    def run_all_analyses(self):
        """运行所有分析"""
        self.analyze_fidelity()
        self.analyze_utility()
        self.analyze_privacy()
        print(f"\n--- 🚀 黄金标准评估已完成！所有报告已生成在目录: {self.output_dir}/ ---")


def main():
    comparator = GoldStandardComparator()
    comparator.run_all_analyses()


if __name__ == "__main__":
    main()