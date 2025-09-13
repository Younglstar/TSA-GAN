# file: final_gold_standard_evaluation.py (推荐的最终文件名)

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from pgmpy.estimators import PC
import networkx as nx
from scipy.spatial.distance import cdist
# 导入 causal-learn 的核心功能
from notears.linear import notears_linear
from causallearn.graph.SHD import SHD # 用于计算结构汉明距离
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Node import Node
from causallearn.graph.Edge import Edge, Endpoint
from causallearn.graph.SHD import SHD
from causallearn.utils.GraphUtils import GraphUtils # 用于可视化的工具
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
        self.output_dir = 'final_gold_standard_report'
        os.makedirs(os.path.join(self.output_dir, '1_fidelity'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, '2_utility'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, '3_privacy'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, '4_causality'), exist_ok=True)
        self.report_content = "--- Synthetic Data Gold Standard Report ---\n\n"

        print("📊 正在加载所有分析所需的数据...")
        try:
            gan_conf = GANConfig()
            encdec_conf = EncoderDecoderConfig()

            # 1. 加载合成数据
            with open(gan_conf.SYNTHETIC_DATA_PATH, 'rb') as f:
                self.synthetic_encoded = pickle.load(f)

            # 2. 加载完整的真实数据（用于保真度和因果分析）
            with open(encdec_conf.ENCODED_DATA_PATH, 'rb') as f:
                self.real_encoded_full = pickle.load(f)

            # 3. 加载在训练前预留的真实测试集
            with open('test_data_encoded.pkl', 'rb') as f:
                self.real_test_data = pickle.load(f)

            # 4. 通过差集，重建出GAN实际使用的训练集
            # 为了精确，我们将数据转换为元组集合进行操作
            full_set = set(map(tuple, self.real_encoded_full))
            test_set = set(map(tuple, self.real_test_data))
            train_set = full_set - test_set
            self.real_train_data = np.array(list(train_set))

            self.n_synthetic = len(self.synthetic_encoded)
            self.n_real = len(self.real_encoded_full)
            self.n_features = self.real_encoded_full.shape[1]

            print("数据加载完成！")
            print(f"  - 真实训练集 (重建后) 大小: {self.real_train_data.shape}")
            print(f"  - 真实测试集 (预留) 大小: {self.real_test_data.shape}")
            print(f"  - 合成数据集大小: {self.synthetic_encoded.shape}")

        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}")
            print("  请确保 'train_gan.py' 已成功运行，并生成了 'test_data_encoded.pkl' 文件。")
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

    def plot_autocorrelation_analysis(self, save_dir, matplotlib=None):
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
                plot_acf(self.real_encoded_full[:, i], ax=axes[i], lags=lags, title=f'Autocorrelation for Feature {i + 1}',
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
                real_feature, synth_feature = self.real_encoded_full[:, i], self.synthetic_encoded[:, i]
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
            moments = {'mean': (np.mean(self.real_encoded_full, axis=0), np.mean(self.synthetic_encoded, axis=0)),
                       'std': (np.std(self.real_encoded_full, axis=0), np.std(self.synthetic_encoded, axis=0)),
                       'skew': (stats.skew(self.real_encoded_full, axis=0), stats.skew(self.synthetic_encoded, axis=0)),
                       'kurtosis': (
                           stats.kurtosis(self.real_encoded_full, axis=0), stats.kurtosis(self.synthetic_encoded, axis=0))}
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
            real_corr, synth_corr = np.corrcoef(self.real_encoded_full.T), np.corrcoef(self.synthetic_encoded.T)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
            sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1).set_title(
                'Real Data Correlations')
            sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1).set_title(
                'Synthetic Data Correlations')
            diff_corr = np.abs(real_corr - synth_corr)
            sns.heatmap(diff_corr, ax=ax3, cmap='YlOrRd', vmin=0, vmax=1).set_title(
                f'Absolute Correlation Differences\nMean Diff: {np.mean(diff_corr):.3f}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'correlation_analysis.png'))
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
            combined_data = np.vstack([self.real_encoded_full, self.synthetic_encoded])
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
            stats_dict = {'Real_Mean': np.mean(self.real_encoded_full, axis=0),
                          'Synthetic_Mean': np.mean(self.synthetic_encoded, axis=0),
                          'Real_Std': np.std(self.real_encoded_full, axis=0),
                          'Synthetic_Std': np.std(self.synthetic_encoded, axis=0),
                          'Real_Skew': stats.skew(self.real_encoded_full, axis=0),
                          'Synthetic_Skew': stats.skew(self.synthetic_encoded, axis=0),
                          'Real_Kurtosis': stats.kurtosis(self.real_encoded_full, axis=0),
                          'Synthetic_Kurtosis': stats.kurtosis(self.synthetic_encoded, axis=0)}
            ks_stats, ks_pvals = [], []
            for i in range(self.n_features):
                ks_stat, p_val = ks_2samp(self.real_encoded_full[:, i], self.synthetic_encoded[:, i])
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
        """(已修改) 使用预定义的训练/测试集进行TSTR评估"""
        print("  -> 正在使用预留测试集进行下游任务可用性评估 (TSTR)...")
        try:
            # 准备合成数据 (训练集)
            X_synth_train, y_synth_reg = self.synthetic_encoded[:, 1:], self.synthetic_encoded[:, 0]
            y_synth_train_clf = y_synth_reg > np.median(y_synth_reg)

            # 准备真实训练数据 (用于训练基准模型)
            X_real_train, y_real_train_reg = self.real_train_data[:, 1:], self.real_train_data[:, 0]
            y_real_train_clf = y_real_train_reg > np.median(y_real_train_reg)

            # 准备真实测试数据 (用于评估所有模型)
            X_real_test, y_real_test_reg = self.real_test_data[:, 1:], self.real_test_data[:, 0]
            y_real_test_clf = y_real_test_reg > np.median(y_real_test_reg)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42)
            }
            results = {}

            for name, model in models.items():
                # 基准模型：在真实训练集上训练，在真实测试集上评估
                model_real = model.fit(X_real_train, y_real_train_clf)
                score_real = roc_auc_score(y_real_test_clf, model_real.predict_proba(X_real_test)[:, 1])

                # 合成模型：在合成数据上训练，在真实测试集上评估
                model_synth = model.fit(X_synth_train, y_synth_train_clf)
                score_synth = roc_auc_score(y_real_test_clf, model_synth.predict_proba(X_real_test)[:, 1])

                results[name] = [score_real, score_synth]

            results_df = pd.DataFrame.from_dict(results, orient='index',
                                                columns=['Benchmark (Train on Real)', 'Synthetic (Train on Synthetic)'])
            results_df['Utility Score (%)'] = (results_df['Synthetic (Train on Synthetic)'] / results_df[
                'Benchmark (Train on Real)']) * 100

            # ... (报告和可视化的代码保持不变) ...
            report = "--- 增强版下游任务可用性 (TSTR) 报告 ---\n\n" + results_df.to_string() + "\n\n解读: Utility Score越接近100%，说明合成数据在各种任务下的可用性越高。\n"
            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '1_utility_tstr_report.txt'), 'w') as f:
                f.write(report)
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

        # 使用在__init__中重建的训练集和加载的测试集
        real_train = self.real_train_data
        real_holdout = self.real_test_data

        self._run_membership_inference_attack(real_train, self.synthetic_encoded, save_dir)
        self._run_attribute_inference_attack(real_train, save_dir)
        self._run_reidentification_attack(real_train, real_holdout, save_dir)
        print("✅ 隐私保护分析完成。")

    def _run_membership_inference_attack(self, real_train, real_holdout, save_dir):
        """
        (已修正) 执行成员推断攻击 (MIA)，以评估合成数据的隐私保护能力。
        """
        print("  -> 正在进行成员推断攻击 (MIA) 模拟...")
        try:
            # ======================= 核心修改开始 =======================

            # 准备攻击模型的数据集
            # 正样本 (标签1): 真正参与了GAN训练的原始数据成员 (real_train)
            # 负样本 (标签0): 由GAN生成的、不对应任何真实个体的合成数据 (self.synthetic_encoded)

            n_real_train = len(real_train)
            n_synthetic = len(self.synthetic_encoded)

            # 为了平衡数据集，我们让正负样本数量一致
            # 如果合成数据比真实训练数据多，我们就从中随机抽样
            if n_synthetic > n_real_train:
                indices = np.random.permutation(n_synthetic)[:n_real_train]
                synthetic_samples_for_attack = self.synthetic_encoded[indices]
            else:
                synthetic_samples_for_attack = self.synthetic_encoded

            attack_X = np.concatenate((real_train[:len(synthetic_samples_for_attack)], synthetic_samples_for_attack))
            attack_y = np.concatenate(
                (np.ones(len(synthetic_samples_for_attack)), np.zeros(len(synthetic_samples_for_attack))))

            # 训练一个攻击模型来区分“真实训练成员”与“合成样本”
            attack_model = LogisticRegression(max_iter=1000, random_state=42).fit(attack_X, attack_y)
            attack_accuracy = accuracy_score(attack_y, attack_model.predict(attack_X))

            report = f"\n--- 成员推断攻击 (MIA) 报告 ---\n\n"
            report += f"攻击模型准确率: {attack_accuracy:.4f}\n\n"
            report += f"解读:\n"
            report += f" - 该准确率反映了攻击者区分“真实训练数据”和“GAN生成数据”的能力。\n"
            report += f" - 准确率 ≈ 0.5 (随机猜测): 理想情况。说明合成数据在统计上与真实数据无法区分，隐私保护效果好。\n"
            report += f" - 准确率 -> 1.0: 危险信号。说明合成数据与真实数据存在明显差异，可能是模式崩溃的迹象，或者未能很好地学习真实分布。\n"

            # ======================== 核心修改结束 ========================

            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '1_privacy_mia_report.txt'), 'w') as f:
                f.write(report)
        except Exception as e:
            print(f"    - 成员推断攻击出错: {e}")

    def _run_attribute_inference_attack(self, real_train, save_dir):
        print("  -> 正在进行全面的属性推断攻击模拟 (遍历所有特征)...")
        try:
            # 如果特征数少于2，则无法进行此攻击
            if self.n_features < 2:
                print("    - 特征数少于2，跳过属性推断攻击。")
                report = "\n--- 属性推断攻击报告 ---\n\n特征数少于2，无法执行此攻击。\n"
                self.report_content += report
                with open(os.path.join(save_dir, '2_privacy_attribute_inference_report.txt'), 'w') as f:
                    f.write(report)
                return

            attack_results = {}

            # 使用tqdm来显示循环进度
            for target_feature_idx in tqdm(range(self.n_features), desc="    - 攻击进度"):
                # 1. 准备数据：将当前特征作为目标(y)，其余作为输入(X)
                # 使用np.delete可以方便地移除目标特征列
                X_synth_train = np.delete(self.synthetic_encoded, target_feature_idx, axis=1)
                X_real_attack = np.delete(real_train, target_feature_idx, axis=1)

                # 将目标特征二值化（大于中位数为1，否则为0）作为分类任务
                y_synth_train = self.synthetic_encoded[:, target_feature_idx] > np.median(
                    self.synthetic_encoded[:, target_feature_idx])
                y_real_attack_true = real_train[:, target_feature_idx] > np.median(real_train[:, target_feature_idx])

                # 2. 在合成数据上训练一个攻击模型
                # 使用LightGBM，它通常比随机森林更快且性能相当
                inference_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                inference_model.fit(X_synth_train, y_synth_train)

                # 3. 用这个模型去猜测真实训练集样本的属性
                y_real_attack_pred = inference_model.predict(X_real_attack)
                attack_accuracy = accuracy_score(y_real_attack_true, y_real_attack_pred)

                # 4. 记录结果
                attack_results[f'Feature_{target_feature_idx}'] = attack_accuracy

            # 5. 生成详细的报告
            results_df = pd.DataFrame.from_dict(attack_results, orient='index', columns=['Attack Accuracy'])
            mean_accuracy = results_df['Attack Accuracy'].mean()

            report = "\n--- 全面属性推断攻击报告 ---\n\n"
            report += "对每个特征作为敏感属性进行推断的攻击准确率:\n"
            report += results_df.to_string()
            report += f"\n\n平均攻击准确率: {mean_accuracy:.4f}\n\n"
            report += "解读:\n"
            report += " - 该报告展示了攻击者利用合成数据模型，分别猜测真实训练数据中每一个特征的能力。\n"
            report += " - 准确率代表了攻击成功的概率。随机猜测的基准是0.5。\n"
            report += " - 整体准确率越接近0.5，说明模型学到的特征关系越泛化，隐私保护效果越好。\n"
            report += " - 如果某个特征的攻击准确率远高于0.5，则表明该特征存在较高的泄露风险。\n"

            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '2_privacy_attribute_inference_report.txt'), 'w') as f:
                f.write(report)

            # 6. 生成可视化图表
            plt.figure(figsize=(14, 7))
            results_df.sort_values('Attack Accuracy').plot(kind='barh', figsize=(12, max(8, self.n_features * 0.3)))
            plt.axvline(x=0.5, color='r', linestyle='--', label='Random Guess (0.5)')
            plt.xlabel('Attack Accuracy')
            plt.ylabel('Target Feature')
            plt.title('Attribute Inference Attack Accuracy per Feature')
            plt.legend()
            plt.grid(axis='x', linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, '3_privacy_attribute_inference_plot.png'))
            plt.close()

        except Exception as e:
            print(f"    - 属性推断攻击出错: {e}")

    def _calculate_min_dists_in_batches(self, source_data, target_data, batch_size=1024):
        """
        (辅助函数) 分块计算源数据中每个点到目标数据集的最近邻距离。
        这避免了创建巨大的距离矩阵，从而解决了内存溢出问题。
        """
        print(f"    - 正在分块计算 {source_data.shape[0]} 个样本的最近邻距离...")

        # 转换为float32可以减少一半的内存占用，同时保持足够的精度
        source_data = source_data.astype(np.float32)
        target_data = target_data.astype(np.float32)

        all_min_dists = []
        num_batches = int(np.ceil(source_data.shape[0] / batch_size))

        # 使用tqdm创建进度条
        for i in tqdm(range(num_batches), desc="  -> 计算距离"):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, source_data.shape[0])
            batch = source_data[start_index:end_index]

            # 计算当前批次与所有目标数据点的距离矩阵
            # 这个矩阵很小，例如 (1024, 100000)
            dist_matrix_chunk = cdist(batch, target_data, metric='euclidean')

            # 找到当前批次中每个点的最近邻距离，并保存
            min_dists_chunk = dist_matrix_chunk.min(axis=1)
            all_min_dists.append(min_dists_chunk)

        # 将所有批次的结果拼接成一个完整的数组
        return np.concatenate(all_min_dists)

    def _run_reidentification_attack(self, real_train, real_holdout, save_dir):
        """
        (全新-已修正) 执行基于最近邻距离的重识别攻击模拟。
        使用分块处理来避免内存溢出。
        """
        print("  -> 正在进行重识别攻击 (最近邻) 模拟...")
        try:
            # 1. (修正) 分块计算每个真实训练集成员到合成数据集的最近邻距离
            dists_train = self._calculate_min_dists_in_batches(real_train, self.synthetic_encoded)

            # 2. (修正) 分块计算每个真实留出集成员到合成数据集的最近邻距离
            dists_holdout = self._calculate_min_dists_in_batches(real_holdout, self.synthetic_encoded)

            # 3. 使用 ROC-AUC 分数来量化可区分性 (此部分无需改动)
            all_dists = np.concatenate([dists_train, dists_holdout])
            all_labels = np.concatenate([np.ones_like(dists_train), np.zeros_like(dists_holdout)])
            auc_score = roc_auc_score(all_labels, -all_dists)

            # 4. 生成报告 (此部分无需改动)
            report = f"\n--- 重识别攻击 (Re-identification) 报告 ---\n\n"
            # ... (报告内容与之前相同) ...
            report += f"方法: 基于最近邻距离的成员与非成员可区分性分析。\n"
            report += f"重识别风险分数 (AUC): {auc_score:.4f}\n\n"
            report += "解读:\n"
            report += " - 该分数衡量了攻击者仅通过“与合成数据的接近程度”来区分真实训练成员和非成员的能力。\n"
            report += " - AUC ≈ 0.5: 理想情况，隐私保护效果好。说明GAN生成的泛化数据很好，没有“死记硬背”训练样本。\n"
            report += " - AUC -> 1.0: 危险信号，存在严重的重识别风险。说明合成数据中有一些点与真实训练成员异常接近，模型可能发生了记忆或过拟合。\n"

            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '4_privacy_reidentification_report.txt'), 'w') as f:
                f.write(report)

            # 5. 可视化距离分布的累积分布函数 (CDF) (此部分无需改动)
            plt.figure(figsize=(10, 7))
            # ... (绘图代码与之前相同) ...
            sorted_dists_train = np.sort(dists_train)
            sorted_dists_holdout = np.sort(dists_holdout)
            cdf_train = np.arange(1, len(sorted_dists_train) + 1) / len(sorted_dists_train)
            cdf_holdout = np.arange(1, len(sorted_dists_holdout) + 1) / len(sorted_dists_holdout)

            plt.plot(sorted_dists_train, cdf_train, label='Training Members (Trained on)')
            plt.plot(sorted_dists_holdout, cdf_holdout, label='Holdout Non-Members (Not Trained on)')
            plt.title('CDF of Nearest Neighbor Distances')
            plt.xlabel('Distance to closest synthetic sample')
            plt.ylabel('Cumulative Probability')
            plt.legend()
            plt.grid(True, linestyle=':')
            plt.text(0.95, 0.05, f'AUC = {auc_score:.4f}', transform=plt.gca().transAxes,
                     ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            plt.savefig(os.path.join(save_dir, '5_privacy_reidentification_plot.png'))
            plt.close()

        except Exception as e:
            print(f"    - 重识别攻击出错: {e}")
    # ==============================================================================
    # == 第四部分：因果关系 (Causality) 分析 (全新) ==
    # ==============================================================================
    def analyze_causality(self):
        """(全新) 运行因果发现与比较分析"""
        print("\n--- 4. 开始因果关系 (Causality) 分析 ---")
        save_dir = os.path.join(self.output_dir, '4_causality')
        os.makedirs(save_dir, exist_ok=True)
        report = "\n--- 因果关系分析报告 ---\n\n错误: 缺少必要的库 (cdt, networkx)，无法执行此分析。\n"
        with open(os.path.join(save_dir, 'causal_discovery_report.txt'), 'w') as f:
            f.write(report)
        self._run_causal_discovery_notears(save_dir)
        self._run_causal_discovery_analysis(save_dir)
        print("✅ 因果关系分析完成。")
        return
    def _run_causal_discovery_analysis(self, save_dir):
            """
            (全新) 使用PC算法发现因果图，并使用结构汉明距离(SHD)进行比较。
            """
            print("  -> 正在运行因果发现 (PC算法)...")
            try:
                # 将numpy数组转换为pandas DataFrame
                real_df = pd.DataFrame(self.real_encoded_full, columns=[f'F{i}' for i in range(self.n_features)])
                synthetic_df = pd.DataFrame(self.synthetic_encoded, columns=[f'F{i}' for i in range(self.n_features)])

                # 1. 从真实数据中发现因果图
                print("    - 正在从真实数据推断因果图...")
                pc_real = PC(data=real_df)
                # variant='stable' 是一种更可靠的PC算法版本
                # 为连续数据指定 ci_test='pearsonr'
                real_model = pc_real.estimate(variant='stable', ci_test='pearsonr', significance_level=0.05,n_jobs=-1)
                real_graph = nx.DiGraph(real_model.edges())

                # 2. 从合成数据中发现因果图
                print("    - 正在从合成数据推断因果图...")
                pc_synth = PC(data=synthetic_df)
                # 为连续数据指定 ci_test='pearsonr'
                synth_model = pc_synth.estimate(variant='stable', ci_test='pearsonr', significance_level=0.05,n_jobs=-1)
                synthetic_graph = nx.DiGraph(synth_model.edges())

                # 3. 比较两个图的结构 - 手动计算SHD
                # 结构汉明距离 (SHD): 将一个图转换为另一个图所需的边操作（增/删/反转）次数。
                shd = 0
                # 检查在real_graph中的每条边
                for u, v in real_graph.edges():
                    if not synthetic_graph.has_edge(u, v):
                        if synthetic_graph.has_edge(v, u):  # 边的方向反了
                            shd += 1
                        else:  # 缺少边
                            shd += 1
                # 检查在synthetic_graph中的每条边，看是否是多出来的
                for u, v in synthetic_graph.edges():
                    if not real_graph.has_edge(u, v) and not real_graph.has_edge(v, u):  # 多出来的边
                        shd += 1

                # 4. 生成报告 (与之前相同)
                report = "\n--- 因果发现与比较报告 ---\n\n"
                report += "方法: pgmpy PC因果发现算法 + 结构汉明距离 (SHD)\n"
                report += "重要假设: 本分析假设数据中不存在未观测到的混杂因素（因果充足性），且因果关系是无环的。\n\n"
                report += f"结构汉明距离 (SHD): {shd}\n\n"
                report += "解读:\n"
                report += f" - SHD衡量了合成数据因果图与真实数据因果图的结构差异。\n"
                report += f" - SHD = 0: 理想情况。说明合成数据完美地复现了从真实数据中推断出的因果结构。\n"
                report += f" - SHD > 0: 值越低，说明因果结构保留得越好。\n"

                print(report)
                self.report_content += report
                with open(os.path.join(save_dir, '1_causal_discovery_report.txt'), 'w') as f:
                    f.write(report)

                # 5. 可视化因果图 (与之前相同)
                fig, axes = plt.subplots(1, 2, figsize=(20, 9))
                fig.suptitle('Causal Graphs Comparison (using pgmpy)', fontsize=16)

                # 确保所有节点都出现在图中，即使它们是孤立的
                all_nodes = list(real_df.columns)
                real_graph.add_nodes_from(all_nodes)
                synthetic_graph.add_nodes_from(all_nodes)

                pos = nx.circular_layout(real_graph)  # 使用相同的布局以便比较
                nx.draw(real_graph, pos=pos, ax=axes[0], with_labels=True, node_color='skyblue', edge_color='gray',
                        node_size=2000, font_size=10, arrows=True)
                axes[0].set_title('Discovered Causal Graph from REAL Data')
                nx.draw(synthetic_graph, pos=pos, ax=axes[1], with_labels=True, node_color='lightcoral',
                        edge_color='gray', node_size=2000, font_size=10, arrows=True)
                axes[1].set_title(f'Discovered Causal Graph from SYNTHETIC Data\nSHD = {shd}')

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(save_dir, '2_causal_graphs_comparison.png'))
                plt.close()

            except Exception as e:
                print(f"    - 因果分析出错: {e}")

    def run_all_analyses(self):
        """运行所有分析"""
        self.analyze_fidelity()
        self.analyze_utility()
        self.analyze_privacy()
        #self.analyze_causality()
        print(f"\n--- 🚀 黄金标准评估已完成！所有报告已生成在目录: {self.output_dir}/ ---")

    def adj_to_graph(adj_matrix, node_names):
        """将邻接矩阵转换为 causal-learn 的 GeneralGraph 对象"""
        nodes = [Node(name) for name in node_names]
        g = GeneralGraph(nodes)
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if adj_matrix[i, j] != 0:  # 非零即认为存在因果边
                    g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
        return g

    def _matrix_to_graph(weight_matrix, columns, threshold=0.3):
        """
        将 NOTEARS 输出的权重矩阵转换为 NetworkX 有向图。
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(columns)
        for i in range(len(columns)):
            for j in range(len(columns)):
                if abs(weight_matrix[i, j]) > threshold:
                    graph.add_edge(columns[i], columns[j])
        return graph

    def _run_causal_discovery_notears(self, save_dir):
        """
        使用 NOTEARS 发现因果结构并计算 SHD
        """
        print("  -> 正在运行因果发现 (NOTEARS算法)...")
        try:
            real_data = self.real_encoded_full.astype(np.float32)
            synthetic_data = self.synthetic_encoded.astype(np.float32)
            feature_names = [f'F{i}' for i in range(self.n_features)]

            # 1. NOTEARS 发现因果权重矩阵
            print("    - 从真实数据推断因果图...")
            real_return_values = notears_linear(real_data, lambda1=0.1, loss_type='l2')
            # 权重矩阵 W 始终是第一个返回的元素
            real_weight_matrix = real_return_values[0]

            print("    - 正在从合成数据推断因果图...")
            synth_return_values = notears_linear(synthetic_data, lambda1=0.1, loss_type='l2')
            synth_weight_matrix = synth_return_values[0]

            # [可选的调试步骤] 如果您好奇，可以取消下面两行的注释，看看函数到底返回了几个值
            print(f"NOTEARS returned {len(real_return_values)} items.")

            # 2. 转换为 networkx 图用于可视化
            real_graph_nx = self._matrix_to_graph(real_weight_matrix, feature_names)
            synth_graph_nx = self._matrix_to_graph(synth_weight_matrix, feature_names)

            # 3. 转换为邻接矩阵 → causal-learn Graph → SHD
            real_adj = nx.to_numpy_array(real_graph_nx, nodelist=feature_names)
            synth_adj = nx.to_numpy_array(synth_graph_nx, nodelist=feature_names)

            real_graph_cl = self.adj_to_graph(real_adj, feature_names)
            synth_graph_cl = self.adj_to_graph(synth_adj, feature_names)

            shd_score = SHD(real_graph_cl, synth_graph_cl).get_shd()

            # 4. 生成报告
            report = "\n--- 因果发现与比较报告 (NOTEARS) ---\n\n"
            report += "方法: NOTEARS 因果发现算法 + 结构汉明距离 (SHD)\n"
            report += "重要假设: 因果关系是无环的，且不存在隐变量。\n\n"
            report += f"结构汉明距离 (SHD): {shd_score}\n\n"
            report += "解读:\n"
            report += " - SHD衡量合成数据与真实数据因果图的差异。\n"
            report += " - SHD = 0 表示合成数据完美复现真实因果结构。\n"
            report += " - 值越小越好。\n"

            print(report)
            self.report_content += report
            with open(os.path.join(save_dir, '1_causal_discovery_report_notears.txt'), 'w') as f:
                f.write(report)

            # 5. 可视化
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            fig.suptitle('Causal Graphs Comparison (using NOTEARS)', fontsize=16)

            pos = nx.circular_layout(real_graph_nx)
            nx.draw(real_graph_nx, pos=pos, ax=axes[0], with_labels=True,
                    node_color='skyblue', edge_color='gray',
                    node_size=2000, font_size=10, arrows=True)
            axes[0].set_title('Discovered Causal Graph from REAL Data')

            nx.draw(synth_graph_nx, pos=pos, ax=axes[1], with_labels=True,
                    node_color='lightcoral', edge_color='gray',
                    node_size=2000, font_size=10, arrows=True)
            axes[1].set_title(f'Discovered Causal Graph from SYNTHETIC Data\nSHD = {shd_score}')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(save_dir, '2_causal_graphs_comparison_notears.png'))
            plt.close()

        except Exception as e:
            print(f"    - 因果分析 (NOTEARS) 出错: {e}")


def main():
    comparator = GoldStandardComparator()
    comparator.run_all_analyses()


if __name__ == "__main__":
    main()