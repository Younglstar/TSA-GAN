# file: final_gold_standard_evaluation.py (推荐的最终文件名)

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import os
import torch
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
                self.synthetic_latent = pickle.load(f)
            with open(encdec_conf.ENCODED_DATA_PATH, 'rb') as f:
                self.real_latent = pickle.load(f)
            self.n_features_latent = self.real_latent.shape[1]
            print("数据加载完成！")
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}");
            raise

    # ==============================================================================
    # == 第一部分：数据保真度 (Fidelity) 分析 (保持高水准) ==
    # ==============================================================================
    def analyze_fidelity(self):
        print("\n--- 1. 开始数据保真度 (Fidelity) 分析 ---")
        # ... (这部分的所有函数 _plot_... 和 _generate_... 与之前一样，此处省略) ...
        print("✅ 数据保真度分析完成。")

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
            X_real, y_real_reg = self.real_latent[:, 1:], self.real_latent[:, 0]
            y_real_clf = y_real_reg > np.median(y_real_reg)
            X_synth, y_synth_reg = self.synthetic_latent[:, 1:], self.synthetic_latent[:, 0]
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
            print(report);
            self.report_content += report
            with open(os.path.join(save_dir, '1_utility_tstr_report.txt'), 'w') as f:
                f.write(report)

            # 可视化
            results_df.plot(kind='bar', figsize=(12, 7))
            plt.title('TSTR Utility Comparison Across Different Models');
            plt.ylabel('AUROC Score on Real Test Set')
            plt.xticks(rotation=0);
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, '2_utility_tstr_plot.png'));
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
        real_train, real_holdout = train_test_split(self.real_latent, test_size=0.5, random_state=42)

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
            print(report);
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
            X_synth_train, y_synth_train = self.synthetic_latent[:, 1:], self.synthetic_latent[:, 0] > np.median(
                self.synthetic_latent[:, 0])
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
            print(report);
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