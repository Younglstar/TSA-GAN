# file: run_pipeline.py (一键执行所有流程的主脚本)

import subprocess
import sys
import os


def run_script(script_name: str):
    """
    一个辅助函数，用于执行指定的Python脚本，并检查是否成功。
    """
    print(f"\n{'=' * 20} 正在执行: {script_name} {'=' * 20}")

    # 确保我们使用的是当前Python环境的解释器
    python_executable = sys.executable
    # 使用subprocess来调用脚本，这样可以隔离每个步骤的环境
    result = subprocess.run([python_executable, script_name], text=True, encoding='utf-8')
    # 检查执行结果
    if result.returncode != 0:
        print(f"❌ 在执行 {script_name} 时发生严重错误！")
        print("--- STDOUT (标准输出) ---")
        print(result.stdout)
        print("--- STDERR (错误输出) ---")
        print(result.stderr)
        # 遇到错误，终止整个流程
        sys.exit(1)
    else:
        print(f"✅ {script_name} 执行成功。")
        # 打印脚本的输出，方便监控
        print(result.stdout)


def main():
    """
    主流程函数，按顺序执行所有步骤。
    """
    print("🚀 开始执行完整的合成数据生成与分析流水线...")

    # --- 定义流水线的每一个步骤 ---
    # 确保脚本名称与您项目中的文件名完全一致
    pipeline_steps = [
        "main.py",  # 1. 初步处理，生成 processed_data.pkl
        "normalize_main.py",  # 2. 数值归一化，生成 normalized_data.pkl
        "embedding_main.py",  # 3. 类别嵌入学习，生成 categorical_embeddings.pkl
        "train_encdec.py",  # 4. 训练CausalVAE模型，生成 causal_vae_model_best.pkl 和 encoded data
        "train_gan.py",  # 5. 训练高级GAN模型，生成 gan_final_model.pkl 和 synthetic data
        "final_decoder.py",  # 6. 解码GAN生成的潜在数据，生成 final_synthetic_data.pkl
        "final_comprarison.py"  # 7. 生成最终的、全面的对比分析报告
    ]

    # --- 按顺序执行所有步骤 ---
    for step_script in pipeline_steps:
        print('1111111')
        if not os.path.exists(step_script):
            print(f"❌ 错误：脚本文件 '{step_script}' 不存在！请检查文件名是否正确。")
            sys.exit(1)
        run_script(step_script)

    print("\n🎉🎉🎉 恭喜！整个流水线已全部成功执行！🎉🎉🎉")
    print(f"   -> 您最终的分析报告已生成在 'final_comparison_report/' 目录中。")
    print(f"   -> 您最终的合成数据CSV文件已生成在 'final_comparison_report/original_space_analysis/' 目录中。")


if __name__ == "__main__":
    main()