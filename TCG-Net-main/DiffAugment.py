# file: debug_opacus_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

print("--- 开始Opacus最小可复现示例(修复版) ---")

BATCH_SIZE = 16
INPUT_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DP_ENABLE = True
DP_NOISE_MULTIPLIER = 1.0
DP_MAX_GRAD_NORM = 1.2
GRAD_PENALTY_WEIGHT = 10.0

class SafeDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def gradient_penalty(discriminator, real_data, fake_data, dp_enabled, device):
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1, device=device).expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # 🚨 用未包装的模型避免 Opacus hook 干扰
    if dp_enabled and hasattr(discriminator, "_module"):
        d_interpolates = discriminator._module(interpolates)
    else:
        d_interpolates = discriminator(interpolates)

    grad = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(bsz, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


try:
    print("\n--- 步骤A: 初始化模型和优化器 ---")
    D = SafeDiscriminator(INPUT_DIM).to(DEVICE)
    d_opt = optim.Adam(D.parameters(), lr=1e-4)

    real_data = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
    fake_data = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)

    print("--- 步骤B: 附加Opacus隐私引擎 ---")
    if DP_ENABLE:
        dummy_ds = torch.utils.data.TensorDataset(real_data)  # 只是占位
        dummy_dl = torch.utils.data.DataLoader(dummy_ds, batch_size=BATCH_SIZE)
        pe = PrivacyEngine()
        D, d_opt, dummy_dl = pe.make_private(
            module=D,
            optimizer=d_opt,
            data_loader=dummy_dl,
            noise_multiplier=DP_NOISE_MULTIPLIER,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )
        print("Opacus附加成功。")

    print("--- 步骤C: 前向(不拼接) ---")
    d_opt.zero_grad(set_to_none=True)

    d_real = D(real_data)   # [16, 1]
    d_fake = D(fake_data)   # [16, 1]
    d_loss = d_fake.mean() - d_real.mean()

    print("--- 步骤D: 计算梯度惩罚(同一模型/同一batch维度) ---")
    gp = gradient_penalty(D, real_data, fake_data,True, device=DEVICE)

    print("--- 步骤E: 反向传播 ---")
    total = d_loss + GRAD_PENALTY_WEIGHT * gp
    total.backward()

    print("--- 步骤F: step ---")
    d_opt.step()
    print("\n✅ 修复版最小示例成功运行！")

except Exception as e:
    import traceback
    print("\n❌ 运行失败")
    traceback.print_exc()
