import torch
from thop import profile
from models import model_dict

model = model_dict['densenet']
# 假设输入是 (批量大小=1, 通道=3, 高度=224, 宽度=224)
input_tensor = torch.randn(1, 1, 400, 640)

# 计算 FLOPs 和参数量
# 注意：thop 计算的是 MACs (乘加运算) 的数量，通常计为 2 FLOPs，
# 但 thop 默认将 MACs 作为 FLOPs 输出。
total_flops, total_params = profile(model, inputs=(input_tensor, ), verbose=False)

# 转换为 GFLOPs (1 GFLOPs = 10^9 FLOPs)
gflops = total_flops / 1e9

print(f"Model Name: RITNET")
print(f"Total FLOPs: {total_flops} (约 {gflops:.2f} GFLOPs)")
print(f"Total Parameters: {total_params / 1e6:.2f} Million")
