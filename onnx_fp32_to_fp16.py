import onnx
from onnxconverter_common import convert_float_to_float16

# 1. 加载你的 FP32 模型
fp32_model_path = 'infrared.onnx'
model = onnx.load(fp32_model_path)

# 2. 转换为 FP16
model_fp16 = convert_float_to_float16(model)

# 3. 保存新的 FP16 模型
fp16_model_path = 'infrared_fp16.onnx'
onnx.save(model_fp16, fp16_model_path)

print(f"FP16 模型已保存到: {fp16_model_path}")