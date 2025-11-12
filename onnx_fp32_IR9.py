import torch

from models import model_dict

example_inputs = (torch.randn(2, 1, 512, 768),)
# frame_0 = cv2.imread('check/widthOrProp_od.jpg')
# frame_1 = example_inputs.copy()
# processor = BatchedEyeGuidanceProcessor(redness=False)
# example_inputs, _ = processor._preprocess_batch(frame_0, frame_1)

onnx_path = "infraredx.onnx"

model = model_dict['densenet']
filename = 'infrared.pkl'
model.load_state_dict(torch.load(filename))
model.eval()

INPUT_NAMES = ['input']
OUTPUT_NAMES = ['output']
torch.onnx.export(
    model,
    example_inputs,
    'infraredx.onnx',
    export_params=True,             # 导出训练的参数权重
    opset_version=12, # **指定目标 Opset 版本**
    do_constant_folding=True,       # 执行常量折叠优化
    input_names=INPUT_NAMES,        # 为输入节点命名
    output_names=OUTPUT_NAMES,      # 为输出节点命名
    dynamic_axes={                  # 定义动态轴 (例如 Batch size)
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)