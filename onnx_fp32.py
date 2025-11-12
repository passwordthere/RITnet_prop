import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
import cv2

from eye_guidance_processor import BatchedEyeGuidanceProcessor


class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)
    
    
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out
    
class DenseNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=32,concat=True,dropout=False,prob=0):
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        # 修正：所有 self.xN 全部改为局部变量 xN
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.up_block1(x4,x5)
        x7 = self.up_block2(x3,x6)
        x8 = self.up_block3(x2,x7)
        x9 = self.up_block4(x1,x8)
        
        if self.dropout:
            out = self.out_conv1(self.dropout1(x9))
        else:
            out = self.out_conv1(x9)
                       
        return out


def export(model, example_inputs, onnx_path):
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    onnx_program.save(onnx_path)


def inference(example_inputs, onnx_path):
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]

    ort_session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
    return onnxruntime_outputs


def torch_inference(model, example_inputs, onnxruntime_outputs):
    torch_outputs = model(*example_inputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")


if __name__ == '__main__':

    # example_inputs = (torch.randn(2, 1, 512, 768),)
    frame_0 = cv2.imread('check/widthOrProp_od.jpg')
    frame_1 = frame_0.copy()
    processor = BatchedEyeGuidanceProcessor(redness=False)
    example_inputs, _ = processor._preprocess_batch(frame_0, frame_1)

    onnx_path = "infraredx.onnx"

    model = DenseNet2D(dropout=True,prob=0.2)
    filename = 'infrared.pkl'
    model.load_state_dict(torch.load(filename))
    model.eval()

    export(model, example_inputs, onnx_path)
    onnxruntime_outputs = inference(example_inputs, onnx_path)
    torch_inference(model, example_inputs, onnxruntime_outputs)