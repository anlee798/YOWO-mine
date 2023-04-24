from net.backbone_2d_yolox.yolo import YoloBody
from torch.autograd import Variable
import torch

num_classes = 80
phi = 's'

model = YoloBody(num_classes, phi)
input_var = Variable(torch.randn(1, 3, 224, 224))
outputs = model(input_var)

print(len(outputs))
print(type(outputs))
print(outputs['pred_reg'][0].size())

# print(outputs[0].shape)
# print(outputs[1].shape)
# print(outputs[2].shape)

'''
    python yolox.py
    
torch.Size([1, 85, 28, 28])
torch.Size([1, 85, 14, 14])
torch.Size([1, 85, 7, 7])

'''