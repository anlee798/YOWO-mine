from yolo import YoloBody
from torch.autograd import Variable
import torch

anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 80
phi = 's'
backbone = 'cspdarknet'
pretrained = False
input_shape = [224, 224]
model = YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
input_var = Variable(torch.randn(1, 3, 224, 224))
outputs = model(input_var)

print(len(outputs))
#print(outputs)

print(outputs[0].shape)
print(outputs[1].shape)
print(outputs[2].shape)

'''
python demo.py

torch.Size([1, 255, 7, 7])
torch.Size([1, 255, 14, 14])
torch.Size([1, 255, 28, 28])
'''