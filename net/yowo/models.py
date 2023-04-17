import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('/Users/zhuanlei/Documents/WorkSpace/ActionRecognition/YOWO-mine/model')
from backbone_3d.shufflenetv2 import ShuffleNetV2
# from backbone_3d.shufflenetv2 import ShuffleNetV2
from backbone_2d import yolo,ConvNext 
from torch.autograd import Variable

anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 80
phi = 's'
backbone = 'cspdarknet'
pretrained = False
input_shape = [224, 224]

class YOWO(nn.Module):
    def __init__(self):
        super(YOWO, self).__init__()
        
        self.backbone_3d = ShuffleNetV2(num_classes=101, sample_size=224, width_mult=1.)
        #self.backbone_3d = shufflenetv2.get_model(num_classes=101, sample_size=224, width_mult=1.)
        self.backbone_2d = yolo.YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
        
    def forward(self,input):
        x_3d = input # Input clip
        x_2d = input[:, :, -1, :, :] # Last frame of the clip that is read
        # print("Start x_3d: ",x_3d.shape) #torch.Size([1, 3, 16, 224, 224])
        # print("Start x_2d: ",x_2d.shape) #torch.Size([1, 3, 224, 224])
        x_3d = self.backbone_3d(x_3d)
        print("x_3d:",x_3d)
        x_2d = self.backbone_2d(x_2d)
        print("x_2d:",x_2d)
        return 0
    
if __name__ == "__main__":
    model = YOWO()
    input_var = Variable(torch.randn(1, 3, 16, 224, 224))
    outputs = model(input_var)
    
    print("outputs.shape",outputs.shape)