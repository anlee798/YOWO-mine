import torch
import torch.nn as nn
import torch.nn.functional as F
# import sys
# sys.path.append('/Users/zhuanlei/Documents/WorkSpace/ActionRecognition/YOWO-mine/model')
from net.backbone_3d.shufflenetv2 import ShuffleNetV2
# from backbone_3d.shufflenetv2 import ShuffleNetV2
from net.backbone_2d import yolo,ConvNext 
from torch.autograd import Variable
from utils.cfam import CFAMBlock

anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 80
phi = 's'
backbone = 'cspdarknet'
pretrained = False
input_shape = [224, 224]

NUM_CLASSES = 80

class YOWO(nn.Module):
    def __init__(self):
        super(YOWO, self).__init__()
        
        self.backbone_3d = ShuffleNetV2(num_classes=101, sample_size=224, width_mult=1.)
        #self.backbone_3d = shufflenetv2.get_model(num_classes=101, sample_size=224, width_mult=1.)
        self.backbone_2d = yolo.YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
        self.cfam1 = CFAMBlock(116+255, 1024)
        self.cfam2 = CFAMBlock(232+255, 1024)
        self.cfam3 = CFAMBlock(464+255, 1024)
        self.conv_final = nn.Conv2d(1024, 3 * (NUM_CLASSES+4+1), kernel_size=1, bias=False)
        
    def forward(self,input):
        x_3d = input # Input clip
        x_2d = input[:, :, -1, :, :] # Last frame of the clip that is read
        # print("Start x_3d: ",x_3d.shape) #torch.Size([1, 3, 16, 224, 224])
        # print("Start x_2d: ",x_2d.shape) #torch.Size([1, 3, 224, 224])
        x_3d = self.backbone_3d(x_3d)
        # print("x_3d:",x_3d[0].shape) #torch.Size([1, 116, 4, 28, 28])
        # print("x_3d:",x_3d[1].shape) #torch.Size([1, 232, 2, 14, 14])
        # print("x_3d:",x_3d[2].shape) #torch.Size([1, 464, 1, 7, 7])
        x = x_3d[0]
        xx = x_3d[1]
        xxx = x_3d[2]
        if x_3d[0].size(2) > 1:
            x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2)
        if x_3d[1].size(2) > 1:
            xx = torch.mean(xx, dim=2, keepdim=True)
        xx = xx.squeeze(2)
        if x_3d[2].size(2) > 1:
            xxx = torch.mean(xxx, dim=2, keepdim=True)
        xxx = xxx.squeeze(2)
        #print(x.size())
        x_2d = self.backbone_2d(x_2d)
        # print("x_2d:",x_2d[0].shape) #torch.Size([1, 255, 7, 7])
        # print("x_2d:",x_2d[1].shape) #torch.Size([1, 255, 14, 14])
        # print("x_2d:",x_2d[2].shape) #torch.Size([1, 255, 28, 28])
        xy1 = torch.cat((x,x_2d[2]),dim=1)
        xy1 = self.cfam1(xy1)
        xy1 = self.conv_final(xy1)
        print(xy1.size())
        
        xy2 = torch.cat((xx,x_2d[1]),dim=1)
        xy2 = self.cfam2(xy2)
        xy2 = self.conv_final(xy2)
        print(xy2.size())
        
        xy3 = torch.cat((xxx,x_2d[0]),dim=1)
        xy3 = self.cfam3(xy3)
        xy3 = self.conv_final(xy3)
        print(xy3.size())
        return 0
    
if __name__ == "__main__":
    model = YOWO()
    input_var = Variable(torch.randn(1, 3, 16, 224, 224))
    outputs = model(input_var)
    
    print("outputs.shape",outputs)