import torch.nn as nn
import torch
# from net.backbone_3d.shufflenetv2 import ShuffleNetV2
from net.backbone_3d.shufflenetv2origin import ShuffleNetV2
from net.backbone_2d_yolox.yolo import YoloBody
from torch.autograd import Variable
import torch.nn.functional as F
from net.encode import build_channel_encoder
import math 
from net.basic.head import build_head

num_classes = 24
phi = 's'

cfg = {
    'stride': [8, 16, 32],
    # head
    'head_dim': 64,
    'head_norm': 'BN',
    'head_act': 'lrelu',
    'num_cls_heads': 2,
    'num_reg_heads': 2,
    'head_depthwise': True,
}

class YOWO(nn.Module):
    def __init__(self,cfg):
        super(YOWO, self).__init__()
        self.backbone_3d = ShuffleNetV2(num_classes=101, sample_size=224, width_mult=1.)
        self.backbone_2d = YoloBody(num_classes, phi)
        self.num_classes = 24
        self.stride = cfg['stride']
        self.device = 'cpu'
    
        bk_dim_3d = 464
        bk_dim_2d = 24
        print("bk_dim_2d",bk_dim_2d)
        # cls channel encoder
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])
        
        ## head
        self.heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))]
        ) 
        
        ## pred
        head_dim = cfg['head_dim']
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
                for _ in range(len(cfg['stride']))
                ]) 
        
    def generate_anchors(self, fmp_size, stride):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.to(self.device)

        return anchors
        
    def forward(self,input):
        video_clips = input # Input clip
        key_frame = input[:, :, -1, :, :] # Last frame of the clip that is read
        
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips) # torch.Size([1, 464, 7, 7])
        #print(feat_3d.size())
        
        # 2D backbone
        #reg_feats, obj_feats, cls_feats,  
        feat_2d= self.backbone_2d(key_frame)
        reg_feats = feat_2d['pred_reg']
        obj_feats = feat_2d['pred_obj']
        cls_feats = feat_2d['pred_cls']
        
        # non-shared heads
        all_conf_preds = []
        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        # outputs = {"pred_reg": all_reg_preds,       # List(Tensor) [B, M, 4]
        #                "pred_obj": all_obj_preds,   # List(Tensor) [B, M, 1]
        #                "pred_cls": all_cls_preds,   # List(Tensor) [B, M, numclass]
        #           }
        
        for level, (reg_feat, obj_feat, cls_feat) in enumerate(zip(reg_feats, obj_feats, cls_feats)):
            #upsample
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            print(math.sqrt(cls_feat.size(1)))
            
            cls_feat = cls_feat.reshape(-1,int(math.sqrt(cls_feat.size(1))),int(math.sqrt(cls_feat.size(1))),cls_feat.size(2)).permute(0, 3, 1, 2)
            # encoder
            print("cls_feat:",cls_feat.size()) #torch.Size([1, 24, 28, 28])
            print("feat_3d_up:",feat_3d_up.size()) #torch.Size([1, 464, 28, 28])
            cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up) #torch.Size([1, 64, 28, 28])
            
            # head
            cls_feat = self.heads[level](cls_feat) #torch.Size([1, 64, 28, 28])
            
            # pred
            cls_pred = self.cls_preds[level](cls_feat) #torch.Size([1, 24, 28, 28])
            
            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            
            print("----cls_pred",cls_pred.size())
            
            print("level:",level)
            
            # generate anchors
            fmp_size = obj_feat.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            print("fmp_size",fmp_size)
            print(self.stride[level])
            print("anchors",anchors.size())
            
            all_conf_preds.append(obj_feat)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(reg_feat)
            all_anchors.append(anchors)
        '''
        outputs = {"pred_conf": all_conf_preds,       # List(Tensor) [B, M, 1]     []
                       "pred_cls": all_cls_preds,         # List(Tensor) [B, M, C] [Done]
                       "pred_box": all_box_preds,         # List(Tensor) [B, M, 4] [Done]
                       
                       "anchors": all_anchors,            # List(Tensor) [B, M, 2]
                       "strides": self.stride}            # List(Int)
        '''
        outputs = {"pred_conf": all_conf_preds,       # List(Tensor) [B, M, 1]     []
                   "pred_cls": all_cls_preds,         # List(Tensor) [B, M, C] [Done]
                   "pred_box": all_box_preds,         # List(Tensor) [B, M, 4] [Done]
                   "anchors": all_anchors,            # List(Tensor) [B, M, 2]
                   "strides": self.stride}            # List(Int)
        
        return outputs
    
if __name__ == "__main__":
    model = YOWO(cfg)
    input_var = Variable(torch.randn(1, 3, 16, 224, 224))
    outputs = model(input_var)
    
    print("outputs.shape",outputs)

'''
    python yowov2.py
'''