from datasets import list_dataset
from torchvision import datasets, transforms
import torch
from cfg import parser

args  = parser.parse_args()
cfg   = parser.load_config(args)

dataset = 'ucf24'

if dataset == 'ucf24':
    train_dataset = list_dataset.UCF_JHMDB_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TRAIN_FILE, dataset=dataset,
                       shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                       transform=transforms.Compose([transforms.ToTensor()]), 
                       train=True, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)
    test_dataset  = list_dataset.UCF_JHMDB_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset=dataset,
                       shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                       transform=transforms.Compose([transforms.ToTensor()]), 
                       train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

    #Mac
    # loss_module   = RegionLoss(cfg).cuda()
    # loss_module   = RegionLoss(cfg).to(device)

    # train = getattr(sys.modules[__name__], 'train_ucf24_jhmdb21')
    # test  = getattr(sys.modules[__name__], 'test_ucf24_jhmdb21')
for batch_idx,(data, target) in enumerate(train_loader):
    print("data.size()",data.size())
    print("target.size()",target.size())#torch.Size([1, 250])
    import sys
    sys.exit(0)
    #print("target.size()",target)
'''
    python main.py --cfg cfg/ucf24.yaml
'''