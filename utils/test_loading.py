import torch
from checkpoint import save_checkpoint, load_checkpoint, load_only_model_from_checkpoint
from trainer_scripts.train_common import checkpoint_path
from models.darknet53 import Darknet53
from models.yolo_v3 import YoloV3, DetectorHead
import os

def save_darknet():
    d= Darknet53()
    optimizer = torch.optim.Adam(d.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
    save_checkpoint(0, d, optimizer,  os.path.join(checkpoint_path, "darknet.tar"), 0, scheduler)

def save_head():
    d = DetectorHead(80)
    optimizer = torch.optim.Adam(d.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
    save_checkpoint(0, d, optimizer, os.path.join(checkpoint_path, "head.tar"), 0, scheduler)

def load_yolo():
    d = Darknet53()
    h = DetectorHead(80)

    d = load_only_model_from_checkpoint(os.path.join(checkpoint_path, "darknet.tar"), d)
    h = load_only_model_from_checkpoint(os.path.join(checkpoint_path, "head.tar"), h)

    y = YoloV3(80, d)
    y.head = h

if __name__ =="__main__":
    save_head()
    save_darknet()
    load_yolo()
