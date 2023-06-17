import torch
import logging
from train_common import setup_logger, cuda_id, inner_train_loop, checkpoint_path
from utils.checkpoint import save_checkpoint, load_checkpoint, load_only_model_from_checkpoint
import os
from tqdm import tqdm
from dataset_loader.coco_loader import get_coco_loader
from models.yolo_v3 import YoloV3
from models.darknet53 import Darknet53
from torchvision import transforms
from utils.bbox_loss import BboxLoss

model_name = "yolo_first_try.tar"
overfit = False

def yolo_epoch_loop(train_loader, model, optimizer, loss_fn, preprocess, epoch, iter_start, scheduler):
    for i, val in enumerate(train_loader):
        if i < iter_start:
            continue
        verbose = i % 500 == 0
        batch = []
        for b in val['image']:
            batch.append(preprocess(b))
        batch = torch.stack(batch)
        if not val['target']:
            continue
        loss = inner_train_loop(batch, val['target'], model, optimizer, loss_fn)

        if overfit and i > 10:
            logging.info(f"breaking in train because script is in overfit mode")
            break

        if verbose:
            msg = "\n"
            for key, val in loss.items():
                msg += f"\t {key}: {val} \n"
            logging.info(f"loss at {i} iteration: {msg}")

        if i % (len(train_loader) / 4) == 0:
            save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), i, scheduler)

def yolo_validation_loop(val_lodaer, model, preprocess):
    res = []
    model.eval()
    loss = 0.0
    loss_fn = BboxLoss(80)
    with torch.no_grad():
        for i, val in tqdm(enumerate(val_lodaer)):
            data = val['image']
            output = model(preprocess(data.cuda(cuda_id)))
            loss += sum(loss_fn(output, [val['target']]).values())
            res.append(1)
            if overfit and i > 10:
                break

    model.train()
    return loss / len(res)

if __name__ == "__main__":
    setup_logger("yolo_train.log")
    train_loader, val_loader = get_coco_loader(80, False, 4)

    backbone = Darknet53()
    model = YoloV3(80, backbone).cuda(cuda_id).train()
    model = load_only_model_from_checkpoint(os.path.join(checkpoint_path, "pretrained_imagenet.tar"), model)
    for param in model.backbone.parameters():
        param.requires_grad = False

    if model is None:
        logging.error(f"failed to load backbone from {os.path.join(checkpoint_path, 'pretrained_imagenet.tar')}")
        exit(10)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

    train_preprocess = transforms.Compose(
        [
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    val_preprocess = transforms.Compose(
        [
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    loss_fn = BboxLoss()

    epoch_start = 0
    iter_start = 0
    checkpoint_ret = load_checkpoint(epoch_start, model, optimizer, os.path.join(checkpoint_path, model_name), iter_start, scheduler)
    if checkpoint_ret is not None:
        epoch, model, optimizer, iter_start, scheduler = checkpoint_ret
        logging.info(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")
        logging.info(f"Continue training from {epoch} epoch, iter {iter_start}")

    for epoch in range(epoch_start, num_epochs):
        yolo_epoch_loop(train_loader, model, optimizer, loss_fn, train_preprocess, epoch, iter_start, scheduler)
        val_acc = yolo_validation_loop(val_loader, model, val_preprocess)
        logging.info(f"validation accuracy in epoch {epoch}: {val_acc } ")
        save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), 0, scheduler)
        iter_start = 0
        scheduler.step()

    save_checkpoint(epoch, model, optimizer, os.path.join(checkpoint_path, model_name), 0, scheduler)
