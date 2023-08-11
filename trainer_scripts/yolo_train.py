import torch
import logging
from train_common import setup_logger, cuda_id, inner_train_loop, checkpoint_path
from utils.checkpoint import save_checkpoint, load_checkpoint, load_only_model_from_checkpoint
import os
from tqdm import tqdm
from dataset_loader.coco_loader import get_coco_loader
from models.yolo_v3 import YoloV3
from models.darknet53 import Darknet53
from models.stolen_darknet53 import darknet53
from torchvision import transforms
from utils.bbox_loss import BboxLoss
from eval_yolo import eval

model_name = "yolo_first_try.tar"
overfit = True
save = False
warm_start = False

def yolo_epoch_loop(train_loader, model, optimizer, loss_fn, preprocess, epoch, iter_start, scheduler):
    for i, val in enumerate(train_loader):
        print(f"running iter num: {i}")
        if i < iter_start:
            continue
        verbose = i % 25 == 0
        batch = preprocess(val['image'])
        print("before loop")
        loss = inner_train_loop(batch, val['target'], model, optimizer, loss_fn)
        print("after loop")

        if overfit and i > 50:
            logging.info(f"breaking in train because script is in overfit mode")
            break

        if verbose:
            msg = "\n"
            for key, val in loss.items():
                msg += f"\t {key}: {val} \n"
            logging.info(f"loss at {i} iteration: {msg}")

        if i % (len(train_loader) // 4) == 0 and save:
            save_checkpoint(epoch, model.head, optimizer,  os.path.join(checkpoint_path, model_name), i, scheduler)

def yolo_validation_loop(val_loader, model, preprocess):
    res = []
    model.eval()
    loss = 0.0
    loss_fn = BboxLoss(80)
    precision, recall = 0.0, 0.0
    with torch.no_grad():
        for i, val in tqdm(enumerate(val_loader)):
            data = val['image']
            output = model(preprocess(data.cuda(cuda_id)))
            l, predicted_bboxes = loss_fn(output, [val['target']])
            sum_loss = 0

            def acc_loss(loss, s):
                if isinstance(loss, dict):
                    for val in loss.values():
                        if isinstance(val, dict):
                            s = acc_loss(val, s)
                        else:
                            s += val
                else:
                    s = loss
                return s

            sum_loss = acc_loss(l, sum_loss)
            loss += sum_loss
            res.append(1)
            _, prec, rec = eval(output, val['target'], 416)
            precision += prec
            recall += rec
            if i > 500:
                break

    model.train()
    return loss / len(res), prec / len(res), rec / len(res)

if __name__ == "__main__":
    setup_logger("yolo_train.log")
    back_bone_model_name = "pretrained_darknet.pth.tar"
    train_loader, val_loader = get_coco_loader(40, False, 0)
    # if overfit:
    #     val_loader, _ = get_coco_loader(1, False, 0)

    backbone = Darknet53()
    model = YoloV3(80, backbone).cuda(cuda_id).train()
    # model = load_only_model_from_checkpoint(os.path.join(checkpoint_path, "pretrained_imagenet.tar"), model)

    backbone = darknet53(1000).cuda(cuda_id).eval()
    backbone = load_only_model_from_checkpoint(os.path.join(checkpoint_path, back_bone_model_name), backbone)
    model.backbone = backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    if backbone is None:
        logging.error(f"failed to load backbone from {os.path.join(checkpoint_path, 'pretrained_imagenet.tar')}")
        exit(10)

    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.0005)
    num_epochs = 120
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_preprocess = transforms.Compose([
        transforms.CenterCrop(416),
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),

    ])

    train_preprocess = transforms.Compose([
        transforms.CenterCrop(416),
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),

    ])

    # val_preprocess = transforms.Compose(
    #     [
    #         transforms.ConvertImageDtype(torch.float32),
    #         # transforms.Normalize(mean=[0.0, 0.0, 0.0],
    #         #                      std=[0.229, 0.224, 0.225])
    #     ]
    # )

    loss_fn = BboxLoss()

    epoch_start = 0
    iter_start = 0
    if warm_start:
        checkpoint_ret = load_checkpoint(epoch_start, model.head, optimizer, os.path.join(checkpoint_path, model_name), iter_start, scheduler)
        if checkpoint_ret is not None:
            epoch_start, model.head, optimizer, iter_start, scheduler = checkpoint_ret
            logging.info(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")
            logging.info(f"Continue training from {epoch_start} epoch, iter {iter_start}")

    skip_to_val = False

    logging.info(f"current LR: {optimizer.param_groups[0]['lr']}")

    for epoch in range(epoch_start, num_epochs):
        if not skip_to_val:
            yolo_epoch_loop(train_loader, model, optimizer, loss_fn, train_preprocess, epoch, iter_start, scheduler)
        val_loss, precision, recall = yolo_validation_loop(val_loader, model, val_preprocess)
        skip_to_val = False
        logging.info(f"validation loss in epoch {epoch}: {val_loss }, precision {precision*100}%, recall {recall * 100}% ")
        if save:
            save_checkpoint(epoch, model, optimizer, os.path.join(checkpoint_path, "yolo_with_stolen_backbone.tar"), 0,
                            scheduler)
            # save_checkpoint(epoch, model.head, optimizer,  os.path.join(checkpoint_path, model_name), 0, scheduler)
        iter_start = 0
        logging.info(f"current LR: {scheduler.get_lr()}")
        scheduler.step()

        # if epoch == 60:
        #     for param in model.backbone.parameters():
        #         param.requires_grad = False

    if save:
        save_checkpoint(epoch, model.head, optimizer, os.path.join(checkpoint_path, model_name), 0, scheduler)
