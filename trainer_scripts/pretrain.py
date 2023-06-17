from dataset_loader.imagenet_loader import get_imagenet_torch_dataloaders
from models.pretrain_model import ImageNetClassifier
from utils.checkpoint import save_checkpoint, load_checkpoint
from train_common import inner_train_loop, cuda_id, setup_logger, checkpoint_path
import torch
from torchvision import transforms
import logging
from tqdm import tqdm
from huggingface_hub import login
import os

overfit = False
model_name = "pretrained_imagenet_downsized.tar"


def pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, preprocess, epoch, iter_start, scheduler):
    for i, val in enumerate(train_loader):
        if i < iter_start:
            continue
        batch = []
        for b in val['image']:
            batch.append(preprocess(b))
        batch = torch.stack(batch)
        loss = inner_train_loop(batch, val['label'], model, optimizer, loss_fn)

        if overfit and i > 1000:
            logging.info(f"breaking in train because script is in overfit mode")
            break

        if i % (200 if overfit else 2000) == 0:
            logging.info(f"loss at {i} iteration: {loss}")

        if i % (800 if overfit else 8000) == 0:
            save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), i, scheduler)

def pretrain_validation_loop(val_lodaer, model, preprocess):
    res = []
    model.eval()
    loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, val in tqdm(enumerate(val_lodaer)):
            data = val['image']
            label = val['label'].cuda(cuda_id)
            output = model(preprocess(data.cuda(cuda_id)))
            loss += loss_fn(output, label)
            res.append(int(label == torch.argmax(torch.sigmoid(output))))
            if overfit and i > 100:
                logging.info(f"breaking in train because script is in overfit mode")
                break

    accuracy = torch.mean(torch.Tensor(res))
    print(f"avg loss: {loss / len(res)}")
    model.train()
    return accuracy

if __name__ == "__main__":
    login(os.environ['HUGGINGFACE_TOKEN'])
    setup_logger("pretrain.log")
    train_loader, val_loader, _ = get_imagenet_torch_dataloaders(75, False, 4)

    model = ImageNetClassifier().cuda(cuda_id).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs // 5)

    train_preprocess = transforms.Compose(
        [
         # transforms.RandomResizedCrop(224),
         transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
         # transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    val_preprocess = transforms.Compose(
        [
         # transforms.Resize(256),
         # transforms.CenterCrop(224),
         # transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_start = 0
    iter_start = 0
    checkpoint_ret = load_checkpoint(epoch_start, model, optimizer, os.path.join(checkpoint_path, model_name), iter_start, scheduler)
    if checkpoint_ret is not None:
        epoch_start, model, optimizer, iter_start, scheduler = checkpoint_ret
        logging.info(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")
        logging.info(f"Continue training from {epoch_start} epoch, iter {iter_start}")

    for epoch in range(epoch_start, num_epochs):
        pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, train_preprocess, epoch, iter_start, scheduler)
        val_acc = pretrain_validation_loop(val_loader, model, val_preprocess)
        logging.info(f"validation accuracy in epoch {epoch}: {val_acc * 100} %")
        save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), 0, scheduler)
        iter_start = 0
        scheduler.step()
        # if overfit:
        #     train_acc = pretrain_validation_loop(train_loader, model, val_preprocess)
        #     logging.info(f"train accuracy in epoch {epoch}: {train_acc * 100} %")
