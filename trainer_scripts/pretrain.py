from dataset_loader.imagenet_loader import get_imagenet_torch_dataloaders
from models.pretrain_model import ImageNetClassifier
from utils.checkpoint import save_checkpoint, load_checkpoint
import torch
from torchvision import transforms
import logging
from tqdm import tqdm
from huggingface_hub import login
import os

overfit = True
cuda_id = 0
checkpoint_path = "./trained_models"
model_name = "pretrained_imagenet.tar"
def inner_train_loop(data, labels, model, optimizer, loss_fn):
    optimizer.zero_grad()

    output = model(data.cuda(cuda_id))
    loss = loss_fn(output, labels.cuda(cuda_id))
    loss.backward()

    optimizer.step()

    return loss

def pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, preprocess, epoch, iter_start):
    for i, val in enumerate(train_loader):
        if i < iter_start:
            continue
        batch = []
        for b in val['image']:
            batch.append(preprocess(b))
        batch = torch.stack(batch)
        loss = inner_train_loop(batch, val['label'], model, optimizer, loss_fn)

        if overfit and i > 100:
            logging.info(f"breaking in train because script is in overfit mode")
            break

        if i % 10 == 0:
            logging.info(f"loss at {i} iteration: {loss}")

        if i % len(train_loader) / 4 == 0:
            save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), iter_start)

def pretrain_validation_loop(val_lodaer, model, preprocess):
    res = []
    model.eval()
    with torch.no_grad():
        for i, val in tqdm(enumerate(val_lodaer)):
            data = val['image']
            label = val['label'].cuda(cuda_id)
            output = model(preprocess(data.cuda(cuda_id)))
            res.append(int(label == torch.argmax(output)))
            if overfit and i > 100:
                logging.info(f"breaking in validation because script is in overfit mode")
                break
    accuracy = torch.mean(torch.Tensor(res))
    model.train()
    return accuracy

if __name__ == "__main__":
    login(os.environ['HUGGINGFACE_TOKEN'])
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("pretrain.log")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


    train_loader, val_loader, _ = get_imagenet_torch_dataloaders(8, False, 4)

    model = ImageNetClassifier().cuda(cuda_id).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    num_epochs = 100

    train_preprocess = transforms.Compose(
        [
        #  transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
         # transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    val_preprocess = transforms.Compose(
        [
         # transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_start = 0
    iter_start = 0
    checkpoint_ret = load_checkpoint(epoch_start, model, optimizer, os.path.join(checkpoint_path, model_name))
    if checkpoint_ret is not None:
        epoch, model, optimizer, iter_start = checkpoint_ret
        logging.info(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")

    for epoch in range(epoch_start, num_epochs):
        pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, train_preprocess, epoch, iter_start)
        val_acc = pretrain_validation_loop(val_loader, model, val_preprocess)
        logging.info(f"validation accuracy in epoch {epoch}: {val_acc * 100} %")
        save_checkpoint(epoch, model, optimizer,  os.path.join(checkpoint_path, model_name), 0)
        # if overfit:
        #     train_acc = pretrain_validation_loop(train_loader, model, val_preprocess)
        #     logging.info(f"train accuracy in epoch {epoch}: {train_acc * 100} %")
