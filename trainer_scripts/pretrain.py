from dataset_loader.imagenet_loader import get_imagenet_torch_dataloaders
from models.pretrain_model import ImageNetClassifier
import torch
from torchvision import transforms
import logging
from tqdm import tqdm
from huggingface_hub import login
import os

overfit = True
cuda_id = 0
def inner_train_loop(data, labels, model, optimizer, loss_fn):
    optimizer.zero_grad()

    output = model(data.cuda(cuda_id))
    loss = loss_fn(output, labels.cuda(cuda_id))
    loss.backward()

    optimizer.step()

    return loss

def pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, preprocess):
    for i, val in enumerate(train_loader):
        batch = []
        for b in val['image']:
            batch.append(preprocess(b))
        batch = torch.stack(batch)
        loss = inner_train_loop(batch, val['label'], model, optimizer, loss_fn)

        if overfit and i > 1000:
            logging.info(f"breaking in train because script is in overfit mode")
            break

        if i % 100 == 0:
            logging.info(f"loss at {i} iteration: {loss}")

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


    train_loader, val_loader, _ = get_imagenet_torch_dataloaders(16, False, 4)

    model = ImageNetClassifier().cuda(cuda_id).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    num_epochs = 100

    train_preprocess = transforms.Compose(
        [
         # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
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

    for epoch in range(num_epochs):
        pretrain_epoch_loop(train_loader, model, optimizer, loss_fn, train_preprocess)
        val_acc = pretrain_validation_loop(val_loader, model, val_preprocess)
        logging.info(f"validation accuracy in epoch {epoch}: {val_acc * 100} %")

        # if overfit:
        #     train_acc = pretrain_validation_loop(train_loader, model, val_preprocess)
        #     logging.info(f"train accuracy in epoch {epoch}: {train_acc * 100} %")
