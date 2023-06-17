from dataset_loader.imagenet_loader import get_imagenet_torch_dataloaders
from models.pretrain_model import ImageNetClassifier
from utils.checkpoint import save_checkpoint, load_checkpoint
import torch
from torchvision import transforms
import logging
from tqdm import tqdm
from huggingface_hub import login
import os

cuda_id = 1
checkpoint_path = "./trained_models"
model_name = "pretrained_imagenet.tar"

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

    accuracy = torch.mean(torch.Tensor(res))
    print(f"avg loss: {loss / len(res)}")
    model.train()
    return accuracy

if __name__=="__main__":
    login(os.environ['HUGGINGFACE_TOKEN'])

    _, val_loader, _ = get_imagenet_torch_dataloaders(24, False, 4)
    model = ImageNetClassifier().cuda(cuda_id).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    val_preprocess = transforms.Compose(
        [
         # transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

    epoch_start = 0
    iter_start = 0

    checkpoint_ret = load_checkpoint(epoch_start, model, optimizer, os.path.join(checkpoint_path, model_name), iter_start)
    if checkpoint_ret is not None:
        epoch, model, optimizer, iter_start = checkpoint_ret
        print(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")
        print(f"Continue training from {epoch} epoch, iter {iter_start}")

    val_acc = pretrain_validation_loop(val_loader, model, val_preprocess)
    print(f"validation accuracy: {val_acc * 100} %")
