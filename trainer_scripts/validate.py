from dataset_loader.imagenet_loader import get_imagenet_torch_dataloaders
from models.pretrain_model import ImageNetClassifier
from models.stolen_darknet53 import StolenDarknet53, darknet53
from utils.checkpoint import save_checkpoint, load_checkpoint, load_only_model_from_checkpoint
import torch
from torchvision import transforms
import logging
from tqdm import tqdm
from huggingface_hub import login
import os

cuda_id = 1
checkpoint_path = "./trained_models"
model_name = "pretrained_darknet.pth.tar"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def pretrain_validation_loop(val_lodaer, model, preprocess):
    res = []
    model.eval()
    loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, val in tqdm(enumerate(val_lodaer)):
            data = val['image']
            label = val['label'].cuda(cuda_id)
            output = model(data.cuda(cuda_id))
            loss += loss_fn(output, label)
            res.append(int(label == torch.argmax(torch.sigmoid(output))))

    accuracy = torch.mean(torch.Tensor(res))
    print(f"avg loss: {loss / len(res)}")
    model.train()
    return accuracy

if __name__=="__main__":
    # login(os.environ['HUGGINGFACE_TOKEN'])

    _, val_loader, _ = get_imagenet_torch_dataloaders(24, False, 4)
    print("loaders done")
    model = darknet53(1000).cuda(cuda_id).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_preprocess = transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            normalize,
        ])

    epoch_start = 0
    iter_start = 0

    checkpoint_ret = load_only_model_from_checkpoint(os.path.join(checkpoint_path, model_name), model)
    # if checkpoint_ret is not None:
    #     epoch, model, optimizer, iter_start = checkpoint_ret
    #     print(f"loaded checkpoint from {os.path.join(checkpoint_path, model_name)}")
    #     print(f"Continue training from {epoch} epoch, iter {iter_start}")

    print("running validate")

    val_acc = pretrain_validation_loop(val_loader, model, val_preprocess)
    print(f"validation accuracy: {val_acc * 100} %")
