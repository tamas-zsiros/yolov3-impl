import torch.utils.data
from torchvision.transforms import Resize
from torch.utils.data import BatchSampler, DataLoader
from .imagenet_1k import Imagenet1k
import datasets

def get_generators():
    conf = datasets.DownloadConfig(resume_download=True, use_auth_token=True)
    # dl_manager = datasets.DownloadManager("ImageNet1K", "E:\AI-Nanodegree\imagenet-1k\data", conf)
    handler = Imagenet1k()

    handler.download_and_prepare(download_config=conf)

    # datasets.load_dataset("C:\\Users\\Tamás\\.cache\\huggingface\\datasets\\imagenet_1k\\default\\1.0.0")
    
    gens = [handler.as_dataset(datasets.Split.TRAIN), handler.as_dataset(datasets.Split.VALIDATION), handler.as_dataset(datasets.Split.TEST)]
    return gens

"""
@brief
ImageNet images are different sizes, so create a custom dataset which pre-resizes them - 
so all images can be put together into a batch in the torch.Dataloader
"""
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.Dataset):
        super().__init__()
        self.data = data
        self.resize = Resize([416, 416], antialias=True)

    def __getitem__(self, item):
        d = self.data[item]
        if d['image'].ndim == 2:
            d['image'] = d['image'].repeat(3, 1, 1).permute(1, 2, 0)
        d['image'] = self.resize(d['image'].permute(2, 0, 1))
        return d

    def __len__(self):
        return len(self.data)


def get_imagenet_torch_dataloaders(batch_size, shuffle, workers):
    train_gen, val_gen, test_gen = get_generators()
    train_torch_loader = DataLoader(CustomDataset(train_gen.with_format("torch")), batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    val_torch_loader = DataLoader(CustomDataset(val_gen.with_format("torch")), batch_size=1, shuffle=False, num_workers=workers)
    test_torch_loader = DataLoader(CustomDataset(test_gen.with_format("torch")), batch_size=1, shuffle=False, num_workers=workers)

    return train_torch_loader, val_torch_loader, test_torch_loader


if __name__ == "__main__":
    a, b, c = get_imagenet_torch_dataloaders(16, True, 1)

    print(f"{a} \n {b} \n {c}")

