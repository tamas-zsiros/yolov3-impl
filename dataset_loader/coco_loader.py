import torchvision.datasets as dset
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import numpy as np
import torch
import collections.abc
import re
from .coco_cat_map import get_non_ordered_map

import torch
from tensordict import TensorDict

if '1.7.1' in torch.__version__:
    from torch.utils.data._utils.collate import *  # noqa: F401, F403
else:
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    int_classes = int
    string_classes = (str, bytes)
    container_abcs = collections.abc


path2train_data="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/train2017"
path2val_data="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/val2017"
path2test_data="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/test017"

path2train_json="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/annotations_trainval2017/annotations/instances_train2017.json"
path2val_json="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/annotations_trainval2017/annotations/instances_val2017.json"

def custom_collate(batch):
    r"""
    Upgraded copy of default collate of version 1.7.1 to handle non tensor objects.
    Puts each data field into a tensor OR LIST with outer dimension batch size.

    Note: May be advantageous to update with subsequent versions.

    Update1: Updated some part with 1.11.1 version to mute a deprecation warning.
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError('Could not collate type : {}.'.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, TensorDict):
        return torch.stack(batch, dim=0)
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    # elif isinstance(elem, container_abcs.Sequence):
    #     # check to make sure that the elements in batch have consistent size
    #     it = iter(batch)
    #     elem_size = len(next(it))
    #     if not all(len(elem) == elem_size for elem in it):
    #         raise RuntimeError('each element in list of batch should be of equal size')
    #     transposed = zip(*batch)
    #     return [custom_collate(samples) for samples in transposed]

    return batch



def _get_single_dataset(data_path: str, json_path:str) -> dset:
    return dset.CocoDetection(root = data_path,
                                annFile = json_path)

class CustomCocoDataset(Dataset):
    def __init__(self, data: dset):
        super().__init__()
        self.data = data
        self.resize = Resize([416, 416], antialias=True)
        self.id_map = get_non_ordered_map()

    def __getitem__(self, item):
        image, target = self.data[item]
        # pad the image, so objects don't get distorted from the resize
        padded_img = np.array(image)
        h, w, _ = padded_img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded_img = np.pad(padded_img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = padded_img.shape
        d = {'image': Tensor(padded_img), 'target': target}
        shape = d['image'].shape
        if d['image'].ndim == 2:
            d['image'] = d['image'].repeat(3, 1, 1).permute(1, 2, 0)
        elif d['image'].shape[2] == 4:
            d['image'] = d['image'][:, :, :3]
        d['image'] = self.resize(d['image'].permute(2, 0, 1))
        shape = d['image'].shape
        for i in range(len(d['target'])):
            bbox = d['target'][i]['bbox']
            # adjust padding
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = (bbox[0] + bbox[2])
            y2 = (bbox[1] + bbox[3])

            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            # pre adjust for training -> which grid and w/h according to grid size
            scale = [shape[1] / padded_w, shape[2] / padded_h]
            center_x = ((x1 + x2) / 2) * scale[0] / shape[1]
            center_y = ((y1 + y2) / 2) * scale[1] / shape[2]
            bbox_w = (x2 - x1) * scale[0] / shape[1]
            bbox_h = (y2 - y1) * scale[1] / shape[2]

            d['target'][i]['bbox'] = [center_x, center_y, bbox_w, bbox_h]
            # annotations are not mapped from 0-79, so convert it
            d['target'][i]['category_id'] = self.id_map[str(d['target'][i]['category_id'])]
        return d

    def __len__(self):
        return len(self.data)


def get_coco_loader(batch_size, shuffle, workers):
    train_torch_loader = DataLoader(CustomCocoDataset(_get_single_dataset(path2train_data, path2train_json)), batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=custom_collate)
    val_torch_loader = DataLoader(CustomCocoDataset(_get_single_dataset(path2val_data, path2val_json)), batch_size=1, shuffle=False, num_workers=workers)
    # test_torch_loader = DataLoader(_get_single_dataset(path2test_data, path2test_json), batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_torch_loader, val_torch_loader#, test_torch_loader


if __name__ == "__main__":
    coco_train, _ = get_coco_loader(1, False, 4)

    print('Number of samples: ', len(coco_train))
    d = coco_train.dataset[69999]
    img, target = d['image'], d['target']
    print (img.size)
    print(target)
    import cv2
    converted = img.permute(1,2,0).numpy().astype(np.uint8).copy()
    for t in target:
        bbox = t["bbox"]
        converted = cv2.rectangle(converted,
                                  [int((bbox[0] - bbox[2] / 2) * 416), int((bbox[1] - bbox[3] / 2) * 416)],
                                  [int((bbox[0] + bbox[2] / 2) * 416), int((bbox[1] + bbox[3] / 2) * 416)], [255, 138, 86], 1)

    cv2.imshow("example", converted)
    cv2.waitKey()
