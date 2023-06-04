import torch
from models.yolo_v3 import DetectorHead
from typing import List, Dict

class BboxLoss:
    def __init__(self, class_number: int = 80, res: int = 416):
        self.class_number = class_number

        anchors = DetectorHead.anchors

        self.grid_sizes = [32, 16, 8]
        self.grid_numbers = [res // self.grid_sizes[0], res // self.grid_sizes[1], res // self.grid_sizes[2]]

        self.priori_bboxes = []  # top left x,y and w,h
        self.priori_index_borders = []

        for nr, size in enumerate(self.grid_sizes):
            scaled_prioris = []
            for i in range(self.grid_numbers[nr]):
                for j in range(self.grid_numbers[nr]):
                    for a in anchors[nr]:
                        scaled_prioris.append([i * size, j * size, *a])
            self.priori_bboxes.append(torch.Tensor(scaled_prioris))

        self.priori_index_borders.append(len(self.priori_bboxes[0]))
        self.priori_index_borders.append(self.priori_index_borders[-1] + len(self.priori_bboxes[1]))
        self.priori_index_borders.append(self.priori_index_borders[-1] + len(self.priori_bboxes[2]))
        self.priori_bboxes = torch.concat(self.priori_bboxes)

        self.class_loss_criterion = torch.nn.CrossEntropyLoss()
        self.obj_loss_criterion = torch.nn.BCEWithLogitsLoss()

    def IoU(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        # top left corner more to the right and to the bottom
        inter_top_left = [max(bbox1[0], bbox2[0]),
                          max(bbox1[1], bbox2[1])]
        # bottom right corner more to the left and to the top
        inter_bottom_right = [min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]),
                              min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])]
        intersection = (inter_bottom_right[0] - inter_top_left[0]) * (inter_bottom_right[1] - inter_top_left[1])

        union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection + 1e-9

        return intersection / union

    def vectorizedIoU(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        # top left corner more to the right and to the bottom
        inter_top_left = torch.Tensor([torch.max(bbox1[:, 0], bbox2[:, 0]),
                                       torch.max(bbox1[:, 1], bbox2[:, 1])])
        # bottom right corner more to the left and to the top
        inter_bottom_right = torch.Tensor([torch.min(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2]),
                                           torch.min(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])])
        intersection = (inter_bottom_right[:, 0] - inter_top_left[:, 0]) * (inter_bottom_right[:, 1] - inter_top_left[:, 1])

        union = bbox1[:, 2] * bbox1[:, 3] + bbox2[:, 2] * bbox2[:, 3] - intersection + 1e-9

        return intersection / union

    def __call__(self, predictions: List[torch.Tensor], gt) -> Dict:
        # predictions: layer x batch x N x N x (nr_classes+5)
        loss = {'class_loss': 0.0, 'bbox_loss': 0.0, 'obj_loss': 0.0}

        ious = self.vectorizedIoU(self.priori_bboxes, gt)

        max_ious, max_indices = torch.max(ious)
        mask = ious < 0.5   # ignore overlaps that are more than 0.5 but not the max
        mask[max_indices] = True

        transformed_indices = [[]] * len(predictions)   # layer and index within it
        non_transformed_indices = [[]] * len(predictions)
        for ind in max_indices:
            layer = 0 if ind < self.priori_index_borders[0] else 1 if ind < self.priori_index_borders[1] else 2
            inner_ind = self.priori_index_borders[layer] - ind if layer > 0 else ind
            transformed_indices[layer].append(inner_ind)
            non_transformed_indices[layer].append(ind)

        for i, l in enumerate(predictions):
            used_predictions = l[transformed_indices[i]]
            offsets = transformed_indices[i][:, 0] / self.grid_numbers[i] * self.grid_sizes[i],\
                      transformed_indices[i][:, 1] % self.grid_numbers[i] * self.grid_sizes
            pred_x = used_predictions[-4].sigmoid() + offsets[0]
            pred_y = used_predictions[-3].sigmoid() + offsets[1]
            pred_w = self.priori_bboxes[non_transformed_indices[i]][:, 2] * torch.exp(used_predictions[-2])
            pred_h = self.priori_bboxes[non_transformed_indices[i]][:, 3] * torch.exp(used_predictions[-1])

            predicted_bboxes = torch.concatenate([pred_x, pred_y, pred_w, pred_h], dim=1)
            loss['bbox_loss'] += torch.nn.MSELoss(predicted_bboxes - gt)

            loss['class_loss'] += self.class_loss_criterion(used_predictions[:self.class_number], gt)
            mask_start = i - 1 if i > 0 else 0
            mask_end = self.priori_index_borders[i]
            loss['obj_loss'] += self.obj_loss_criterion(l[mask[mask_start:mask_end]])

        return loss


