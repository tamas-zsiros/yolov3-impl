import logging

import torch
from torchvision.ops import box_iou
from models.yolo_v3 import DetectorHead
from typing import List, Dict
from trainer_scripts.train_common import cuda_id

class BboxLoss:
    def __init__(self, class_number: int = 80, res: int = 416):
        self.class_number = class_number

        self.anchors = DetectorHead.anchors

        self.grid_sizes = [8, 16, 32]
        self.grid_numbers = [res // self.grid_sizes[0], res // self.grid_sizes[1], res // self.grid_sizes[2]]

        self.priori_bboxes = []  # top left x,y and w,h
        self.priori_index_borders = []
        self.grid_indices = []

        for nr, size in enumerate(self.grid_sizes):
            scaled_prioris = []
            grid_indices = []
            for a in self.anchors[nr]:
                for i in range(self.grid_numbers[nr]):
                    for j in range(self.grid_numbers[nr]):
                        scaled_prioris.append([i * size, j * size, i * size + a[0], j * size + a[1]])
                        grid_indices.append([nr * (class_number + 5), i, j])
            self.priori_bboxes.append(torch.Tensor(scaled_prioris))
            self.grid_indices.append(torch.Tensor(grid_indices).type(torch.long))

        self.priori_index_borders.append(len(self.priori_bboxes[0]))
        self.priori_index_borders.append(self.priori_index_borders[-1] + len(self.priori_bboxes[1]))
        self.priori_index_borders.append(self.priori_index_borders[-1] + len(self.priori_bboxes[2]))
        self.priori_bboxes = torch.concat(self.priori_bboxes)
        self.grid_indices = torch.concat(self.grid_indices)

        self.class_loss_criterion = torch.nn.BCEWithLogitsLoss()
        self.obj_loss_criterion = torch.nn.BCEWithLogitsLoss()

        self.anchors = torch.Tensor(self.anchors)

    def __call__(self, predictions: List[torch.Tensor], gt_batch, verbose = False) -> Dict:
        # predictions: layer x batch x (nr_classes+5) x N x N
        loss = {'class_loss': 0.0, 'bbox_loss': 0.0, 'obj_loss': 0.0}
        obj_counter = 0
        bbox_counter = 0

        printed_one_batch = False
        for batch, gt in enumerate(gt_batch):
            if not gt:
                continue
            gt_boxes = []
            for t in gt:
                gt_boxes.append(t['bbox'])

            gt_boxes = torch.Tensor(gt_boxes)

            ious = box_iou(self.priori_bboxes, gt_boxes)

            max_ious, max_indices = torch.max(ious, dim=0)
            mask = torch.all(ious < 0.5, dim=1)   # ignore overlaps that are more than 0.5 but not the max
            mask[max_indices] = True

            transformed_indices = [[], [], []]   # layer and index within it
            non_transformed_indices = [[], [], []]

            small_objectness = torch.zeros((1, 3, predictions[0].shape[2], predictions[0].shape[2]), dtype=torch.float32).cuda(cuda_id)
            med_objectness = torch.zeros((1, 3, predictions[1].shape[2], predictions[1].shape[2]), dtype=torch.float32).cuda(cuda_id)
            big_objectness = torch.zeros((1, 3, predictions[2].shape[2], predictions[2].shape[2]), dtype=torch.float32).cuda(cuda_id)
            objectness_gt = [small_objectness, med_objectness, big_objectness]

            small_classes = torch.zeros((1, 3, self.class_number, predictions[0].shape[2], predictions[0].shape[2]), dtype=torch.float32).cuda(cuda_id)
            med_classes = torch.zeros((1, 3, self.class_number, predictions[1].shape[2], predictions[1].shape[2]), dtype=torch.float32).cuda(cuda_id)
            big_classes = torch.zeros((1, 3, self.class_number, predictions[2].shape[2], predictions[2].shape[2]), dtype=torch.float32).cuda(cuda_id)
            classes_gt = [small_classes, med_classes, big_classes]

            remapped_gt_boxes = [[], [], []]

            for bbox_id, ind in enumerate(max_indices):
                if gt[bbox_id]['category_id'] > 79:
                    logging.error(f"invalid category id: {gt[bbox_id]['category_id']}")
                    continue
                layer = 0 if ind < self.priori_index_borders[0] else 1 if ind < self.priori_index_borders[1] else 2
                transformed_indices[layer].append(self.grid_indices[ind])
                non_transformed_indices[layer].append(ind)
                i,j,k = self.grid_indices[ind]
                objectness_gt[layer][0, int(i // (self.class_number + 5)), j, k] = 1.0
                classes_gt[layer][0, int(i // (self.class_number + 5)), gt[bbox_id]['category_id'], j, k] = 1.0
                gt_boxes[bbox_id][2] -= gt_boxes[bbox_id][0]  # width
                gt_boxes[bbox_id][3] -= gt_boxes[bbox_id][1]  # height
                remapped_gt_boxes[layer].append(gt_boxes[bbox_id].cuda(cuda_id))

            for i, l in enumerate(predictions):
                for bbox_id, feature_map_index in enumerate(transformed_indices[i]):
                    used_predictions = l[batch, feature_map_index[0]: feature_map_index[0] + 1 * 85 if feature_map_index[0] != 2 else -1,
                                       feature_map_index[1], feature_map_index[2]]
                    offsets = self.priori_bboxes[non_transformed_indices[i][bbox_id]][:2]
                    priori_w = self.priori_bboxes[non_transformed_indices[i][bbox_id]][2] - self.priori_bboxes[non_transformed_indices[i][bbox_id]][0]
                    priori_h = self.priori_bboxes[non_transformed_indices[i][bbox_id]][3] - self.priori_bboxes[non_transformed_indices[i][bbox_id]][1]

                    # pred_centerx = used_predictions[-4].sigmoid() * self.grid_sizes[i] + offsets[0]
                    # pred_centery = used_predictions[-3].sigmoid() * self.grid_sizes[i] + offsets[1]
                    pred_centerx = used_predictions[-4].sigmoid()
                    pred_centery = used_predictions[-3].sigmoid()
                    # pred_w = priori_w * torch.exp(used_predictions[-2])
                    pred_w = torch.exp(used_predictions[-2])
                    # pred_h = priori_h * torch.exp(used_predictions[-1])
                    pred_h = torch.exp(used_predictions[-1])

                    # pred_x = pred_centerx - pred_w / 2
                    # pred_y = pred_centery - pred_h / 2
                    # pred_x2 = pred_centerx + pred_w / 2
                    # pred_y2 = pred_centery + pred_h / 2

                    gt_center_x = remapped_gt_boxes[i][bbox_id][0] + remapped_gt_boxes[i][bbox_id][2]/ 2.0
                    gt_center_y = remapped_gt_boxes[i][bbox_id][1] + remapped_gt_boxes[i][bbox_id][3] / 2.0
                    gt_x = (gt_center_x - offsets[0]) / priori_w
                    gt_y = (gt_center_y - offsets[1]) / priori_h

                    gt_w = remapped_gt_boxes[i][bbox_id][2] / priori_w
                    gt_h = remapped_gt_boxes[i][bbox_id][3] / priori_h

                    predicted_bboxes = torch.unsqueeze(torch.FloatTensor([pred_centerx, pred_centery, pred_w, pred_h]).cuda(cuda_id),dim=0)
                    # used_gt_box = torch.unsqueeze(remapped_gt_boxes[i][bbox_id], dim=0)
                    transformed_gt_bboxes = torch.FloatTensor([gt_x, gt_y, gt_w, gt_h]).cuda(cuda_id)
                    if verbose and not printed_one_batch or True:
                        logging.info(f"bbox loss { torch.nn.functional.mse_loss(predicted_bboxes, transformed_gt_bboxes)}, \n gt box: \n\t{remapped_gt_boxes[i][bbox_id]} \n calc_box \n\t"
                              f" {[pred_centerx * pred_w + offsets[0], pred_centery * pred_h + offsets[1], pred_w * priori_w, pred_h * priori_h]}")
                    loss['bbox_loss'] += torch.nn.functional.mse_loss(predicted_bboxes, transformed_gt_bboxes)

                    # loss['bbox_loss'] += 1.0 - box_iou(predicted_bboxes, used_gt_box).squeeze()
                    cl = 0.0

                    loss['class_loss'] += self.class_loss_criterion(used_predictions[1:self.class_number+1],
                                                        classes_gt[i][0, int(feature_map_index[0] // (self.class_number + 5)), :, feature_map_index[1], feature_map_index[2]])
                    bbox_counter +=1
                mask_start = self.priori_index_borders[i - 1] if i > 0 else 0
                mask_end = self.priori_index_borders[i]
                loss['obj_loss'] += self.obj_loss_criterion(torch.flatten(l[batch, 0:255:85])[mask[mask_start:mask_end]], torch.flatten(objectness_gt[i])[mask[mask_start:mask_end]])
                obj_counter +=1

            printed_one_batch = True

        if obj_counter > 0:
            loss['obj_loss'] /= obj_counter
        if bbox_counter > 0:
            loss['bbox_loss'] /= bbox_counter
            loss['class_loss'] /= bbox_counter

        return loss


