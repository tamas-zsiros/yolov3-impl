import logging

import numpy as np
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

        self.class_loss_criterion = torch.nn.BCELoss(reduction='sum')
        self.obj_loss_criterion = torch.nn.BCELoss(reduction='sum')

        self.anchors = torch.Tensor(self.anchors)

    def reconstruct_batch(self, gt_batch, layer_number):
        batch_number = gt_batch.size(0)
        number_of_anchors = self.anchors[layer_number].size(0)
        number_of_classes = self.class_number
        number_of_grids = self.grid_numbers[layer_number]
        mask = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        tx = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        ty = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        tw = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        th = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        tobj = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids)
        tclass = torch.zeros(batch_number, number_of_anchors, number_of_grids, number_of_grids, number_of_classes)
        used_anchors = self.anchors[layer_number] / self.grid_sizes[layer_number]

        for b in range(batch_number):
            for t in range(gt_batch.shape[1]):
                if gt_batch[b, t].sum() == 0:
                    continue

                gx = gt_batch[b, t, 1] * number_of_grids
                gy = gt_batch[b, t, 2] * number_of_grids
                gw = gt_batch[b, t, 3] * number_of_grids
                gh = gt_batch[b, t, 4] * number_of_grids
                gi = int(gx)
                gj = int(gy)
                gt_box = torch.zeros([1, 4], device=gw.device, dtype=torch.float32)
                gt_box[:, 2] = gw
                gt_box[:, 3] = gh

                anchor_shapes = torch.Tensor(
                    np.concatenate((np.zeros((len(used_anchors), 2)), np.array(used_anchors)), 1)).cuda(gt_box.device)

                anchor_ious = box_iou(gt_box, anchor_shapes)

                best_n = torch.argmax(anchor_ious)

                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw / used_anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / used_anchors[best_n][1] + 1e-16)
                # label and confidence
                target_label = int(gt_batch[b, t, 0])
                tclass[b, best_n, gj, gi, target_label] = 1
                tobj[b, best_n, gj, gi] = 1

        mask = mask.type(torch.bool).cuda(gt_batch.device)
        tx = tx.cuda(gt_batch.device)
        ty = ty.cuda(gt_batch.device)
        tw = tw.cuda(gt_batch.device)
        th = th.cuda(gt_batch.device)
        tobj = tobj.cuda(gt_batch.device)
        tclass = tclass.cuda(gt_batch.device)

        return mask, tx, ty, tw, th, tobj, tclass

    def __call__(self, predictions: List[torch.Tensor], gt_batch, verbose=False) -> Dict:
        # predictions: layer x batch x (nr_classes+5) x N x N
        loss = {'class_loss': 0.0, 'bbox_loss': {'x': 0.0, 'y': 0.0, 'w': 0.0, 'h': 0.0}, 'obj_loss': 0.0}
        obj_counter = 0
        bbox_counter = 0

        # for i, l in enumerate(predictions):     # per layer
        #
        #
        # for batch, gt in enumerate(gt_batch):
        #     predicted_bboxes.append([])
        #     if not gt:
        #         continue
        #     gt_boxes = []
        #     class_ids = []
        #     if isinstance(gt[0], list):
        #         gt = gt[0]
        #     for t in gt:
        #         gt_boxes.append(t['bbox'])
        #         class_ids.append(t['category_id'])
        #
        #     gt_boxes = torch.Tensor(gt_boxes)

        for i, layer in enumerate(predictions):

            mask, tx, ty, tw, th, tobj, tclass = self.reconstruct_batch(gt_batch, i)

            transformed_pred = layer.view(gt_batch.size(0), self.anchors[i].size(0), self.class_number + 5, self.grid_numbers[i], self.grid_numbers[i]).permute(0, 1, 3, 4, 2)

            pred_centerx = transformed_pred[..., -4].sigmoid()
            pred_centery = transformed_pred[..., -3].sigmoid()
            pred_w = transformed_pred[..., -2]
            pred_h = transformed_pred[..., -1]

            pred_class = transformed_pred[..., 1:self.class_number+1].sigmoid()
            pred_obj = transformed_pred[..., 0].sigmoid()

            loss['bbox_loss']['x'] += torch.nn.functional.mse_loss(pred_centerx[mask], tx[mask])
            loss['bbox_loss']['y'] += torch.nn.functional.mse_loss(pred_centery[mask], ty[mask])
            loss['bbox_loss']['w'] += torch.nn.functional.mse_loss(pred_w[mask], tw[mask])
            loss['bbox_loss']['h'] += torch.nn.functional.mse_loss(pred_h[mask], th[mask])
            loss['class_loss'] += self.class_loss_criterion(pred_class[mask], tclass[mask])
            loss['obj_loss'] += self.obj_loss_criterion(pred_obj, tobj)

            # # obj loss occurs in all grids
            # objectness_gt = torch.zeros((gt_batch.size[0], 3, predictions[i].shape[2], predictions[i].shape[2]),
            #                             dtype=torch.float32).cuda(cuda_id)
            # objectness_mask = torch.ones((gt_batch.size[0], 3, predictions[i].shape[2], predictions[i].shape[2]),
            #                             dtype=torch.bool).cuda(cuda_id)
            # for class_id, gt_box in zip(class_ids, gt_boxes):
            #     gt_center_x = gt_box[0] * self.grid_numbers[i]
            #     gt_center_y = gt_box[1] * self.grid_numbers[i]
            #     gt_w = gt_box[2] * self.grid_numbers[i]
            #     gt_h = gt_box[3] * self.grid_numbers[i]
            #
            #     grid_i, grid_j = int(gt_center_x), int(gt_center_y)
            #     gt_box_t = torch.FloatTensor([0, 0, gt_w, gt_h]).unsqueeze(0)
            #     anchor_boxes = torch.FloatTensor(np.concatenate(
            #         [np.zeros((len(self.anchors[i]), 2)), np.array(self.anchors[i] / self.grid_sizes[i])], 1
            #     ))
            #     ious = box_iou(anchor_boxes, gt_box_t)
            #     max_iou, max_index = torch.max(ious, dim=0)
            #
            #     # offset from the top left corner
            #     tx = (gt_center_x - grid_i).cuda(cuda_id)
            #     ty = (gt_center_y - grid_j).cuda(cuda_id)
            #     # gt box w/h is also in ratio to the img size -> scale the boxes according to the grid size
            #     tw = torch.log(gt_w / (self.anchors[i][max_index][:, 0] / self.grid_sizes[i]) + 1e-9).cuda(cuda_id)
            #     th = torch.log(gt_h / (self.anchors[i][max_index][:, 1] / self.grid_sizes[i]) + 1e-9).cuda(cuda_id)
            #
            #     prediction_index = max_index * 85, (max_index + 1) * 85 if max_index != 2 else -1
            #
            #     used_predictions = l[:, prediction_index[0]: prediction_index[1], grid_i, grid_j]
            #
            #     pred_centerx = used_predictions[-4].sigmoid()
            #     pred_centery = used_predictions[-3].sigmoid()
            #     pred_w = used_predictions[-2]
            #     pred_h = used_predictions[-1]
            #
            #     loss['bbox_loss']['x'] += torch.nn.functional.mse_loss(pred_centerx, tx)
            #     loss['bbox_loss']['y'] += torch.nn.functional.mse_loss(pred_centery, ty)
            #     loss['bbox_loss']['w'] += torch.nn.functional.mse_loss(pred_w, tw.squeeze())
            #     loss['bbox_loss']['h'] += torch.nn.functional.mse_loss(pred_h, th.squeeze())
            #
            #     objectness_gt[0, max_index, grid_i, grid_j] = 1
            #     for iou_index in range(len(ious)):
            #         if iou_index == max_index:
            #             continue
            #         if ious[iou_index] > 0.5:
            #             objectness_mask[0, iou_index, grid_i, grid_j] = False
            #
            #     classes_gt = torch.zeros((1, self.class_number), dtype=torch.float32).cuda(cuda_id)
            #     classes_gt[0, class_id] = 1.0
            #     loss['class_loss'] += self.class_loss_criterion(used_predictions[1:self.class_number+1].sigmoid(), classes_gt.squeeze())
            #     bbox_counter += 1
            #
            # loss['obj_loss'] += self.obj_loss_criterion(torch.flatten(l[:, 0:255:85]).sigmoid()[torch.flatten(objectness_mask)], torch.flatten(objectness_gt[objectness_mask]))
            # obj_counter += 1

        # if obj_counter > 0:
        #     loss['obj_loss'] /= obj_counter
        # if bbox_counter > 0:
        #     loss['bbox_loss']['x'] /= bbox_counter
        #     loss['bbox_loss']['y'] /= bbox_counter
        #     loss['bbox_loss']['w'] /= bbox_counter
        #     loss['bbox_loss']['h'] /= bbox_counter
        #     loss['class_loss'] /= bbox_counter

        return loss
