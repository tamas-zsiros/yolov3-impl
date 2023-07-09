import torch
import numpy as np
from torchvision.ops import box_iou
from models.yolo_v3 import DetectorHead
from train_common import cuda_id
def convert_bbox(gt_box, res):
    #from relative of image to real x1y1, x2y2
    return torch.FloatTensor([(gt_box[0] - gt_box[2] / 2) * res, (gt_box[1] - gt_box[3] / 2) * res,
            (gt_box[0] + gt_box[2] / 2) * res, (gt_box[1] + gt_box[3] / 2) * res])


def postprocess_output(output):
    grid_sizes = [8, 16, 32]
    grid_numbers = [52, 26, 13]
    pred_boxes = []
    pred_obj = []
    pred_class = []
    for l, layer in enumerate(output):
        anchors = DetectorHead.anchors[l]
        for i, anchor in enumerate(anchors):

            grid = np.arange(grid_numbers[l])
            a, b = np.meshgrid(grid, grid)

            x_offset = torch.FloatTensor(a).view(-1, 1)
            y_offset = torch.FloatTensor(b).view(-1, 1)

            x_offset = x_offset.cuda(cuda_id)
            y_offset = y_offset.cuda(cuda_id)

            prediction_index = i * 85, (i + 1) * 85 if i != 2 else -1
            used_preds = layer[0, prediction_index[0]: prediction_index[1], :, :]

            pred_x = used_preds[-4, :, :].sigmoid() + torch.reshape(x_offset, layer.shape[2:4])
            pred_y = used_preds[-3, :, :].sigmoid() + torch.reshape(y_offset, layer.shape[2:4])
            pred_x /= grid_numbers[l]
            pred_y /= grid_numbers[l]
            pred_w = torch.exp(used_preds[-2, :, :]) * (anchor[0] / grid_sizes[l]) / grid_numbers[l]
            pred_h = torch.exp(used_preds[-1, :, :]) * (anchor[1] / grid_sizes[l]) / grid_numbers[l]

            obj = used_preds[0, :, :].sigmoid().unsqueeze(0)
            class_scores = used_preds[1:81, :, :].sigmoid()

            pred_boxes.append(torch.cat([pred_y.unsqueeze(0), pred_x.unsqueeze(0), pred_w.unsqueeze(0), pred_h.unsqueeze(0)], dim=0).permute(1, 2, 0))
            pred_class.append(class_scores.permute(1, 2, 0))
            pred_obj.append(obj.permute(1, 2, 0))
    return pred_boxes, pred_class, pred_obj

def non_max_supression(boxes, classes, objectness, obj_threshold = 0.5, nms_threshold = 0.4):
    """
    algo :
        per layer
        mask low confidence detections
        get bbox corners
        get max class values per grid cell
        per class find max value bboxes, throw out bboxes with same class and high iou
    """
    ret = []
    collect_img_pred = []
    for b, c, o in zip(boxes, classes, objectness):
        confidence_mask = (o > obj_threshold).float()

        box_corners = b.new(b.shape)
        box_corners[:, :, 0] = ((b[:, :, 0] - b[:, :, 2] / 2) * 416)
        box_corners[:, :, 1] = ((b[:, :, 1] - b[:, :, 3] / 2) * 416)
        box_corners[:, :, 2] = ((b[:, :, 0] + b[:, :, 2] / 2) * 416)
        box_corners[:, :, 3] = ((b[:, :, 1] + b[:, :, 3] / 2) * 416)
        b = box_corners

        max_conf, max_conf_index = torch.max(c, dim=2)
        max_conf = max_conf.float().unsqueeze(2)
        max_conf_index = max_conf_index.float().unsqueeze(2)
        seq = (confidence_mask, b, max_conf, max_conf_index)
        image_pred = torch.cat(seq, 2)

        try:
            image_pred_ = image_pred.view(-1, 7)[torch.nonzero(image_pred.view(-1, 7)[:,0])].view(-1, 7)
        except:
            continue

        collect_img_pred.append(image_pred_)

    image_pred_ = torch.cat(collect_img_pred, dim=0)

    img_classes = torch.unique(image_pred_[:,-1])
    for class_index in img_classes:
        cls_mask = image_pred_ * (image_pred_[:, -1] == class_index).float().unsqueeze(1)
        class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
        image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

        conf_sort_index = torch.sort(image_pred_class[:, 0], descending=True)[1]
        image_pred_class = image_pred_class[conf_sort_index]
        idx = image_pred_class.size(0)

        for i in range(idx):
            # Get the IOUs of all boxes that come after the one we are looking at
            # in the loop
            try:
                ious = box_iou(image_pred_class[i].unsqueeze(0)[:, 1:5], image_pred_class[i + 1:, 1:5])
            except ValueError:
                break

            except IndexError:
                break

            # Zero out all the detections that have IoU > treshhold
            iou_mask = (ious < nms_threshold).float().unsqueeze(1)
            image_pred_class[i + 1:] *= iou_mask.reshape(1, -1).permute(1, 0).repeat(1, 7)

            # Remove the non-zero entries
            non_zero_ind = torch.nonzero(image_pred_class[:, 0]).squeeze()
            image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
        ret.append(image_pred_class)

    return ret

def eval(output, target, res, img = None, obj_th = 0.5):
    pred_boxes, pred_class, pred_obj = postprocess_output(output)
    supressed_predictions = non_max_supression(pred_boxes, pred_class, pred_obj, obj_threshold=obj_th)
    number_of_proposals = sum([len(p) for p in supressed_predictions])
    number_of_gt = 0
    number_of_true_positive = 0
    if len(target) == 0:
        return 0,0,0
    if isinstance(target[0], list):
        target = target[0]
    if img is not None:
        np_img = (img[0].cpu().permute(1, 2, 0) * 255).to(torch.uint8).numpy().copy()
    for t in target:
        converted_bbox = convert_bbox(t['bbox'], res).unsqueeze(0).cuda(cuda_id)
        class_id = t['category_id']
        number_of_gt += 1
        if img is not None:
            np_img = cv2.rectangle(np_img,
                                   [int(converted_bbox[0][0]), int(converted_bbox[0][1])],
                                   [int(converted_bbox[0][2]), int(converted_bbox[0][3])],
                                   [255, 138, 86], 1)
        for i in range(len(supressed_predictions)):
            for sp in supressed_predictions[i]:
                pred_box = sp[1:5].unsqueeze(0)
                iou = box_iou(pred_box, converted_bbox)
                pred_label = sp[-1].int().cpu()
                if img is not None:
                    np_img = cv2.rectangle(np_img,
                                              [int(pred_box[0][0]), int(pred_box[0][1])],
                                               [int(pred_box[0][2]), int(pred_box[0][3])],
                                              [int(class_id) * 3, 138, 255], 1)
                if iou > 0.5 and pred_label == class_id:
                    number_of_true_positive += 1
    if img is not None:
        np_img = cv2.resize(np_img, [832, 832], cv2.INTER_LINEAR)
        cv2.imshow("pred vs gt", np_img)
        cv2.waitKey()
    recall = float(number_of_true_positive / number_of_gt) if number_of_gt else 1
    precision = float(number_of_true_positive / number_of_proposals) if number_of_proposals else 0
    F1_score = 2 * recall * precision / (recall + precision + 1e-16)

    return F1_score, precision, recall

if __name__ == "__main__":
    import torch
    import logging
    from train_common import setup_logger, cuda_id, inner_train_loop, checkpoint_path
    from utils.checkpoint import save_checkpoint, load_checkpoint, load_only_model_from_checkpoint
    import os
    from tqdm import tqdm
    from dataset_loader.coco_loader import get_coco_loader
    from models.yolo_v3 import YoloV3
    from models.darknet53 import Darknet53
    from torchvision import transforms
    from utils.bbox_loss import BboxLoss
    import cv2

    model_name = "yolo_full_proper_div.tar"
    _, val_loader = get_coco_loader(1, False, 1)
    backbone = Darknet53()
    model = YoloV3(80, backbone).cuda(cuda_id).train()
    model = load_only_model_from_checkpoint(os.path.join(checkpoint_path, model_name), model)

    val_preprocess = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )


    model.eval()

    loss_fn = BboxLoss(80)

    ths = [0.2, 0.3, 0.4, 0.5, 0.6]
    ps =[]
    rs=[]
    with torch.no_grad():
        for th in ths:
            res = []
            loss = 0.0
            precision, recall = 0.0, 0.0
            for i, val in tqdm(enumerate(val_loader)):
                data = val['image']
                output = model(val_preprocess(data.cuda(cuda_id)))
                l, predicted_bboxes = loss_fn(output, [val['target']])
                sum_loss = 0

                def acc_loss(loss, s):
                    if isinstance(loss, dict):
                        for val in loss.values():
                            if isinstance(val, dict):
                                s = acc_loss(val, s)
                            else:
                                s += val
                    else:
                        s = loss
                    return s

                sum_loss = acc_loss(l, sum_loss)
                loss += sum_loss
                res.append(1)
                _, prec, rec = eval(output, val['target'], 416, obj_th=th)
                precision += prec
                recall += rec
                if i > 100:
                    break
            print(f"{recall / len(res)}, {precision / len(res)}, {loss / len(res)}")
            ps.append(precision / len(res))
            rs.append(recall / len(res))

    print(f"precision: {ps}")
    print(f"recalls: {rs}")
