import logging

import torch

cuda_id = 1
checkpoint_path = "/home/ad.adasworks.com/tamas.zsiros/work/yolov3-impl/trainer_scripts/trained_models"

def setup_logger(file_name):
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

def inner_train_loop(data, labels, model, optimizer, loss_fn):
    optimizer.zero_grad()

    output = model(data.cuda(cuda_id))
    if isinstance(labels, torch.Tensor):
        labels = labels.cuda(cuda_id)
    loss = loss_fn(output, labels)
    if len(loss) > 1:
        loss, _ = loss
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
    sum_loss = acc_loss(loss, sum_loss)
    sum_loss.backward()

    optimizer.step()

    return loss

