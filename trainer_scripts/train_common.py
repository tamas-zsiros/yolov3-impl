import logging

import torch

cuda_id = 0
checkpoint_path = "./trained_models"

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
    sum_loss = 0
    if isinstance(loss, dict):
        for val in loss.values():
            sum_loss += val
    else:
        sum_loss = loss
    sum_loss.backward()

    optimizer.step()

    return loss

