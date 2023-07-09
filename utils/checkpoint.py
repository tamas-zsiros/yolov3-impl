import logging
import torch
from trainer_scripts.train_common import cuda_id

def save_checkpoint(epoch, model, optimizer, path, iter, scheduler):
    logging.info(f"saving checkpoint to {path}")
    torch.save({
        'epoch': epoch,
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)

def load_checkpoint(epoch, model, optimizer, path, iter, scheduler):
    try:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_id))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        iter = checkpoint['iter']
        return epoch, model, optimizer, iter, scheduler
    except BaseException as e:
        logging.error(f"failed to load checkpoint: {e}")
    return None

def load_only_model_from_checkpoint(path, model):
    try:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_id))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except BaseException as e:
        logging.error(f"failed to load checkpoint: {e}")
        return model
    return None


