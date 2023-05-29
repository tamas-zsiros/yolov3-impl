import logging
import torch

def save_checkpoint(epoch, model, optimizer, path, iter):
    logging.info(f"saving checkpoint to {path}")
    torch.save({
        'epoch': epoch,
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(epoch, model, optimizer, path, iter):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        iter = checkpoint['iter']
        return epoch, model, optimizer, iter
    except BaseException as e:
        logging.error(f"failed to load checkpoint: {e}")
    return None


