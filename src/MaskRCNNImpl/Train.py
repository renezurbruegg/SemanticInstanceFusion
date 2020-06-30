import math
import sys

import src.MaskRCNNImpl.Model as Model
from src.MaskRCNNImpl import utils as utils


#########################################################################
# This File contains all functions dedicated to training the Mask R-CNN #
#########################################################################

# Trains a given Model a given number of epochs TODO: validation
def train(model, optimizer, lr_scheduler, num_epochs, data_loader, data_loader_test, device ):
    """Trains model for a number of epochs"""
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    Model.save_weights(model)


# Trains model for one epoch
def train_one_epoch(model, optimizer, data_loader, dataset, device, epoch, print_freq, skip_nan = False, verbose = False,
                    bl_name = "blacklist.txt", update_blacklist = False):
    """Trains model for one epoch
    model: Model which you want to train
    optimizer: Optimizer which should be used (Adam, SGD or else...)
    dataset: A pytorch dataloader which handles the training data
    device: Specifies which device should be used for training (preferably a GPU)
    epoch: Integer which refers to the current epoch
    print_freq: Specifies at which frequency the trainer should print out a short summary of the training process
    skip_nan: If True this will prevent that the whole process stops when it encounters a loss of not a number.
        It will abort the current epoch and just try the next one. (was used to find bad pictures in the dataset which caused
        the loss to be NaN)
    verbose: If True the process will print some additional information about the training process.
    bl_name: Name of the blacklist file used for a given dataset. Some images cause the loss to become NaN so they have to
        be blacklisted.
    update_blacklist: If True this will cause the training process to update the given blacklist if it encounters bad images.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            log = dataset.get_debug_string()
            print(log)
            if update_blacklist:
                f = open(bl_name, "a+")
                for i in log["paths"]:
                    f.write(i+ "\n")
                f.close()
            if skip_nan:
                break
            else:
                sys.exit(0)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if verbose: print(dataset.get_debug_string())
        dataset.clear_debug()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def train_one_epoch_nyu(model, optimizer, data_loader, dataset, device, epoch, print_freq, skip_nan = False, verbose = False):
    """Trains model for one epoch. Specifically for the NYU dataset. (Used in previous models)
    model: Model which you want to train
    optimizer: Optimizer which should be used (Adam, SGD or else...)
    data_loader: A pytorch dataloader which handles the training data
    dataset: not really used anymore TODO: remove
    device: Specifies which device should be used for training (preferably a GPU)
    epoch: Integer which refers to the current epoch
    print_freq: Specifies at which frequency the trainer should print out a short summary of the training process
    skip_nan: If True this will prevent that the whole process stops when it encounters a loss of not a number.
        It will abort the current epoch and just try the next one. (was used to find bad pictures in the dataset which caused
        the loss to be NaN)
    verbose: If True the process will print some additional information about the training process."""

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            if skip_nan:
                break
            else:
                sys.exit(0)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])