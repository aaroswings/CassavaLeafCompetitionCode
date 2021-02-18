from pandas import DataFrame
import matplotlib.pyplot as plt
import math
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.cuda.amp import autocast

from util import plot_curve_w_linefit

config = pd.read_json('train_config.json', typ='series')
device = torch.device(config['device'])

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Credit: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    Copy-pasted from the original source, y_a <-> labels, y_b <-> mixup_labels
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def sam_mixup_epoch(model, 
                    optimizer, 
                    criterion, 
                    data,
                    lr_scheduler=None, 
                    class_weight_scheduler=None):
    train_losses = []
    instance_losses = []

    
    for i, (key, images, labels, lams, mixup_labels) in enumerate(data):
        print('\rRunning batch', i, end='')
        images = images.to(device)
        labels = labels.to(device)
        lams = lams.to(device)
        mixup_labels = mixup_labels.to(device)
        
        with autocast():
            outputs = model(images)
            losses = mixup_criterion(criterion, outputs, labels, mixup_labels, lams)
        losses.mean().backward()
        optimizer.first_step(zero_grad=True)
        
        with autocast():
            outputs = model(images)
            losses = mixup_criterion(criterion, outputs, labels, mixup_labels, lams)
        losses.mean().backward()
        optimizer.second_step(zero_grad=True)
        
        instance_losses += list(zip(key, losses.detach().cpu().numpy()))
        
        # Multiplicative lr needs to be stepped only after epoch
        if lr_scheduler is not None and config['lr_scheduler'] != 'multiplicative':
            lr_scheduler.step()
        if class_weight_scheduler is not None:
            class_weight_scheduler.step()

        train_losses.append(losses.mean().detach().cpu().numpy())
        
    return train_losses, instance_losses

def validate_model(model, 
                   valid_loader, 
                   epoch_title, 
                   label_map, 
                   repeat=5):
    val_predictions = []
    val_labels = []

    for _ in range(repeat):
        for i, (keys, images, labels) in enumerate(valid_loader):
            print(f'\rRunning inference on validation batch {i}', end='')
            val_labels += list(labels)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                predictions = predictions.cpu()
                val_predictions += [int(np.argmax(x)) for x in predictions]

    valid_acc = accuracy_score(val_predictions, val_labels)

    display(HTML(f'<h1>{epoch_title}</h1><br>Valid accuracy: {valid_acc}'))
    cm  = confusion_matrix(val_labels, val_predictions)
    cm_df = DataFrame(cm, label_map, label_map)
    heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.show()
    return valid_acc
    
