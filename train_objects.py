import math
import pandas as pd
import sys
sys.path.extend(['input/sam-optimizer-pytorch', 'input/pytorch-image-models-master'])
from sam import SAM
import timm
import torch

config = pd.read_json('train_config.json', typ='series')

device = torch.device(config['device'])

def get_model(name=None):
    if name is None:
        name = config['model_name']
    if 'efficientnet' in name:
        model = timm.create_model(name, pretrained = True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)

    if 'resnext' in name:
        # or model = torch.hub.load('facebookresearch/WSL-Images', name)
        model = torch.load(f'input/networks/resnext101/{name}.pickle')
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = False
        else:
            for p in module.parameters():
                p.requires_grad = True
    model.to(device)
    return model

def get_optimizer(model):
     return SAM(filter(lambda p: p.requires_grad, model.parameters()), 
                torch.optim.SGD, 
                lr=config['base_lr'], 
                momentum=config['momentum'],
                weight_decay=config['weight_decay'])

def get_lr_scheduler(optimizer, n_epochs, steps):
    if config['lr_scheduler'] == 'multiplicative':
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
                                            lr_lambda=lambda epoch: config['multiplicative_lr_factor'])
    if config['lr_scheduler'] == 'one_cycle':
        lr_schedule = lr_scheduler.OneCycleLR(optimizer, max_lr=config['cycle_max_lr'], epochs=n_epochs,
                             steps_per_epoch=epochs)
    raise ValueError('No known lr scheduler set in config')
        
def get_cycle_class_weight(criterion, base_class_weight, converge=False, steps_to_converge=None):
    return CyclicClassWeighting(criterion, 
                               base_class_weight, 
                               config['cyclic_class_weight_period'],
                               device,
                               converge,
                               steps_to_converge)
        
class CyclicClassWeighting:
    def __init__(self, criterion, base_class_weight, period, device, converge, steps_to_converge):
        """Modulates the weighting of different classes between equal weight and some base weight over time.
        Specifically, it interpolates between equal weight and the base weight according to a triangle wave.
        Args:
            criterion: training criterion, like categorical cross entropy etc, which takes class weight
            base_class_weight: the class weight at the bottom of the triangle wave'sr ange
            period: number of steps for a complete period of the triangle wave
            device: which device to put the weight vector at each step
            converge: whether or not to linearly fade alpha to 1 and converge to equal class weight
            steps_to_converge: how many steps to fade out the alpha variable
        """
        self.criterion = criterion
        self.base_class_weight = base_class_weight
        self.period = period
        self.step_count = 0
        self.device = device
        self.converge = converge
        if self.converge:
            self.steps_to_converge = steps_to_converge
            self.steps_taken = 0
        else:
            self.steps_to_converge = None
            self.steps_taken = None
        
    def step(self):
        self.step_count += 1 
        relative_step = self.step_count % self.period
        # Triangle wave
        alpha = 2 * abs(relative_step/self.period - math.floor(relative_step/self.period+.5))
        if self.converge:
            '''If we're converging to equal weight, fade out alpha.
            Beta is driven to 1 as steps_taken converges to steps_to_converge and (1 - beta) * alpha is driven to 0.
            '''
            if self.steps_taken < self.steps_to_converge:
                beta = self.steps_taken / self.steps_to_converge
                alpha  = (1 - beta) * alpha + beta
                self.steps_taken += 1
            else:
                alpha = 1
        
        self.criterion.weight = self.set_class_weight(alpha).to(self.device)
        
    def set_class_weight(self, alpha):
        return (torch.ones(self.base_class_weight.shape) 
                * alpha + (1 - alpha) * self.base_class_weight)