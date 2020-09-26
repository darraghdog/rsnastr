from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    model= model.to(device)
    torch.nn.DataParallel(model)
    for input_dict in tqdm(loader):
        inputs = input_dict['image']
        inputs = inputs.to(device)
        input_var = torch.autograd.Variable(inputs)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def detach_params(model):
    for param in model.parameters():
        param.detach_()
    return model


def swa(base_model, weightfiles, dataloader, batch_size, device):
    
    def load_model(net, wt):
        checkpoint = torch.load(wt, map_location=torch.device(device))
        net.load_state_dict(checkpoint['state_dict'])
        net = net.to(device)
        return net
    
    net = load_model(copy.deepcopy(base_model), weightfiles[0])

    for i, f in enumerate(weightfiles[1:]):
        net2 = load_model(copy.deepcopy(base_model), f)
        moving_average(net, net2, 1. / (i + 2))

    with torch.no_grad():
        bn_update(dataloader, net, device)
    return net