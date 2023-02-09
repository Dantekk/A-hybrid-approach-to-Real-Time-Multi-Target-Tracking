from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os


from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn

from .networks.convnext import get_conv_next as get_conv_next
from .networks.convnext_fpn import get_conv_next_fpn as get_conv_next_fpn

from .networks.efficientnet import get_efficientnet as get_efficientnet

_model_factory = {

  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'convnext' : get_conv_next,
  'convnextfpn' : get_conv_next_fpn,
  'efficientnet' : get_efficientnet,

}

_loss_unc_weights = {"s_id" : 0, "s_det" : 0}
_classifier_id = [0]
state_dict_re_id = [0]

def create_model(arch, heads, head_conv):
  
  if "convnext" or "efficientnet" in arch:
      type_arch = arch[:arch.find('_')] if '_' in arch else arch
      get_model = _model_factory[type_arch]
      model = get_model(type_model=arch, heads=heads, head_conv=head_conv)
  else:
      num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
      arch = arch[:arch.find('_')] if '_' in arch else arch
      get_model = _model_factory[arch]
      model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  
  # For counting number of all model parameters
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  # For counting number of ONLY training model parameters
  #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  #print(model)
  print("Number of model parameter : ",pytorch_total_params)

  return model

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None, pretrained=0, train_only_det=0, demo=False):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  if demo:
    return model

  # Se sta caricando un modello pre-trained -> non deve caricare l'ultimo stato dell'ottimizzatore!
  if pretrained==1:
      return model, optimizer, 0

  if pretrained==0 and resume==True:
      _loss_unc_weights["s_id"] = checkpoint["s_id"]
      _loss_unc_weights["s_det"] = checkpoint["s_det"]
      state_dict_re_id[0] = torch.load(model_path[:-4]+"_classifier_id.pth")

  if pretrained==0 and resume==True:
      # resume optimizer parameters
      if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
          optimizer.load_state_dict(checkpoint['optimizer'])
          start_epoch = checkpoint['epoch']
          start_lr = lr
          for step in lr_step:
            if start_epoch >= step:
              start_lr *= 0.1
          ####
          ##start_lr=1e-5
          ####
          for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
          print('Resumed optimizer with start lr', start_lr)
        else:
          print('No optimizer parameters in checkpoint.')
      if optimizer is not None:
        return model, optimizer, start_epoch
      else:
        return model

def save_model(path, epoch, model, optimizer=None, save_classified_id=True):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict,
          's_id' : _loss_unc_weights['s_id'],
          's_det' : _loss_unc_weights['s_det']}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

  ##
  if save_classified_id:
      state_dict_classified_id = _classifier_id[0].state_dict()
      data_classifier_id = {'state_dict' : state_dict_classified_id}
      torch.save(data_classifier_id, path[:-4]+"_classifier_id.pth")
