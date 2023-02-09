from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer

###
from models.model import _loss_unc_weights, state_dict_re_id

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        ### Se bisogna addestrare tutto -> non solo la fase di detection
        if opt.train_only_det==0:
            self.classifier = nn.Linear(self.emb_dim, self.nID)
            if opt.id_loss == 'focal':
                torch.nn.init.normal_(self.classifier.weight, std=0.01)
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                torch.nn.init.constant_(self.classifier.bias, bias_value)
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

            ### Uncertainty loss weights
            self.s_det = nn.Parameter(-1.85 * torch.ones(1))
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.first = True


    def forward(self, outputs, batch):
        opt = self.opt


        ### ripristina i pesi della uncertainty loss e del classificatore della re-id -> se stiamo caricando il modello!
        if opt.resume==True and self.first==True and opt.pretrained==0 :#or opt.train_only_det==0):
            self.s_det.data = torch.tensor(_loss_unc_weights['s_det']* torch.ones(1)).to('cuda:0')
            self.s_id.data = torch.tensor(_loss_unc_weights['s_id']* torch.ones(1)).to('cuda:0')
            print("************ Prima iterazione : \ns_det : ",self.s_det.item())
            print("s_id : ",self.s_id.item())
            ###
            #print(state_dict_re_id[0]['state_dict'])
            #xxx = input()
            self.classifier.load_state_dict(state_dict_re_id[0]['state_dict'])
            print(self.classifier.weight)
            print(self.classifier.bias)
            self.first=False


        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            ###
            if opt.train_only_det==1:
                continue
            ###

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        ###
        if opt.train_only_det==1:
            loss=det_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                          'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': torch.zeros(batch['input'].size(0))}

            return loss, loss_stats, 0, 0, None
        ###

        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
            #print("s_det : ",self.s_det.item())
            #print("s_id : ",self.s_id.item())
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats, self.s_det.item(), self.s_id.item(), self.classifier


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _update_unc_loss_weights(self, opt):
        MotLoss().update_unc_loss_weights()

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
