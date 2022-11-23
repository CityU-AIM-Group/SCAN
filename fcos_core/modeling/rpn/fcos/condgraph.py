# --------------------------------------------------------
# SCAN++: Enhanced Semantic Conditioned Adaptation for Domain Adaptive Object Detection (TMM)
# Written by Wuyang Li
# This file covers the core operation on the feature maps for domain adaptation
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from .loss import make_fcos_loss_evaluator, make_prototype_evaluator, PrototypeComputation
from fcos_core.layers import  FocalLoss, MultiHeadAttention
from fcos_core.layers import Scale

import os
# import numpy as np

eps = 1e-8
INF = 1e10


# POST PROCESSING

def sim_matrix(a, b, eps=eps):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class PROTOTYPECounter():
    def __init__(self, cycle=3, stop=False):
        self.cycle = cycle
        self.counter = -1
        self.stop = stop

    def __call__(self, *args, **kwargs):
        # 0, 1, 2, 3, 4, ..., n, n, n, n
        if self.stop:
            if self.counter == self.cycle:
                return self.cycle
            else:
                self.counter += 1
                return self.counter
        # 0, 1, 2, 3, 0, 1, 2, 3, ...
        else:
            self.counter += 1
            if self.counter == self.cycle:
                self.counter = 0
            return self.counter


class GRAPHHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        # TODO: Implement the sigmoid version first.

        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = cfg.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):

            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            middle_tower.append(nn.ReLU())
        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        # initialization
        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower


class GRAPHModule(torch.nn.Module):
    """
    Module for Semantic Middle Head
    """

    def __init__(self, cfg, in_channels):
        super(GRAPHModule, self).__init__()

        self.debug_cfg = cfg.MODEL.DEBUG_CFG
        self.cfg = cfg.clone()

        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES  # FG+1
        self.with_bias = cfg.MODEL.MIDDLE_HEAD.COND_WITH_BIAS
        self.with_concated_maps = cfg.MODEL.MIDDLE_HEAD.CAT_ACT_MAP

        self.head_in = GRAPHHead(cfg, in_channels, in_channels, mode='in')
        if self.with_concated_maps:
            head_out = GRAPHHead(cfg, in_channels + self.num_classes, in_channels, mode='out')
            self.head_out = head_out

        self.with_self_training = cfg.MODEL.MIDDLE_HEAD.WITH_SELF_TRAINING
        self.with_adaptive_self_training = cfg.MODEL.MIDDLE_HEAD.WITH_ADAPTIVE_SELF_TRAINING
        self.with_OT = cfg.MODEL.MIDDLE_HEAD.WITH_OT  # 'NODES', 'EDGE', 'ADJ'

        self.subsampling_type = cfg.MODEL.MIDDLE_HEAD.SUB_SAMPLING  # source sub sampling
        self.update_bank_iter = cfg.MODEL.MIDDLE_HEAD.UPDATE_BANK_ITER  # bank update
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Prototype settings
        self.prototype_iter = cfg.MODEL.MIDDLE_HEAD.PROTO_ITER  # 1, 3, 9
        self.counter_rnn = PROTOTYPECounter(self.prototype_iter, stop=True)
        self.node_generator = make_prototype_evaluator(cfg)
        self.register_buffer('prototype',
                             torch.randn(self.num_classes, cfg.MODEL.MIDDLE_HEAD.PROTO_CHANNEL, self.prototype_iter))
        # self.prototype_buffer_batch = torch.zeros(self.num_classes, cfg.MODEL.MIDDLE_HEAD.PROTO_CHANNEL)

        # graph settings
        self.multihead_attn = MultiHeadAttention(256, 1, dropout=0.1, version=cfg.MODEL.MIDDLE_HEAD.ATT_VERSION)
        # self.multihead_attn = MultiHeadAttention_v1(256, 1, dropout=0.1)
        self.node_cls = nn.Sequential(
            torch.nn.Linear(cfg.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL, 512),
            nn.ReLU(),
            torch.nn.Linear(512, self.num_classes),
        )

        # conditional kernel
        self.cond_rnn = nn.RNN(256, 512, 2, nonlinearity='tanh')
        self.cond_nx1 = torch.nn.Conv2d(512, 256, kernel_size=(self.prototype_iter, 1))
        self.bank_update_counter = 0

        # loss weight
        self.node_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.act_loss_func = FocalLoss(
            self.num_classes
        )
        # self.OT_loss_fn = torch.nn.KLDivLoss(reduction='mean')
        self.OT_loss_fn = torch.nn.MSELoss(reduction='mean')

        self.node_weight = cfg.MODEL.MIDDLE_HEAD.NODE_LOSS_WEIGHT
        self.conditional_kernel_weight = cfg.MODEL.MIDDLE_HEAD.ACT_LOSS_WEIGHT
        self.OT_weight = cfg.MODEL.MIDDLE_HEAD.OT_LOSS_WEIGHT

    def _init_weight(self):

        for layer in self.node_cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cond_nx1.weight, std=0.01)
        nn.init.constant_(self.cond_nx1.bias, 0)

        nn.init.kaiming_normal_(self.cond_nx1.weight, std=0.01)
        nn.init.constant_(self.cond_nx1.bias, 0)

    def dynamic_conv_on_feats(self, features):
        '''
        :param features: per-level image features
        :return: activation maps. activation logits
        '''
        conded_weight = self.get_conded_weight()
        return_act_maps = []
        return_act_logits = []

        for l, feature in enumerate(features):
            act_maps_logits = self.dynamic_conv(feature, conded_weight)
            return_act_maps.append(act_maps_logits.softmax(dim=1))
            return_act_logits.append(act_maps_logits)
        return return_act_maps, return_act_logits

    def get_act_loss(self, logits, labels):
        act_maps_logits_flatten = []
        act_maps_labels_flatten = []
        for l, feature in enumerate(logits):
            act_maps_logits_flatten.append(logits[l].permute(0, 2, 3, 1).reshape(-1, self.num_classes))
            act_maps_labels_flatten.append(labels[l].reshape(-1))

        act_maps_logits_flatten = torch.cat(act_maps_logits_flatten, dim=0)
        act_maps_labels_flatten = torch.cat(act_maps_labels_flatten, dim=0)

        act_loss = self.conditional_kernel_weight * self.act_loss_func(
            act_maps_logits_flatten,
            act_maps_labels_flatten.long()
        )

        return act_loss

    def _get_node_loss(self, nodes, labels, weight=None):
        logits = self.node_cls(nodes)
        target = (labels).long()

        if weight is not None:
            assert weight.size() == labels.size()
            node_loss = self.node_weight * (self.node_loss_fn(logits, target) * weight).mean()
        else:
            node_loss = self.node_weight * (self.node_loss_fn(logits, target)).mean()

        return node_loss

    def _forward_graph_aggregation(self, sampled_points, pos_labels):
        prototype_buffer_batch = sampled_points[0].new_zeros(self.num_classes, 256)
        agg_points = self.multihead_attn(sampled_points, sampled_points, sampled_points)[0]

        for i in range(self.num_classes):
            indx = pos_labels == i
            if indx.any() and indx.sum() > 1:
                prototype_buffer_batch[i] = agg_points[indx].mean(dim=0)
            elif indx.sum() == 1:
                prototype_buffer_batch[i] = agg_points[indx]

        return agg_points, prototype_buffer_batch

    def OT_transfer(self, tg_prototype, tg_nodes, tg_labels):
        sr_prototype = self.prototype.detach()  # [class, channel, iter]
        sr_prototype = sr_prototype.permute(1, 0, 2).reshape(256, -1).t()  # [class x iter, channel]
        sim = torch.mm(tg_nodes, sr_prototype.t())

        M = self.InstNorm_layer(sim[None, None, :, :])
        M = self.sinkhorn(M[:, 0, :, :], n_iters=20).squeeze().exp().detach()
        target = torch.mm(M, sr_prototype)
        # tg_nodes = tg_nodes.log_softmax(-1)
        # target = target.softmax(-1)
        transfer_loss = self.OT_loss_fn(tg_nodes, target)
        return transfer_loss

    def _forward_train_source(self, features, targets=None):
        # Sample graph nodes
        sr_loss_dict = {}
        locations = self.compute_locations(features)

        if self.subsampling_type == 'condition':
            with torch.no_grad():
                act_maps, _ = self.dynamic_conv_on_feats(features)
            sampled_points, pos_labels, act_maps_labels = self.node_generator(
                locations, features, targets, act_maps
            )
            del act_maps
        else:
            sampled_points, pos_labels, act_maps_labels = self.node_generator(
                locations, features, targets,
            )

        # graph aggregation
        agg_nodes, sr_proto = self._forward_graph_aggregation(sampled_points, pos_labels)
        node_loss = self._get_node_loss(agg_nodes, pos_labels)
        sr_loss_dict.update(node_loss=node_loss)

        self.update_prototype_nx1_rnn(sr_proto)

        # learn conditional kernels
        act_maps, act_logits = self.dynamic_conv_on_feats(features)
        act_loss = self.get_act_loss(act_logits, act_maps_labels)

        features = self.features_post_processing(features, act_maps)

        sr_loss_dict.update(conditional_kernel_loss=act_loss)

        return features, sr_loss_dict, act_maps

    def _forward_train_target(self, features):

        tg_loss_dict = {}
        # get probability maps
        act_maps, _ = self.dynamic_conv_on_feats(features)
        # Graph node sampling
        sampled_points, sampled_plabels, sampled_weights = self.node_generator(
            locations=None, features=features, targets=act_maps
        )
        features = self.features_post_processing(features, act_maps)

        if len(sampled_points) > 0:
            agg_points, tg_proto = self._forward_graph_aggregation(sampled_points, sampled_plabels)
            if self.with_self_training:
                assert agg_points.size(0) == sampled_plabels.size(0)

                if self.with_adaptive_self_training:
                    node_loss = self._get_node_loss(agg_points, sampled_plabels, sampled_weights)
                else:
                    node_loss = self._get_node_loss(agg_points, sampled_plabels)
                tg_loss_dict.update(node_loss=node_loss)

            if self.with_OT:
                OT_loss = self.OT_weight * self.OT_transfer(tg_proto, sampled_points, sampled_plabels)
                tg_loss_dict.update(OT_loss=OT_loss)

        return features, tg_loss_dict, act_maps

    def features_post_processing(self, features, act_maps):
        if self.with_concated_maps:
            for l, feature in enumerate(features):
                features[l] = torch.cat([features[l], act_maps[l]], dim=1)
            features = self.head_out(features)
        return features

    def _forward_inference(self, features):
        act_maps_list, _ = self.dynamic_conv_on_feats(features)
        features = self.features_post_processing(features, act_maps_list)
        return features, {}, act_maps_list

    def forward(self, images, features, targets=None, return_maps=False, forward_target=False):

        features = self.head_in(features)  # project to the graphical space
        if self.training and not forward_target and targets:
            return self._forward_train_source(features, targets)
        elif self.training and forward_target:
            return self._forward_train_target(features)
        else:
            return self._forward_inference(features)

    def update_prototype_nx1_rnn(self, prototype_batch, mom=0.95):

        iter = self.counter_rnn()
        exist_indx = prototype_batch.sum(-1).bool()
        prototype_batch = prototype_batch.detach()

        if iter == self.prototype_iter:
            mom = torch.cosine_similarity(self.prototype[exist_indx, :, iter - 1],
                                          prototype_batch[exist_indx]).unsqueeze(1)
            # move
            for i in range(iter - 1):
                self.prototype[:, :, i] = self.prototype[:, :, i + 1]
            # update t+1
            self.prototype[exist_indx, :, iter - 1] = self.prototype[exist_indx, :, iter - 1] * mom \
                                                      + prototype_batch[exist_indx] * (1 - mom)
        else:
            mom = torch.cosine_similarity(self.prototype[exist_indx, :, iter],
                                          prototype_batch[exist_indx]).unsqueeze(1)
            self.prototype[exist_indx, :, iter] = self.prototype[exist_indx, :, iter] * mom \
                                                  + prototype_batch[exist_indx] * (1 - mom)

    def dynamic_conv(self, features, kernel_par):
        num_classes = self.num_classes
        if self.with_bias:
            # WITH BIAS TERM
            weight = kernel_par[:, :-1]
            bias = kernel_par[:, -1]
            weight = weight.view(num_classes, -1, 1, 1)
            return torch.nn.functional.conv2d(features, weight, bias=bias)
        else:
            weight = kernel_par.view(num_classes, -1, 1, 1)
            return torch.nn.functional.conv2d(features, weight)

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def get_conded_weight(self):
        # if self.use_rnn:
        # num_classes_fg_bg, channel, iter [9, 256, 3]
        conded_weight = self.cond_nx1(
            self.cond_rnn(
                self.prototype.permute(2, 0, 1))[0].permute(1, 2, 0).unsqueeze(-1)
        ).squeeze()
        return conded_weight

    def sinkhorn(self, log_alpha, n_iters=20, slack=True, eps=-1):

        # Sinkhorn iterations
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)

                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)

                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()

        return log_alpha

    def save_targets(self, cfg, features, return_act_maps, conded_weight, targets):
        # Save many features
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        locations = self.compute_locations(features)
        box_cls_gt, box_reg_gt, box_ctr_gt = self.loss_evaluator.replace_targets(
            locations, return_act_maps, None, None, targets
        )
        self.debugger.save_feat(box_cls_gt, id='target_gt')
        self.debugger.save_feat(return_act_maps, id='target_act_maps')
        self.debugger.save_feat(features, id='target_feats')
        self.debugger.save_feat(conded_weight, id='cond_weitht')
        self.debugger.save_feat(self.prototype, id='prototype')
        os._exit(0)


def build_condgraph(cfg, in_channels):
    return GRAPHModule(cfg, in_channels)


def test_nan(para, name='gcn'):
    assert para.max() < INF, 'nan of {}'.format(name)
    return para

