# --------------------------------------------------------
# SCAN: Cross-domain Object Detection with Semantic Conditioned Adaptation (AAAI22 ORAL)
# Modified by Wuyang Li
# This file contains specific functions for computing losses of FCOS
# This file covers the core operation for node sampling
# --------------------------------------------------------
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss, SigmoidFocalLoss,  MeanShift_GPU

from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
import os
from sklearn import *
import time
INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        locations = locations.cuda()

        object_sizes_of_interest= object_sizes_of_interest.cuda()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]

            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox.cuda()
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)


            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def replace_targets(self,locations, box_cls, box_regression, centerness, targets):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        reg_targets_flatten = []
        labels, reg_targets = self.prepare_targets(locations, targets)
        tmp = []

        for l in range(len(labels)):
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            tmp.append(reg_targets[l].size(0))
        reg_targets_flatten = torch.cat(reg_targets_flatten,dim=0)
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
        centerness_targets_list = []
        k = 0
        for i in tmp:
            centerness_targets_list.append(centerness_targets[k:k+i])
            k += i

        box_cls_gt = []
        box_reg_gt = []
        box_ctr_gt = []

        for l in range(len(labels)):
            n, c, h, w = box_cls[l].size()
            if c >len(labels):
                c=c-1
            lb = F.one_hot(labels[l].reshape(-1), 9)[:,1:].float()
            box_cls_gt.append(lb.reshape(n,h,w,c).permute(0,3,1,2).cuda())
            box_reg_gt.append(reg_targets[l].reshape(-1).reshape(n,h,w,4).permute(0,3,1,2).cuda())
            box_ctr_gt.append(centerness_targets_list[l].reshape(-1).reshape(n,h,w,1).permute(0,3,1,2).float().cuda())
        return box_cls_gt, box_reg_gt, box_ctr_gt


    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()


        return cls_loss, reg_loss, centerness_loss




def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

class PrototypeComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.num_class = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.num_class_fgbg = cfg.MODEL.FCOS.NUM_CLASSES
        self.cfg =cfg.clone()
        if self.cfg.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG == 'mean_shift':
            from fcos_core.layers import MeanShift_GPU
            self.meanshift = MeanShift_GPU(batch_size=10000, bandwidth=0.1)
        self.fg_bg_centers = torch.zeros(2, 256)

        self.thrd_min_max = cfg.SOLVER.MIDDLE_HEAD.PLABEL_TH

        # self.norm_cfg = cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_NORM
        # self.relu = torch.nn.ReLU().cuda()
        # self.leakyrelu = torch.nn.LeakyReLU(0.1).cuda()
        # self.gcn_out_activation = cfg.MODEL.MIDDLE_HEAD.GCN_OUT_ACTIVATION

        # self.prototype_buffer_batch = torch.zeros(self.num_class, 256).cuda()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            # reg_targets_level_first.append(
            #     torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            # )

        return labels_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0


            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def KMEANS_batch(self, act_maps):

        N, C, H, W = act_maps.size()
        km = cluster.KMeans(init='k-means++', n_clusters=2,
                            random_state=1, n_jobs=-1, n_init=2)
        Y = km.fit_predict(act_maps.numpy().reshape(-1, 1))

        return Y.reshape(N, C, H, W).cuda()

    def KMEANS_batch_ClS_FEAT(self, act_maps, features, score_mask=True):

        mask = (act_maps > 0.5).int()
        act_maps=act_maps.detach().cpu()
        features=features.detach().cpu()
        N, CLS, H, W = act_maps.size()
        N, CHANNEL, H, W = features.size()
        features_n = []

        #  CLS, N, CHANNEL, H, W
        for i in range(CLS):
            # print(act_maps[i].unsqueeze(1).shape)
            features_n.append((features* act_maps[:, i, :, :].unsqueeze(1)).unsqueeze(0))
        features_n = torch.cat(features_n, dim=0)
        #  N, CLS,H, W, CHANNEL
        features_n = features_n.permute(1, 0, 3, 4, 2).reshape(-1, CHANNEL)

        center_update = self.fg_bg_centers.sum() == 0
        if center_update:
            km = cluster.KMeans(init='k-means++', n_clusters=2,
                                random_state=1, n_init=2)
            Y = km.fit_predict(features_n.numpy())
            self.fg_bg_centers = torch.Tensor(km.cluster_centers_)
        else:
            Y = cluster.KMeans(init=self.fg_bg_centers.numpy(), n_clusters=2,
                                random_state=1,n_init=2).fit_predict(features_n.numpy())
        Y = torch.Tensor(Y.reshape(N, CLS, H, W)).cuda() * mask
        # Post processing
        if (Y == 0).sum() < (Y == 1).sum():
            Y = 1 - Y
            print(' clustering error!!!!!!!')
        Y = Y.permute(0, 2, 3, 1).reshape(-1, CLS).sum(-1).bool()
        return Y



    def DBSCAN_batch_cpu(self, act_maps, features,):
        act_maps=act_maps.detach().cpu()
        features=features.detach().cpu()
        N, CLS, H, W = act_maps.size()
        N, CHANNEL, H, W = features.size()
        features_n = []

        #  CLS, N, CHANNEL, H, W
        for i in range(CLS):
            # print(act_maps[i].unsqueeze(1).shape)
            features_n.append((features* act_maps[:, i, :, :].unsqueeze(1)).unsqueeze(0))
        features_n = torch.cat(features_n, dim=0)
        features_n = features_n.permute(1, 0, 3, 4, 2).reshape(-1, CHANNEL)
        # Score mask
        mask = (act_maps >  self.cfg.MODEL.MIDDLE_HEAD.DBSCAN_THR).bool().unsqueeze(-1).reshape(-1)
        mask_float = (act_maps > self.cfg.MODEL.MIDDLE_HEAD.DBSCAN_THR).float().unsqueeze(-1).reshape(-1)

        pos_feats = features_n[mask]
        if pos_feats.bool().any():
            Y = cluster.DBSCAN(eps=self.cfg.MODEL.MIDDLE_HEAD.DBSCAN_EPS, n_jobs=-1).fit_predict(pos_feats.numpy())
            Y[Y < 0] = 1
            mask_float[mask] = torch.Tensor(Y)

        Y = torch.Tensor(mask_float.reshape(N, CLS, H, W))
        Y = Y.permute(0, 2, 3, 1).reshape(-1, CLS).sum(-1).bool().cuda()

        return Y




    def __call__(self, locations, features, targets):

        if locations:
            N, C, _, _ = features[0].size()
            labels = self.prepare_targets(locations, targets)

            pos_points = []
            pos_labels = []
            neg_points = []
            for l in range(len(labels)):
                pos_indx =  labels[l].reshape(-1) > 0
                neg_indx =  labels[l].reshape(-1) == 0
                pos_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[pos_indx])
                pos_labels.append(labels[l][pos_indx])

                if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    if len(labels[l][pos_indx]) > len(labels[l][neg_indx]):
                        neg_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx])
                    else:
                        neg_indx = list(np.floor(np.linspace(0,len(labels[l][neg_indx])-2, len(labels[l][pos_indx]))).astype(int))
                        neg_points.append(neg_points_temp[neg_indx])

            pos_points = torch.cat(pos_points,dim=0)
            pos_labels = torch.cat(pos_labels,dim=0)

            if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                neg_points = torch.cat(neg_points, dim=0)
                neg_labels = pos_labels.new_zeros((neg_points.size(0)))
                pos_points = torch.cat([neg_points, pos_points] ,dim=0)
                pos_labels = torch.cat([neg_labels, pos_labels] )
            # ipdb.set_trace()
            if self.cfg.MODEL.MIDDLE_HEAD.ACT_LOSS:
                return pos_points, pos_labels, labels
            else:
                return pos_points, pos_labels
        else:
            # k =  (targets[0][:, 1:, :, :] > 0.05).sum().float() /(targets[0][:,1:,:,:]>0.05).numel()
            # print(k)
            if True:
                # print('start self-training')
                ## for tg
                act_maps_lvl_first = targets
                N, C, _, _ = features[0].size()
                N, Cls, _, _ = targets[0].size()
                neg_points =[]
                pos_plabels = []
                pos_points = []

                start = time.time()
                for l, feature in enumerate(features):
                    if self.cfg.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG == 'score_threshold':
                        act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class_fgbg)
                        conf_pos_indx = (act_maps[:, 1:] > self.thrd_min_max[0]).sum(dim=-1).bool()
                    elif self.cfg.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG == 'mean_shift':
                        act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(N, -1,self.num_class_fgbg)
                        conf_pos_indx = act_maps.new_zeros(N, act_maps.size(1), self.num_class_fgbg-1)
                        for img in range(N):
                            for cls in range(self.num_class_fgbg-1):
                                conf_pos_indx[img,:,cls] = self.meanshift.fit(act_maps[img,:,cls+1].unsqueeze(-1))[0]
                        conf_pos_indx = conf_pos_indx.sum(dim=-1).bool()

                    elif self.cfg.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG == 'kmeans':
                        conf_pos_indx = self.KMEANS_batch_ClS_FEAT(act_maps_lvl_first[l][:,1:,:,:],feature, score_mask=True)
                    elif self.cfg.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG == 'dbscan':
                        conf_pos_indx = self.DBSCAN_batch_cpu(act_maps_lvl_first[l][:, 1:, :, :], feature)
                    else:
                        raise  KeyError('unknown target labels!')
                    # Balanced sampling BG pixels
                    if conf_pos_indx.any():
                        act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class_fgbg)
                        pos_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx])
                        pos_plabels.append(act_maps[conf_pos_indx,1:].argmax(dim=-1) + 1)
                        neg_indx = ~conf_pos_indx
                        neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                        neg_indx_new = list(np.floor(np.linspace(0, (neg_indx==True).sum().item()- 2, (conf_pos_indx==True).sum().item())).astype(int))
                        neg_points.append(neg_points_temp[neg_indx_new])
                end = time.time()

                # print(end-start)
                if len(pos_points)>0:
                    pos_points = torch.cat(pos_points,dim=0)
                    pos_plabels = torch.cat(pos_plabels,dim=0)
                    neg_points = torch.cat(neg_points, dim=0)
                    neg_plabels = pos_plabels.new_zeros((neg_points.size(0)))
                    points = torch.cat([neg_points, pos_points], dim=0)
                    plabels = torch.cat([neg_plabels, pos_plabels])
                    # print(len(pos_points))
                    return points, plabels, None
                else:
                    return None, None, None
            else:
                return None, None, None




def make_prototype_evaluator(cfg):
    prototype_evaluator = PrototypeComputation(cfg)
    return prototype_evaluator
