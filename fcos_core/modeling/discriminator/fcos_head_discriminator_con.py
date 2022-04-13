# --------------------------------------------------------
# SCAN: Cross-domain Object Detection with Semantic Conditioned Adaptation (AAAI22 ORAL)
# Written by Wuyang Li
# This file covers the Conditonal Kernel guided Alignment (CKA)
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from .layer import GradientReversal


class FCOSDiscriminator_con(nn.Module):
    def __init__(self, with_GA=False, fusion_cfg='concat', num_convs=3, in_channels=256, num_classes=2,
                 grad_reverse_lambda=-1.0, grl_applied_domain='both', patch_stride=None, cfg=None):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_con, self).__init__()
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))
        self.use_bg = False


        if self.use_bg:
            self.num_classes = num_classes
        else:
            self.num_classes = num_classes - 1

        self.with_GA = with_GA
        self.fusion_cfg = fusion_cfg

        self.class_cond_map = []
        for i in range(self.num_classes):
            self.class_cond_map.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels + 1 if self.fusion_cfg == 'concat' else in_channels,
                        128, #128
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    # nn.GroupNorm(32, in_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        128, 1, kernel_size=3, stride=1,
                        padding=1)
                )
            )

        for i, block in enumerate(self.class_cond_map):
            self.add_module('classifier_cls_{}'.format(i), block)

        self.patch_stride = patch_stride
        assert patch_stride == None or type(patch_stride) == int, 'wrong format of patch stride'
        for modules in [self.dis_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        for modules in self.class_cond_map:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

    def forward(self, feature, target, act_maps=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
            act_maps = self.grad_reverse(act_maps)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)


        if self.patch_stride:
            feature = self.pool(feature)
        x = self.dis_tower(feature)
        loss = 0
        # if not self.adv_w_bg:
        for c in range(self.num_classes):

            map_indx = c if self.use_bg else c + 1
            if self.fusion_cfg == 'concat':
                x_cls = torch.cat((x, act_maps[:, map_indx, :, :].unsqueeze(1)), dim=1)
            elif self.fusion_cfg == 'mul':
                x_cls = torch.mul(x, act_maps[:, map_indx, :, :].unsqueeze(1)).contiguous()
            elif self.fusion_cfg == 'mul_detached':
                x_cls = torch.mul(x, act_maps[:, map_indx, :, :].unsqueeze(1).detach())
            else:
                raise KeyError("Unknown fusion config!")
            logits = self.class_cond_map[c](x_cls)
            targets = torch.full(logits.shape, target, dtype=torch.float, device=x.device)
            if self.num_classes>1:
                loss_cls = F.binary_cross_entropy_with_logits(logits, targets,
                                                              weight=act_maps[:, map_indx, :, :].unsqueeze(1).detach(),
                                                              reduction='sum') / ( act_maps[:, map_indx, :, :].sum().detach())
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, targets  )
            loss += loss_cls / self.num_classes


        return loss