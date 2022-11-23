# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
# import ipdb
import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list

import os
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.utils.miscellaneous import mkdir
from .validation import _inference
from fcos_core.utils.comm import synchronize

def foward_detector(cfg, model, images, targets=None, return_maps=True,forward_target=False):
    with_middle_head = cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()
    model_backbone = model["backbone"]

    # if with_middle_head:
    model_fcos = model["fcos"]
    images = to_image_list(images)
    features = model_backbone(images.tensors)
    losses = {}
    if with_middle_head:
        model_middle_head = model["middle_head"]
        features, loss_middle_head, return_act_maps = model_middle_head(images, features, targets=targets,
                                           return_maps=return_maps, forward_target=forward_target)
        losses.update(loss_middle_head)

    else:
        return_act_maps = None

    proposals, proposal_losses, score_maps = model_fcos(
        images, features, targets=targets, return_maps=return_maps, act_maps=return_act_maps)

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }

    if return_act_maps:
        return_act_maps = {
            layer: return_act_maps[map_layer_to_index[layer]]
            for layer in feature_layers
        }

    if model_fcos.training:
        if not targets:
            assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0
        losses.update(proposal_losses)
        return losses, f, return_act_maps
    else:
        # inference
        result = proposals
        return result



def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def validataion(cfg, model, data_loader, distributed=False):
    if distributed:
        model["backbone"] = model["backbone"].module
        model["fcos"] = model["fcos"].module
    iou_types = ("bbox",)
    dataset_name = cfg.DATASETS.TEST
    assert len(data_loader) == 1, "More than one validation sets!"
    data_loader = data_loader[0]
    # for  dataset_name, data_loader_val in zip( dataset_names, data_loader):
    results = _inference(
        cfg,
        model,
        data_loader,
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
    )
    synchronize()
    return results

def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg,
        distributed,
        meters,
):
    with_DA = cfg.MODEL.DA_ON
    stage_2_ON = False
    data_loader_source = data_loader["source"]
    # Start training
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    # model.train()
    for k in model:
        model[k].train()

    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    AP50 = cfg.SOLVER.INITIAL_AP50
    AP50_online = 0
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    if not with_DA:

        max_iter = len(data_loader_source)
        for iteration, (images_s,targets_s, _) in enumerate(data_loader_source, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            if not pytorch_1_1_0_or_later:
                # scheduler.step()
                for k in scheduler:
                    scheduler[k].step()
            images_s = images_s.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]
            for k in optimizer:
                optimizer[k].zero_grad()
            ##########################################################################
            #################### (1): train G with source domain #####################
            ##########################################################################
            loss_dict, features_s, score_maps_s = foward_detector(cfg, model, images_s, targets=targets_s, return_maps=True, mode='source')

            loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

            losses.backward()

            for k in optimizer:
                optimizer[k].step()

            if pytorch_1_1_0_or_later:
                # scheduler.step()
                for k in scheduler:
                    scheduler[k].step()
            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, ))
            if cfg.SOLVER.ADAPT_VAL_ON:
                if iteration % cfg.SOLVER.VAL_ITER == 0:
                    val_results = validataion(cfg, model, data_loader["val"], distributed)
                    # used for saving model
                    AP50_online = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # used for logging
                    meter_AP50= val_results.results['bbox']['AP50'] * 100
                    meter_AP = val_results.results['bbox']['AP']* 100
                    meters.update(AP = meter_AP, AP50 = meter_AP50 )

                    if AP50_online > AP50:
                        AP50 = AP50_online
                        checkpointer.save("model_{}_{:07d}".format(AP50, iteration), **arguments)
                        print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50,
                                                                                           iteration))
                    if distributed:
                        model["backbone"] = model["backbone"].module
                        model["middle_head"] = model["middle_head"].module
                        model["fcos"] = model["fcos"].module
                    for k in model:
                        model[k].train()

            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)

            # save the last model
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
    else:

        data_loader_target = data_loader["target"]
        max_iter = max(len(data_loader_source), len(data_loader_target))
        USE_DIS_GLOBAL = arguments["use_dis_global"]
        USE_DIS_CENTER_AWARE = arguments["use_dis_ca"]
        USE_DIS_OUT = arguments["use_dis_out"]
        USE_DIS_CON = arguments["use_dis_con"]
        used_feature_layers = arguments["use_feature_layers"]
        # dataloader

        # classified label of source domain and target domain
        source_label = 1.0
        target_label = 0.0
        # dis_lambda
        if USE_DIS_GLOBAL:
            ga_dis_lambda = arguments["ga_dis_lambda"]
        if USE_DIS_CENTER_AWARE:
            ca_dis_lambda = arguments["ca_dis_lambda"]
        if USE_DIS_OUT:
            out_dis_lambda = arguments["out_dis_lambda"]
        if USE_DIS_CON:
            con_dis_lambda = arguments["con_dis_lambda"]

        assert len(data_loader_source) == len(data_loader_target)
        for iteration, ((images_s, targets_s, _), (images_t, targets_t, _)) in enumerate(zip(data_loader_source, data_loader_target), start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                # scheduler.step()
                for k in scheduler:
                    scheduler[k].step()
            images_s = images_s.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]

            images_t = images_t.to(device)

            # targets_t = [target_t.to(device) for target_t in targets_t]
            for k in optimizer:
                optimizer[k].zero_grad()

            ##########################################################################
            #################### (1): train G with source domain #####################
            ##########################################################################

            loss_dict, features_s, score_maps_s = foward_detector(cfg,
                model, images_s, targets=targets_s, return_maps=True)


            loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_gs=losses_reduced, **loss_dict_reduced)
            losses.backward(retain_graph=True)

            # del loss_dict, losses

            ##########################################################################
            #################### (2): train D with source domain #####################
            ##########################################################################
            # TODO GCNs computation graph
            # loss_dict = {}
            if cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON:

                loss_dict = {'zeros': 0 * loss_dict['node_loss_gs']}
            else:
                loss_dict = {}

            for layer in used_feature_layers:

                if USE_DIS_GLOBAL:
                    loss_dict["loss_adv_%s_ds" % layer] = \
                        ga_dis_lambda * model["dis_%s" % layer](features_s[layer], source_label, domain='source')
                if USE_DIS_CENTER_AWARE:
                    # detatch score_map
                    for map_type in score_maps_s[layer]:
                        score_maps_s[layer][map_type] = score_maps_s[layer][map_type].detach()
                    loss_dict["loss_adv_%s_CA_ds" % layer] = \
                        ca_dis_lambda * model["dis_%s_CA" % layer]\
                            (features_s[layer], source_label, score_maps_s[layer], domain='source')
                if USE_DIS_OUT:
                    loss_dict["loss_adv_%s_OUT_ds" % layer] = \
                        out_dis_lambda * model["dis_%s_OUT" % layer]\
                            (source_label, score_maps_s[layer], domain='source')
                if USE_DIS_CON:
                    loss_dict["loss_adv_%s_CON_ds" % layer] = \
                        con_dis_lambda * model["dis_%s_CON" % layer]\
                            (features_s[layer], source_label,score_maps_s[layer], domain='source')


            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes

            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_ds=losses_reduced, **loss_dict_reduced)

            losses.backward()
            del loss_dict, losses

            ##########################################################################
            #################### (3): train D with target domain #####################
            ##########################################################################
            #TODO A better dynamic strategy
            forward_target = AP50_online > cfg.SOLVER.INITIAL_AP50
            # forward_target = iteration>1000
            # forward_target = True

            loss_dict, features_t, score_maps_t = foward_detector(cfg, model, images_t, return_maps=True, forward_target=forward_target)
            loss_dict = {k + "_gt": loss_dict[k] for k in loss_dict}
            # losses = sum(loss for loss in loss_dict.values())
            # assert len(loss_dict) == 1 and loss_dict["zero"] == 0  # loss_dict should be empty dict
            # loss_dict["loss_adv_Pn"] = model_dis_Pn(features_t["Pn"], target_label, domain='target')
            for layer in used_feature_layers:
                # detatch score_map
                if USE_DIS_GLOBAL:
                    loss_dict["loss_adv_%s_dt" % layer] = \
                        ga_dis_lambda * model["dis_%s" % layer]\
                            (features_t[layer], target_label, domain='target')
                if USE_DIS_CENTER_AWARE:
                    for map_type in score_maps_t[layer]:
                        score_maps_t[layer][map_type] = score_maps_t[layer][map_type].detach()
                    loss_dict["loss_adv_%s_CA_dt" %layer] = \
                        ca_dis_lambda * model["dis_%s_CA" % layer]\
                            (features_t[layer], target_label, score_maps_t[layer], domain='target')
                if USE_DIS_OUT:
                    loss_dict["loss_adv_%s_OUT_dt" %layer] = \
                        out_dis_lambda * model["dis_%s_OUT" % layer]\
                            (target_label, score_maps_t[layer], domain='target')
                if USE_DIS_CON:
                    loss_dict["loss_adv_%s_CON_dt" % layer] = \
                        con_dis_lambda * model["dis_%s_CON" % layer]\
                            (features_t[layer], target_label, score_maps_t[layer],  domain='target')

            losses = sum(loss for loss in loss_dict.values())

            # del loss_dict['zero_gt']
            # # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_dt=losses_reduced, **loss_dict_reduced)
            losses.backward()
            del loss_dict, losses

                # saved GRL gradient
                # grad_list = []
                # for layer in used_feature_layers:
                #     def save_grl_grad(grad):
                #         grad_list.append(grad)
                #     features_t[layer].register_hook(save_grl_grad)
            # print(' back D with target domain')
            # Uncomment to log GRL gradient
            #     grl_grad = {}
            #     grl_grad_log = {}
            # grl_grad = {
            #     layer: grad_list[i]
            #     for i, layer in enumerate(used_feature_layers)
            # }
            # for layer in used_feature_layers:
            #     saved_grad = grl_grad[layer]
            #     grl_grad_log["grl_%s_abs_mean" % layer] = torch.mean(
            #         torch.abs(saved_grad)) * 10e4
            #     grl_grad_log["grl_%s_mean" % layer] = torch.mean(saved_grad) * 10e6
            #     grl_grad_log["grl_%s_std" % layer] = torch.std(saved_grad) * 10e6
            #     grl_grad_log["grl_%s_max" % layer] = torch.max(saved_grad) * 10e6
            #     grl_grad_log["grl_%s_min" % layer] = torch.min(saved_grad) * 10e6
            # meters.update(**grl_grad_log)
            #     del loss_dict, losses, grad_list, grl_grad, grl_grad_log

            ##########################################################################
            ##########################################################################
            ##########################################################################

            # optimizer.step()
            for k in optimizer:
                optimizer[k].step()

            if pytorch_1_1_0_or_later:
                # scheduler.step()
                for k in scheduler:
                    scheduler[k].step()

            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            sample_layer = used_feature_layers[0]  # sample any one of used feature layer
            if USE_DIS_GLOBAL:
                sample_optimizer = optimizer["dis_%s" % sample_layer]
            if USE_DIS_CENTER_AWARE:
                sample_optimizer = optimizer["dis_%s_CA" % sample_layer]
            if USE_DIS_OUT:
                sample_optimizer = optimizer["dis_%s_OUT" % sample_layer]
            if USE_DIS_CON:
                sample_optimizer = optimizer["dis_%s_CON" % sample_layer]
            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        "lr_middle_head: {lr_middle_head:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "lr_dis: {lr_dis:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        lr_middle_head=optimizer["middle_head"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        lr_dis=sample_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    ))

            if cfg.SOLVER.ADAPT_VAL_ON :
                if iteration % cfg.SOLVER.VAL_ITER == 0:
                    val_results = validataion(cfg, model, data_loader["val"], distributed)

                    if type(val_results)== dict:
                        AP50_online = val_results['map'] * 100
                        meters.update(AP50=AP50_online)
                    else:
                        val_results = val_results[0]
                        # used for saving model
                        AP50_online = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                        # used for logging
                        meter_AP50 = val_results.results['bbox']['AP50'] * 100
                        meter_AP = val_results.results['bbox']['AP'] * 100
                        meters.update(AP=meter_AP, AP50=meter_AP50)

                    if AP50_online > AP50:
                        stage_2_ON = True
                        AP50 = AP50_online
                        checkpointer.save("model_{}_{:07d}".format(AP50, iteration), **arguments)
                        print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50, iteration))
                    # if distributed:
                    #     model["backbone"] = model["backbone"].module
                    #     model["middle_head"] = model["middle_head"].module
                    #     model["fcos"] = model["fcos"].module
                    for k in model:
                        model[k].train()
            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))
