import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
# from visdom import Visdom

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from torch.utils import data
from tqdm import tqdm

from data import create_dataset
from models import create_model
from utils.utils import get_logger
from augmentations import get_composed_augmentations
from models.adaptation_model import CustomModel, CustomMetrics
from optimizers import get_optimizer
from schedulers import get_scheduler
from metrics import runningScore, averageMeter
from loss import get_loss_function
from utils import sync_batchnorm
from tensorboardX import SummaryWriter


def train(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  #source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)

    # Setup Metrics
    running_metrics_val = runningScore(cfg['data']['target']['n_class'])
    source_running_metrics_val = runningScore(cfg['data']['target']['n_class'])
    val_loss_meter = averageMeter()
    source_val_loss_meter = averageMeter()
    time_meter = averageMeter()
    loss_fn = get_loss_function(cfg)
    flag_train = True

    epoches = cfg['training']['epoches']

    source_train_loader = datasets.source_train_loader
    target_train_loader = datasets.target_train_loader
    logger.info('source train batchsize is {}'.format(source_train_loader.args.get('batch_size')))
    print('source train batchsize is {}'.format(source_train_loader.args.get('batch_size')))
    logger.info('target train batchsize is {}'.format(target_train_loader.batch_size))
    print('target train batchsize is {}'.format(target_train_loader.batch_size))

    val_loader = None
    if cfg.get('valset') == 'gta5':
        val_loader = datasets.source_valid_loader
        logger.info('valset is gta5')
        print('valset is gta5')
    else:
        val_loader = datasets.target_valid_loader
        logger.info('valset is cityscapes')
        print('valset is cityscapes')
    logger.info('val batchsize is {}'.format(val_loader.batch_size))
    print('val batchsize is {}'.format(val_loader.batch_size))

    # load category anchors
    objective_vectors = torch.load('category_anchors')
    model.objective_vectors = objective_vectors['objective_vectors']
    model.objective_vectors_num = objective_vectors['objective_num']

    # begin training
    model.iter = 0
    for epoch in range(epoches):
        if not flag_train:
            break
        if model.iter > cfg['training']['train_iters']:
            break

        # monitoring the accuracy and recall of CAG-based PLA and probability-based PLA
        score_cl, _ = model.metrics.running_metrics_val_clusters.get_scores()
        print('clus_IoU: {}'.format(score_cl["Mean IoU : \t"]))

        logger.info('clus_IoU: {}'.format(score_cl["Mean IoU : \t"]))
        logger.info('clus_Recall: {}'.format(model.metrics.calc_mean_Clu_recall()))
        logger.info(model.metrics.classes_recall_clu[:, 0] / model.metrics.classes_recall_clu[:, 1])
        logger.info('clus_Acc: {}'.format(np.mean(model.metrics.classes_recall_clu[:, 0] / model.metrics.classes_recall_clu[:, 1])))
        logger.info(model.metrics.classes_recall_clu[:, 0] / model.metrics.classes_recall_clu[:, 2])

        score_cl, _ = model.metrics.running_metrics_val_threshold.get_scores()
        logger.info('thr_IoU: {}'.format(score_cl["Mean IoU : \t"]))
        logger.info('thr_Recall: {}'.format(model.metrics.calc_mean_Thr_recall()))
        logger.info(model.metrics.classes_recall_thr[:, 0] / model.metrics.classes_recall_thr[:, 1])
        logger.info('thr_Acc: {}'.format(np.mean(model.metrics.classes_recall_thr[:, 0] / model.metrics.classes_recall_thr[:, 1])))
        logger.info(model.metrics.classes_recall_thr[:, 0] / model.metrics.classes_recall_thr[:, 2])
        model.metrics.reset()

        for (target_image, target_label, target_img_name) in datasets.target_train_loader:
            model.iter += 1
            i = model.iter
            if i > cfg['training']['train_iters']:
                break
            source_batchsize = cfg['data']['source']['batch_size']
            images, labels, source_img_name = datasets.source_train_loader.next()
            start_ts = time.time()

            images = images.to(device)
            labels = labels.to(device)
            target_image = target_image.to(device)
            target_label = target_label.to(device)
            model.scheduler_step()
            model.train(logger=logger)
            if cfg['training'].get('freeze_bn') == True:
                model.freeze_bn_apply()
            model.optimizer_zerograd()

            loss, loss_cls_L2, loss_pseudo = model.step(images, labels, target_image, target_label)
            if loss_cls_L2 > 10:
                logger.info('loss_cls_l2 abnormal!!')

            time_meter.update(time.time() - start_ts)
            if (i + 1) % cfg['training']['print_interval'] == 0:
                unchanged_cls_num = 0
                fmt_str = "Epoches [{:d}/{:d}] Iter [{:d}/{:d}]  Loss: {:.4f}  Loss_cls_L2: {:.4f}  Loss_pseudo: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                                        epoch+1,
                                        epoches,
                                        i + 1,
                                        cfg['training']['train_iters'], 
                                        loss.item(),
                                        loss_cls_L2,
                                        loss_pseudo,
                                        time_meter.avg / cfg['data']['source']['batch_size'])

                print(print_str)
                logger.info(print_str)
                logger.info('unchanged number of objective class vector: {}'.format(unchanged_cls_num))
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                writer.add_scalar('loss/train_cls_L2Loss', loss_cls_L2, i+1)
                writer.add_scalar('loss/train_pseudoLoss', loss_pseudo, i+1)
                time_meter.reset()

                score_cl, _ = model.metrics.running_metrics_val_clusters.get_scores()
                logger.info('clus_IoU: {}'.format(score_cl["Mean IoU : \t"]))
                logger.info('clus_Recall: {}'.format(model.metrics.calc_mean_Clu_recall()))
                logger.info('clus_Acc: {}'.format(np.mean(model.metrics.classes_recall_clu[:, 0] / model.metrics.classes_recall_clu[:, 2])))


                score_cl, _ = model.metrics.running_metrics_val_threshold.get_scores()
                logger.info('thr_IoU: {}'.format(score_cl["Mean IoU : \t"]))
                logger.info('thr_Recall: {}'.format(model.metrics.calc_mean_Thr_recall()))
                logger.info('thr_Acc: {}'.format(np.mean(model.metrics.classes_recall_thr[:, 0] / model.metrics.classes_recall_thr[:, 2])))

            # evaluation
            if (i + 1) % cfg['training']['val_interval'] == 0 or \
                (i + 1) == cfg['training']['train_iters']:
                validation(
                    model, logger, writer, datasets, device, running_metrics_val, val_loss_meter, loss_fn,\
                    source_val_loss_meter, source_running_metrics_val, iters = model.iter
                    )
                torch.cuda.empty_cache()
                logger.info('Best iou until now is {}'.format(model.best_iou))
            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

def validation(model, logger, writer, datasets, device, running_metrics_val, val_loss_meter, loss_fn,\
        source_val_loss_meter, source_running_metrics_val, iters):
    iters = iters
    _k = -1
    for v in model.optimizers:
        _k += 1
        for param_group in v.param_groups:
            _learning_rate = param_group.get('lr')
        logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(
            datasets.target_valid_loader, device, model, running_metrics_val,
            val_loss_meter, loss_fn
            )
        
    writer.add_scalar('loss/val_loss', val_loss_meter.avg, iters+1)
    logger.info("Iter %d Loss: %.4f" % (iters + 1, val_loss_meter.avg))

    writer.add_scalar('loss/source_val_loss', source_val_loss_meter.avg, iters+1)
    logger.info("Iter %d Source Loss: %.4f" % (iters + 1, source_val_loss_meter.avg))

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/{}'.format(k), v, iters+1)

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/cls_{}'.format(k), v, iters+1)

    val_loss_meter.reset()
    running_metrics_val.reset()

    source_val_loss_meter.reset()
    source_running_metrics_val.reset()

    torch.cuda.empty_cache()
    state = {}
    _k = -1
    for net in model.nets:
        _k += 1
        new_state = {
            "model_state": net.state_dict(),
            "optimizer_state": model.optimizers[_k].state_dict(),
            "scheduler_state": model.schedulers[_k].state_dict(),                            
        }
        state[net.__class__.__name__] = new_state
    state['iter'] = iters + 1
    state['best_iou'] = score["Mean IoU : \t"]
    save_path = os.path.join(writer.file_writer.get_logdir(),
                                "from_{}_to_{}_on_{}_current_model.pkl".format(
                                    cfg['data']['source']['name'],
                                    cfg['data']['target']['name'],
                                    cfg['model']['arch'],))
    torch.save(state, save_path)

    if score["Mean IoU : \t"] >= model.best_iou:
        torch.cuda.empty_cache()
        model.best_iou = score["Mean IoU : \t"]
        state = {}
        _k = -1
        for net in model.nets:
            _k += 1
            new_state = {
                "model_state": net.state_dict(),
                "optimizer_state": model.optimizers[_k].state_dict(),
                "scheduler_state": model.schedulers[_k].state_dict(),                            
            }
            state[net.__class__.__name__] = new_state
        state['iter'] = iters + 1
        state['best_iou'] = model.best_iou
        save_path = os.path.join(writer.file_writer.get_logdir(),
                                    "from_{}_to_{}_on_{}_best_model.pkl".format(
                                        cfg['data']['source']['name'],
                                        cfg['data']['target']['name'],
                                        cfg['model']['arch'],))
        torch.save(state, save_path)
    return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val, val_loss_meter, loss_fn):
    for (images_val, labels_val, filename) in tqdm(valid_loader):


        images_val = images_val.to(device)
        labels_val = labels_val.to(device)
        _, _, feat_cls, outs = model.forward(images_val)

        outputs = F.interpolate(outs, size=images_val.size()[2:], mode='bilinear', align_corners=True)
        val_loss = loss_fn(input=outputs, target=labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)
        val_loss_meter.update(val_loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        # default="configs/pspnet_cityscapes.yml",
        # default="configs/pspnet_gta5.yml",
        default='configs/adaptation_from_city_to_gta.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    # path = cfg['training']['save_path']
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    # train(cfg, writer, logger)
    train(cfg, writer, logger)
