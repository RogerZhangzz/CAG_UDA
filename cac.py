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


def CAC(cfg, writer, logger):
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
    class_features = Class_Features(numbers=19)

    # begin training
    model.iter = 0
    for epoch in range(epoches):
        if not flag_train:
            break
        if model.iter > cfg['training']['train_iters']:
            break

        # monitoring the accuracy and recall of CAG-based PLA and probability-based PLA

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
            if model.PredNet.training:
                model.PredNet.eval()
            with torch.no_grad():
                _, _, feat_cls, output = model.PredNet_Forward(images)
                batch, w, h = labels.size()
                newlabels = labels.reshape([batch, 1, w, h]).float()
                newlabels = F.interpolate(newlabels, size=feat_cls.size()[2:], mode='nearest')
                vectors, ids = class_features.calculate_mean_vector(feat_cls, output, newlabels, model)
                for t in range(len(ids)):
                    model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')

            time_meter.update(time.time() - start_ts)
            if model.iter % 20 == 0:
                print("Iter [{:d}] Time {:.4f}".format(model.iter, time_meter.avg))

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break
    save_path = os.path.join(writer.file_writer.get_logdir(),
                                "anchors_on_{}_from_{}".format(
                                    cfg['data']['source']['name'],
                                    cfg['model']['arch'],))
    torch.save(model.objective_vectors, save_path)


class Class_Features:
    def __init__(self, numbers = 19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        # self.class_features = np.zeros((19, 256))
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.all_vectors = []
        self.pred_ids = []
        self.ids = []
        self.pred_num = np.zeros(numbers + 1)
        self.labels = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            'ignored',]
        self.markers = [".",
            ",",
            "o",
            "v",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "|",]
        return

    def calculate_mean_vector(self, feat_cls, outputs, labels_val, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        # outputs_pred = model.process_pred(outputs_softmax, 0.5)
        # outputs_pred = outputs_argmax[:, 0:19, :, :] * outputs_softmax
        labels_expanded = model.process_label(labels_val)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids
    
    def calculate_mean(self,):
        out = [[] for i in range(self.class_numbers)]
        for i in range(self.class_numbers):
            out[i] = self.class_features[i] / max(self.num[i], 1)
        return out

    def calculate_dis(self, vector, id):
        if isinstance(vector, torch.Tensor): vector = vector.detach().cpu().numpy().squeeze()
        mean = self.calculate_mean()
        dis = []
        for i in range(self.class_numbers):
            dis_vec = [x - y for x,y in zip(mean[i], vector)]
            dis.append(np.linalg.norm(dis_vec, 2))
        return dis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        # default="configs/pspnet_cityscapes.yml",
        # default="configs/pspnet_gta5.yml",
        default='configs/CAC_from_gta_to_city.yml',
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
    CAC(cfg, writer, logger)
