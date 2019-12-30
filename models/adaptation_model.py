from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import torch
import numpy as np

from torch.autograd import Variable
from optimizers import get_optimizer
from schedulers import get_scheduler
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
# from utils.bn import InPlaceABN, InPlaceABNSync
from models.networks import define_D
from models.deeplab import DeepLab
from models.decoder import Decoder
from models.aspp import ASPP
from models.discriminator import FCDiscriminator, FCDiscriminator_low, FCDiscriminator_out, FCDiscriminator_class
from loss import get_loss_function
from .utils import freeze_bn, GradReverse, normalisation_pooling
from metrics import runningScore



class CustomMetrics():
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        self.classes_recall_thr = np.zeros([19, 3])
        self.classes_recall_thr_num = np.zeros([19])
        self.classes_recall_clu = np.zeros([19, 3])
        self.classes_recall_clu_num = np.zeros([19])
        self.running_metrics_val_threshold = runningScore(self.class_numbers)
        self.running_metrics_val_clusters = runningScore(self.class_numbers)
        self.clu_threshold = np.full((19), 2.5)
    
    def update(self, feat_cls, outputs, labels, model):     
        '''calculate accuracy. caring about recall but not IoU'''
        batch, width, height = labels.shape
        labels = labels.reshape([batch, 1, width, height]).float()
        labels = F.interpolate(labels, size=feat_cls.size()[2:], mode='nearest')
        outputs_threshold = outputs.clone()
        outputs_threshold = F.softmax(outputs_threshold, dim=1)
        self.running_metrics_val_threshold.update(labels.cpu().numpy(), outputs_threshold.argmax(1).cpu().numpy())
        for i in range(19):
            outputs_threshold[:, i, :, :] = torch.where(outputs_threshold[:, i, :, :] > model.class_threshold[i], torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())

        _batch, _channel, _w, _h = outputs_threshold.shape
        _tmp = np.full([_batch, 1, _w, _h], 0.2,)
        _tmp = torch.Tensor(_tmp).cuda()
        _tmp = torch.cat((outputs_threshold, _tmp), 1)
        threshold_arg = _tmp.argmax(1, keepdim=True)
        threshold_arg[threshold_arg == 19] = 250 #ignore index
        truth, pred_all, truth_all = self.calc_recall(labels.cpu().int().numpy(), threshold_arg.cpu().int().numpy())
        self.classes_recall_thr[:, 0] += truth
        self.classes_recall_thr[:, 2] += pred_all
        self.classes_recall_thr[:, 1] += truth_all


        outputs_cluster = outputs.clone()
        for i in range(19):
            filters = torch.Tensor(model.objective_vectors[i]).reshape(256,1,1)
            outputs_cluster[:, i, :, :] = torch.norm( torch.Tensor(model.objective_vectors[i]).reshape(-1,1,1).expand(-1,128,224).cuda() - feat_cls, 2, dim=1,)
        outputs_cluster_min, outputs_cluster_arg = outputs_cluster.min(dim=1, keepdim=True)
        outputs_cluster_second = outputs_cluster.scatter_(1, outputs_cluster_arg, 100)
        if torch.unique(outputs_cluster_second.argmax(1) - outputs_cluster_arg.squeeze()).squeeze().item() != 0:
            raise NotImplementedError('wrong when computing L2 norm!!')
        outputs_cluster_secondmin, outputs_cluster_secondarg = outputs_cluster_second.min(dim=1, keepdim=True)
        self.running_metrics_val_clusters.update(labels.cpu().numpy(), outputs_cluster_arg.cpu().numpy())
        
        tmp_arg = np.copy(outputs_cluster_arg.cpu().numpy())
        outputs_cluster_arg[(outputs_cluster_secondmin - outputs_cluster_min) < torch.Tensor(self.clu_threshold[tmp_arg]).cuda()] = 250
        truth, pred_all, truth_all = self.calc_recall(labels.cpu().int().numpy(), outputs_cluster_arg.cpu().int().numpy())
        self.classes_recall_clu[:, 0] += truth
        self.classes_recall_clu[:, 2] += pred_all
        self.classes_recall_clu[:, 1] += truth_all
        return threshold_arg, outputs_cluster_arg

    def calc_recall(self, gt, argmax):
        truth = np.zeros([self.class_numbers])
        pred_all = np.zeros([self.class_numbers])
        truth_all = np.zeros([self.class_numbers])
        for i in range(self.class_numbers):
            truth[i] = (gt == i)[argmax == i].sum()
            pred_all[i] = (argmax == i).sum()
            truth_all[i] = (gt == i).sum()
        pass
        return truth, pred_all, truth_all
    
    def calc_mean_Clu_recall(self, ):
        # recall = np.zeros([self.class_numbers])
        # for i in range(self.class_numbers):
        #     if self.classes_recall_clu_num[i] != 0:
        #         recall[i] = self.classes_recall_clu[i] / self.classes_recall_clu_num[i]
        return np.mean(self.classes_recall_clu[:, 0] / self.classes_recall_clu[:, 1])
    
    def calc_mean_Thr_recall(self, ):
        # recall = np.zeros([self.class_numbers])
        # for i in range(self.class_numbers):
        #     if self.classes_recall_thr_num[i] != 0:
        #         recall[i] = self.classes_recall_thr[i] / self.classes_recall_thr_num[i]
        return np.mean(self.classes_recall_thr[:, 0] / self.classes_recall_thr[:, 1])

    def reset(self, ):
        self.running_metrics_val_clusters.reset()
        self.running_metrics_val_threshold.reset()
        self.classes_recall_clu = np.zeros([19, 3])
        self.classes_recall_thr = np.zeros([19, 3])

class CustomModel():
    def __init__(self, cfg, writer, logger):
        # super(CustomModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.class_numbers = 19
        self.logger = logger
        cfg_model = cfg['model']
        self.cfg_model = cfg_model
        self.best_iou = -100
        self.iter = 0
        self.nets = []
        self.split_gpu = 0
        self.default_gpu = cfg['model']['default_gpu']
        self.PredNet_Dir = None
        self.valid_classes = cfg['training']['valid_classes']
        self.G_train = True
        self.objective_vectors = np.zeros([19, 256])
        self.objective_vectors_num = np.zeros([19])
        self.objective_vectors_dis = np.zeros([19, 19])
        self.class_threshold = np.zeros(self.class_numbers)
        self.class_threshold = np.full([19], 0.95)
        self.metrics = CustomMetrics(self.class_numbers)
        self.cls_feature_weight = cfg['training']['cls_feature_weight']

        bn = cfg_model['bn']
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        # elif bn == 'sync_abn':
        #     BatchNorm = InPlaceABNSync
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        # elif bn == 'abn':
        #     BatchNorm = InPlaceABN
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))
        self.PredNet = DeepLab(
                num_classes=19,
                backbone=cfg_model['basenet']['version'],
                output_stride=16,
                bn=cfg_model['bn'],
                freeze_bn=True,
                ).cuda()
        self.load_PredNet(cfg, writer, logger, dir=None, net=self.PredNet)
        self.PredNet_DP = self.init_device(self.PredNet, gpu_id=self.default_gpu, whether_DP=True) 
        self.PredNet.eval()
        self.PredNet_num = 0

        self.BaseNet = DeepLab(
                            num_classes=19,
                            backbone=cfg_model['basenet']['version'],
                            output_stride=16,
                            bn=cfg_model['bn'],
                            freeze_bn=False,
                            )

        logger.info('the backbone is {}'.format(cfg_model['basenet']['version']))

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets.extend([self.BaseNet])
        self.nets_DP = [self.BaseNet_DP]

        self.optimizers = []
        self.schedulers = []        
        # optimizer_cls = get_optimizer(cfg)
        optimizer_cls = torch.optim.SGD
        optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                            if k != 'name'}
        # optimizer_cls_D = torch.optim.SGD
        # optimizer_params_D = {k:v for k, v in cfg['training']['optimizer_D'].items() 
        #                     if k != 'name'}
        self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
        self.schedulers.extend([self.BaseSchedule])
        self.setup(cfg, writer, logger)

        self.adv_source_label = 0
        self.adv_target_label = 1
        self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        self.loss_fn = get_loss_function(cfg)
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.smoothloss = nn.SmoothL1Loss()
        self.triplet_loss = nn.TripletMarginLoss()

    def create_PredNet(self,):
        ss =  DeepLab(
                num_classes=19,
                backbone=self.cfg_model['basenet']['version'],
                output_stride=16,
                bn=self.cfg_model['bn'],
                freeze_bn=True,
                ).cuda()
        ss.eval()
        return ss

    def setup(self, cfg, writer, logger):
        '''
        set optimizer and load pretrained model
        '''
        for net in self.nets:
            # name = net.__class__.__name__
            self.init_weights(cfg['model']['init'], logger, net)
            print("Initializition completed")
            if hasattr(net, '_load_pretrained_model') and cfg['model']['pretrained']:
                print("loading pretrained model for {}".format(net.__class__.__name__))
                net._load_pretrained_model()
        '''load pretrained model
        '''
        if cfg['training']['resume_flag']:
            self.load_nets(cfg, writer, logger)
        pass

    def forward(self, input):
        feat, feat_low, feat_cls, output = self.BaseNet_DP(input)
        return feat, feat_low, feat_cls, output

    def forward_Up(self, input):
        feat, feat_low, feat_cls, output = self.forward(input)
        output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        return feat, feat_low, feat_cls, output

    def PredNet_Forward(self, input):
        with torch.no_grad():
            _, _, feat_cls, output_result = self.PredNet_DP(input)
        return _, _, feat_cls, output_result

    def calculate_mean_vector(self, feat_cls, outputs, labels, ):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        labels_expanded = self.process_label(labels)
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
                scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
                s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def step(self, source_x, source_label, target_x, target_label):

        _, _, source_feat_cls, source_output = self.forward(input=source_x)
        source_outputUp = F.interpolate(source_output, size=source_x.size()[2:], mode='bilinear', align_corners=True)

        loss_GTA = self.loss_fn(input=source_outputUp, target=source_label)
        self.PredNet.eval()

        with torch.no_grad():
            _, _, feat_cls, output = self.PredNet_Forward(target_x)
            # calculate pseudo-labels
            threshold_arg, cluster_arg = self.metrics.update(feat_cls, output, target_label, self)

        loss_L2_source_cls = torch.Tensor([0]).cuda(self.split_gpu)
        loss_L2_target_cls = torch.Tensor([0]).cuda(self.split_gpu)
        _, _, target_feat_cls, target_output = self.forward(target_x)

        if self.cfg['training']['loss_L2_cls']:     # distance loss
            _batch, _w, _h = source_label.shape
            source_label_downsampled = source_label.reshape([_batch,1,_w, _h]).float()
            source_label_downsampled = F.interpolate(source_label_downsampled.float(), size=source_feat_cls.size()[2:], mode='nearest')   #or F.softmax(input=source_output, dim=1)
            source_vectors, source_ids = self.calculate_mean_vector(source_feat_cls, source_output, source_label_downsampled)
            target_vectors, target_ids = self.calculate_mean_vector(target_feat_cls, target_output, cluster_arg.float())
            loss_L2_source_cls = self.class_vectors_alignment(source_ids, source_vectors)
            loss_L2_target_cls = self.class_vectors_alignment(target_ids, target_vectors)
            # target_vectors, target_ids = self.calculate_mean_vector(target_feat_cls, target_output, threshold_arg.float())
            # loss_L2_target_cls += self.class_vectors_alignment(target_ids, target_vectors)
        loss_L2_cls = self.cls_feature_weight * (loss_L2_source_cls + loss_L2_target_cls)

        loss = torch.Tensor([0]).cuda()
        batch, _, w, h = cluster_arg.shape
        # cluster_arg[cluster_arg != threshold_arg] = 250
        loss_CTS = (self.loss_fn(input=target_output, target=cluster_arg.reshape([batch, w, h])) \
            + self.loss_fn(input=target_output, target=threshold_arg.reshape([batch, w, h]))) / 2   # CAG-based and probability-based PLA
        # loss_CTS = self.loss_fn(input=target_output, target=cluster_arg.reshape([batch, w, h]))   # CAG-based PLA
        # loss_CTS = self.loss_fn(input=target_output, target=threshold_arg.reshape([batch, w, h])) # probability-based PLA
        if self.G_train and self.cfg['training']['loss_pseudo_label']:
            loss = loss + loss_CTS
        if self.G_train and self.cfg['training']['loss_source_seg']:
            loss = loss + loss_GTA
        if self.cfg['training']['loss_L2_cls']:
            loss = loss + torch.sum(loss_L2_cls)

        if loss.item() != 0:
            loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()
        return loss, loss_L2_cls.item(), loss_CTS.item()

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def class_vectors_alignment(self, ids, vectors):
        loss = torch.Tensor([0]).cuda(self.default_gpu)
        for i in range(len(ids)):
            if ids[i] not in self.valid_classes:
                continue
            new_loss = self.smoothloss(vectors[i].squeeze().cuda(self.default_gpu), torch.Tensor(self.objective_vectors[ids[i]]).cuda(self.default_gpu))
            while (new_loss.item() > 5):
                new_loss = new_loss / 10
            loss = loss + new_loss
        loss = loss / len(ids) * 10
        pass
        return loss

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        # for net in self.nets:
        #     self.schedulers[net.__class__.__name__].step()
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    
    def optimizer_step(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].step()
        for opt in self.optimizers:
            opt.step()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger!=None:    
                logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            if logger!=None:    
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
            # if logger!=None:    
            #     logger.info("Successfully set the model train mode") 
        else:
            net.train()
            # if logger!= None:
            #     logger.info(print("Successfully set {} train mode".format(net.__class__.__name__)))
        return

    def set_requires_grad(self, logger, net, requires_grad = False):
        """Set requires_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            net (BaseModel)       -- the network which will be operated on
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        for parameter in net.parameters():
            parameter.requires_grad = requires_grad
        # print("Successfully set {} requires_grad with {}".format(net.__class__.__name__, requires_grad))
        # return
        
    def set_requires_grad_layer(self, logger, net, layer_type='batchnorm', requires_grad=False):  
        '''    set specific type of layers whether needing grad
        '''

        # print('Warning: all the BatchNorm params are fixed!')
        # logger.info('Warning: all the BatchNorm params are fixed!')
        for net in self.nets:
            for _i in net.modules():
                if _i.__class__.__name__.lower().find(layer_type.lower()) != -1:
                    _i.weight.requires_grad = requires_grad
        return

    def init_weights(self, cfg, logger, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, SynchronizedBatchNorm2d) or classname.find('BatchNorm2d') != -1 \
                or isinstance(m, nn.GroupNorm):
                # or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_() # BatchNorm Layer's weight is not a matrix; only normal distribution applies.


        print('initialize {} with {}'.format(init_type, net.__class__.__name__))
        logger.info('initialize {} with {}'.format(init_type, net.__class__.__name__))
        net.apply(init_func)  # apply the initialization function <init_func>
        pass

    def adaptive_load_nets(self, net, model_weight):
        model_dict = net.state_dict()
        pretrained_dict = {k : v for k, v in model_weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def load_nets(self, cfg, writer, logger):    # load pretrained weights on the net
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            _k = -1
            for net in self.nets:
                name = net.__class__.__name__
                _k += 1
                if checkpoint.get(name) == None:
                    continue
                if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                    continue
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if cfg['training']['optimizer_resume']:
                    self.adaptive_load_nets(self.optimizers[_k], checkpoint[name]["optimizer_state"])
                    self.adaptive_load_nets(self.schedulers[_k], checkpoint[name]["scheduler_state"])
            self.iter = checkpoint["iter"]
            self.best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["iter"]
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(cfg['training']['resume']))


    def load_PredNet(self, cfg, writer, logger, dir=None, net=None):    # load pretrained weights on the net
        dir = dir or cfg['training']['Pred_resume']
        best_iou = 0
        if os.path.isfile(dir):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(dir)
            )
            checkpoint = torch.load(dir)
            name = net.__class__.__name__
            if checkpoint.get(name) == None:
                return
            if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                return
            self.adaptive_load_nets(net, checkpoint[name]["model_state"])
            iter = checkpoint["iter"]
            best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {}) (best iou {}) for PredNet".format(
                    dir, checkpoint["iter"], best_iou
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(dir))
        if hasattr(net, 'best_iou'):
            net.best_iou = best_iou
        return best_iou


    def set_optimizer(self, optimizer):  #set optimizer to all nets
        pass

    def reset_objective_SingleVector(self,):
        self.objective_vectors = np.zeros([19, 256])
        self.objective_vectors_num = np.zeros([19])
        self.objective_vectors_dis = np.zeros([19, 19])

    def update_objective_SingleVector(self, id, vector, name='moving_average', ):
        if isinstance(vector, torch.Tensor):
            vector = vector.squeeze().detach().cpu().numpy()
        if np.sum(vector) == 0:
            return
        if self.objective_vectors_num[id] < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * 0.9999 + 0.0001 * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

def grad_reverse(x):
    return GradReverse()(x)