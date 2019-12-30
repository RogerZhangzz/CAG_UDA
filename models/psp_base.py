"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""

from .base_model import BaseModel
from . import networks
from .utils import *
import torch
import torch.nn as nn

class psp_base(nn.Module):

    pspnet_specs = {
            "res101": [3, 4, 23, 3],    #TODO: recover the config
            "res50": [3, 4, 6, 3],
    }

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        # if is_train:
        #     parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        return parser

    def __init__(self, cfg, cfg_model, writer, logger):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        super(psp_base,self).__init__()  # call the initialization method of BaseModel
        # # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        # self.loss_names = ['loss_G']
        # # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        # self.visual_names = ['data_A', 'data_B', 'output']
        # # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        # self.model_names = ['G']
        # # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # if self.isTrain:  # only defined during training time
        #     # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
        #     # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
        #     self.criterionLoss = torch.nn.L1Loss()
        #     # define and initialize optimizers. You can define one optimizer for each network.
        #     # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        #     self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizers = [self.optimizer]

        self.cfg = cfg
        self.cfg_model = cfg_model
        self.aux_loss = self.cfg_model['aux_loss']
        self.block_config = self.pspnet_specs[self.cfg_model['version']]  #TODO: what happends when () added
        
        self.n_classes = (
            cfg['data']['n_class']
        )
        # self.input_size = (       #TODO:  recover code
        #     pspnet_specs[version]["input_size"] if version is not None else input_size
        # )

        #devices id
        # self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device0 = torch.device("cpu")
        # device0 = self.device0
        # self.device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # device1 = torch.device("cpu")
        # self.device1 = device0
        # device1 = self.device1

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        # self.res_block5 = residualBlockPSP(self.block_config[3], 512, 256, 1024, 1, 2, include_range="identity")
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)

        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        # self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        if self.aux_loss:
            self.convbnrelu4_aux = conv2DBatchNormRelu(
                in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False
            )
            self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Define auxiliary loss function

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    # def set_input(self, input):
    #     """Unpack input data from the dataloader and perform necessary pre-processing steps.

    #     Parameters:
    #         input: a dictionary that contains the data itself and its metadata information.
    #     """
    #     AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
    #     self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
    #     self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
    #     self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self, x):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # self.output = self.netG(self.data_A)  # generate output image given the input data_A

        inp_shape = x.shape[2:]
        # device0 = self.device0.to(device1)
        # device1 = self.device1

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        #transfer data to cuda1
        x = self.res_block3(x)
        x = self.res_block4(x)
        # x = x.to(device1)

        # x = self.res_block5(x)

        if self.aux_loss:  # Auxiliary layers for training
            x_aux = self.convbnrelu4_aux(x)
            x_aux = self.dropout(x_aux)
            x_aux = self.aux_cls(x_aux)
            x_aux = F.interpolate(x_aux, size=inp_shape, mode='bilinear', align_corners=True)

        x = self.res_block5(x)
        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        # x = self.dropout(x)

        # x = self.classification(x)
        # x = F.interpolate(x, size=inp_shape, mode='bilinear', align_corners=True)

        if self.aux_loss:
            return (x, x_aux)
        else:  # eval mode
            return x

    # def backward(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
    #     # calculate loss given the input and intermediate results
    #     self.loss_G = self.criterionLoss(self.output, self.data_B) * self.opt.lambda_regression
    #     self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G

    # def optimize_parameters(self):
    #     """Update network weights; it will be called in every training iteration."""
    #     self.forward()               # first call forward to calculate intermediate results
    #     self.optimizer.zero_grad()   # clear network G's existing gradients
    #     self.backward()              # calculate gradients for network G
    #     self.optimizer.step()        # update gradients for network G
