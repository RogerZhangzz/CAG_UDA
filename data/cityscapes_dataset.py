import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from .utils import recursive_glob
from augmentations import *
from data.base_dataset import BaseDataset

import random


class Cityscapes_loader(BaseDataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        cfg,
        writer,
        logger,
        augmentations = None,
    ):
        """__init__

        :param cfg: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations: 
        """
        
        self.cfg = cfg
        self.root = cfg['rootpath']
        self.split = cfg['split']
        self.is_transform = cfg.get('is_transform', True)
        self.augmentations = augmentations
        self.img_norm = cfg.get('img_norm', True)
        self.n_classes = 19
        self.img_size = (
            cfg['img_cols'], cfg['img_rows']
        )
        self.mean = np.array(self.mean_rgb['cityscapes'])
        self.files = {}
        self.paired_files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files = recursive_glob(rootdir=self.images_base, suffix=".png") #find all files from rootdir and subfolders with suffix = ".png"
        if cfg.get('shuffle'):
            np.random.shuffle(self.files)

        self.distribute = np.zeros(self.n_classes, dtype=float)
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
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
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))   #zip: return tuples

        if not self.files:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files), self.split))
        # self.fun1(self.files)
        # self.calc_distribution(writer)
        if self.cfg.get('whether_paired') == True:
            self.load_paired_imgname(cfg)
    
    def load_paired_imgname(self, cfg):
        path = cfg['dict']
        if os.path.isfile(path):
            dict1 = torch.load(path)
            self.paired_files = dict1
            print('successfully load dict at {}'.format(path))
        else:
            raise Exception('no dict file at {} found!'.format(path))
        pass

    def get_paired_imgname(self, img_name, batchsize=1):
        output = []
        for i in range(batchsize):
            name = img_name[i]
            namenew = ''
            for u in range(6, len(name.split('/'))):
                namenew = os.path.join(namenew, name.split('/')[u])
            length = len(self.paired_files[namenew])
            candidate_list = list(range(length))
            index = random.sample(candidate_list, k=batchsize)
            out = []
            for i in range(batchsize):
                out.append(self.paired_files[namenew][index[i]])
            pass
            output.append(out)
        # return self.paired_files[img_name][index]
        return out

    def update_pairlist(self, candidate_list, filename):
        if isinstance(filename, list):
            filename = filename[0]
        if self.paired_files.get(filename) != None:
            print('error! the name {} has been allocated candidate list!'.format(filename))
        self.paired_files[filename] = candidate_list
        pass

    def find_pairs(self, index):
        pass

    def fun1(self, parameter_list):
        for id in tqdm(self.ids):
            filename = '{:05d}.png'.format(id)
            img_path = os.path.join(self.image_base_path, filename)
            lbl_path = os.path.join(self.label_base_path, filename)

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
            unique, counts = np.unique(lbl, return_counts=True)
            lbl_dict = dict(zip(unique, counts))
            deactive_list = []
            for label_id, label_counts in lbl_dict.items():
                if label_counts <= 120:
                    deactive_list.append(label_id)
                    lbl = self.remove_label(lbl, label_id)
            lbl_img = Image.fromarray(lbl)
            lbl_img.save(os.path.join(self.root,'labels_new',filename))
        pass

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations!=None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, self.files[index]

    def calc_distribution(self, writer,):
        for path in tqdm(self.files):
            img_path = path.rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
            distribute = np.zeros(self.n_classes, dtype=float)
            for u in range(lbl.shape[0]):
                for v in range(lbl.shape[1]):
                    if lbl[u, v] == self.ignore_index:
                        continue
                    id = lbl[u, v]
                    distribute[id] += 1
            # for id in range(self.n_classes):
            #     self.distribute[id] += 1.0*distribute[id]/lbl.size()
            distribute = distribute*100.0/lbl.size
            self.distribute = np.sum([self.distribute, distribute], axis=0)
        self.distribute = self.distribute/len(self.files)
        for id in range(self.n_classes):
            writer.add_scalar('distribute/cityscapes-'+self.split, self.distribute[id], id+1)
        pass

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):   #todo: understanding the meaning 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
