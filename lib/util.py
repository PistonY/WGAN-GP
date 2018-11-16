#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util.py
# @Author: Piston Yang
# @Date  : 18-8-27
# refer to https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import mxnet as mx
from mxnet import nd
from mxnet.gluon.model_zoo.vision import inception_v3
import numpy as np
from scipy.stats import entropy
import os
from mxnet.gluon.data import Dataset, DataLoader
import cv2

import matplotlib

matplotlib.use('Agg')

# if in China
os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'


class lsun_bedroom(Dataset):
    def __init__(self, map_path, transform=None, **kwargs):
        super(lsun_bedroom, self).__init__(**kwargs)
        self.map_path = map_path
        self.file_map = self._get_buff()
        self.transform = transform

    def _get_buff(self):
        map_list = []
        filenames = os.listdir(self.map_path)
        for filename in filenames:
            with open(os.path.join(self.map_path, filename), 'r') as f:
                lines = f.readlines()
                map_list += lines
        return map_list

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, item):
        file_path = self.file_map[item].replace('\n', '')
        im = cv2.imread(file_path)

        if im is None:
            print(item)
        # assert im is not None, 'Got None from cv2'
        if self.transform:
            return self.transform(nd.array(im), np.array([0]))
        return nd.array(im), np.array([0])


def inception_score(imgs, use_gpu=True, batch_size=32, resize=False, splits=1):
    """
    Computes the inception score of the generated images imgs
    :param imgs: nd dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    :param use_gpu: whether or not to run on GPU
    :param batch_size: batch size for feeding into Inception v3
    :param resize:
    :param splits: number of splits
    :return:
    """

    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # Set up device
    ctx = mx.gpu(0) if use_gpu else mx.cpu()

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, ctx=ctx)

    def up(data):
        output = nd.contrib.BilinearResize2D(data, height=299, width=299)
        return output

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return nd.softmax(x).asnumpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.as_in_context(ctx)
        batch_size_i = batch.shape[0]
        preds[i * batch_size: i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)


def mk_bdm_lst(path, sp_num=None, outf=None):
    outf = outf if outf else './data/lsun_bedroom/'
    sp_num = sp_num if sp_num else 5e5
    lst = ""
    if not os.path.exists(outf):
        os.makedirs(outf)
    for i, fullname in (enumerate((iterbrowse(path)))):
        print(i)
        lst += fullname + '\n'
        if i % sp_num == 0 and i > 0:
            file_name = os.path.join(outf, 'lsun_bedroom{}.txt'.format(int(i // sp_num)))
            print(file_name)
            with open(file_name, 'w') as f:
                f.write(lst)
            lst = ""


if __name__ == '__main__':
    # class IgnoreLabelDataset(Dataset):
    #     def __init__(self, orig):
    #         self.orig = orig
    #
    #     def __getitem__(self, item):
    #         return self.orig[item][0]
    #
    #     def __len__(self):
    #         return len(self.orig)
    #
    #
    # cifar = mx.gluon.data.vision.CIFAR10(root='./data', transform=image.CreateAugmenter(data_shape=(3, 32, 32),
    #                                                                                     mean=True, std=True))
    #
    # IgnoreLabelDataset(cifar)
    #
    # print("Calculating Inception Score...")
    # print(inception_score(IgnoreLabelDataset(cifar), use_gpu=True, batch_size=32, resize=True, splits=10))
    # mk_bdm_lst('/home/piston/EXTVOL/bedroom')
    lsun_da = lsun_bedroom('data/lsun_bedroom/')
    # print(len(lsun_da))
    # for i, _ in enumerate(lsun_da):
    #     # print(i)
    #     if i == 3000000:
    #         break
    # lsum_train = DataLoader(
    #     lsun_bedroom('/home/piston/EXTVOL/bedroom/', 'data/lsun_bedroom/'),
    #     batch_size=64, shuffle=True, num_workers=0, last_batch='discard'
    # )
    # for bt, _ in lsum_train:
    #     print(bt)
    #     break
    print(lsun_da[np.random.randint(low=0, high=3e6)])
