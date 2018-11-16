#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gan_datasets.py
# @Author: Piston Yang
# @Date  : 18-8-21

import os
from mxnet.gluon.data import Dataset
from mxnet import nd
import cv2


class tcy_dataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.root_path = root_path
        self._file_name = self._get_file_name()

    def _get_file_name(self):
        file_name = os.listdir(self.root_path)
        return file_name

    def __len__(self):
        return len(self._file_name)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_path, self._file_name[item])
        data = cv2.imread(img_name)
        data = nd.array(data)
        if self.transform:
            return self.transform(data, 0)
        return data, 0


if __name__ == '__main__':
   ds = tcy_dataset(root_path='/home/piston/DataSets/faces')
   for data, _ in ds:
       print(data)
       break