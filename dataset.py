#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class PathDataset(Dataset):
    def __init__(self, root, alphabetChinese, transform=None, target_transform=None):
        """
        加载本地目录图片
        目录结构：
        dataset/images 存放文本行图片
        dataset/labels.txt 存放每张文本行图像对应的真值字符串，一个样本一行，每行格式如下：
                            pic_0001.jpg 你好中国

        """

        self.root = root
        self.jpgPaths = os.listdir(os.path.join(root, "images"))
        self.jpgPaths.sort()
        self.nSamples = len(self.jpgPaths)

        with open(os.path.join(root, "labels.txt"), "r") as f:
            self.labels = f.readlines()
        self.labels = [x.strip().split(' ') for x in self.labels]
        self.labels = sorted(self.labels, key=lambda x: x[0])

        self.alphabetChinese = alphabetChinese
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index >= len(self):
            index = 0

        imP = os.path.join(self.root, "images", self.jpgPaths[index])
        im = Image.open(imP).convert('L')

        label = self.labels[index][1]
        label = ''.join([x for x in label if x in self.alphabetChinese])

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return im, label


class SytheticChinese(Dataset):
    def __init__(self, root, phase="train", transform=None, target_transform=None):
        """

        :param root: 数据集目录
        :param phase: 'test' or 'train' 指定数据用于训练还是测试
        :param transform: 
        :param target_transform:

        ---------------------------------------------------------        
        目录结构：
        root/images 存放文本行图片
        root/char_std_5990.txt 字符集，一个字符一行
        root/data_test.txt 测试的样本名及label
        root/data_train.txt 训练的样本名及label。一个样本一行，每行格式如下：
                            pic_0001.jpg label_id1 label_id2 ...
        """
        """


        """
        self.root = root

        with open(os.path.join(root, "char_std_5990.txt"), 'r') as f:
            self.charset = f.readlines()
        self.charset = [x.strip() for x in self.charset]

        with open(os.path.join(root, "data_%s.txt" % phase), 'r') as f:
            self.sample_labels = f.readlines()

        self.nSamples = len(self.sample_labels)

        self.transform = transform
        self.target_transform = target_transform

    @property
    def alphabet(self):
        return "".join(self.charset[1:])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index >= len(self):
            index = 0

        line_str = self.sample_labels[index].strip().split(' ')
        jpg_file = os.path.join(self.root, 'images', line_str[0])
        if not os.path.exists(jpg_file):
            print("warning: file %s not exist." % jpg_file)

        im = Image.open(jpg_file).convert('L')

        label = ""
        for char_id in line_str[1:]:
            label += self.charset[int(char_id)]

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return im, label


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        """
        将一个batch的图片转为tenor。支持不定长图像batch，输出的高度为设置值(32）,宽度取batch中最大值(寬高比最大值*预设的高度),
        靠左对齐，宽度不够的填充默认0
        :param batch: 
        :return: 
        """
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
        imgW = int(np.floor(max_ratio * imgH))
        imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        batch_size = len(images)
        out_images = np.zeros([batch_size, 1, imgH, imgW])

        for i, image in enumerate(images):
            w, h = image.size
            newW = int(w * imgH/h)
            image = image.resize((newW, imgH), Image.BILINEAR)
            image_arr = np.array(image, dtype=np.float32)
            image_arr = image_arr/255.0
            image_arr = image_arr -0.5/0.5
            print('actual w ', imgH, newW)
            out_images[i, 0, :,:newW] = image_arr
        print('out batch ', out_images.shape)
        out_tensor = torch.FloatTensor(out_images)
        return out_tensor, labels
