#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: eval_crnn.py
@time: 19-1-21 下午6:35
@brief： 
"""
import argparse
from collections import OrderedDict

import sys, os

from dataset import PathDataset, alignCollate, SytheticChinese
from models.crnn import CRNN
from models import keys
from utils import averager, loadData, strLabelConverter

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import cv2
import torch
from torch.autograd import Variable



def val(model, converter, data_loader, max_iter=100):
    print('Start val')

    # input tensor
    image = torch.FloatTensor(opt.batch_size, 3, imgH, imgH)
    image = image.cuda()

    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        print('-------\ninput ', cpu_images.size())
        batch_size = cpu_images.size(0) #30个
        loadData(image, cpu_images)

        preds = model(image) #[483*10*]
        print('out ', preds.size())

        preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
        print("len ", preds_size.data)

        _, preds = preds.max(2)

        preds = preds.contiguous().view(-1)
        print("out preds ", preds.size())
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        if batch_size==1:
            sim_preds = [sim_preds]
        for pred, target in zip(sim_preds, cpu_texts):
            print("pred ", pred, 'gt ', target)
            if pred == target.lower():
                print('true')
                n_correct += 1
            else:
                print('false')


        # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:10]
        # for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        #     print('%-20s => %-20s, gt: %-20s\n' % (raw_pred, pred, gt))

        # img = cpu_images.numpy()[0]
        # img = np.squeeze(img)
        # if len(img.shape) == 3 and img.shape[2] != 3:
        #     img = img.transpose((1, 2, 0))
        # cv2.imshow("im", img)
        # cv2.waitKey(0)

    accuracy = n_correct / float(max_iter * opt.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valRoot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--ngpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")

    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    print(opt)

    ngpu = opt.ngpus
    imgH = 32
    imgW = 280
    keep_ratio = True

    # testdataset = PathDataset(opt.valRoot, alphabetChinese)

    testdataset = SytheticChinese(opt.valRoot, 'test')
    val_loader = torch.utils.data.DataLoader(testdataset,
                                             shuffle=False,
                                             batch_size=opt.batch_size,
                                             num_workers=int(opt.workers),
                                             collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

    alphabet = keys.alphabetChinese
    print("char num ", len(alphabet))
    model = CRNN(32, 1, len(alphabet) + 1, 256, 1)

    converter = strLabelConverter(''.join(alphabet))

    state_dict = torch.load("../SceneOcr/model/ocr-lstm.pth", map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if "num_batches_tracked" not in k:
            # name = name.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2])


    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    curAcc = val(model, converter, val_loader, max_iter=5)