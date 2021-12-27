import torch
from torch.utils.data import Dataset
from PIL import Image
from IPython import embed
import numpy as np
from torchvision import transforms
import os.path as osp


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar10unzip(file):
    """
    [10000 * 3072] -> [10000, 32, 32, 3]
    """
    d1 = unpickle(file)
    data = d1[b'data']
    label = d1[b'labels']
    name = d1[b'filenames']
    return data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1), label, name


if __name__ == '__main__':
    train_path = ['./data/cifar-10-batches-py/data_batch_1',
                  './data/cifar-10-batches-py/data_batch_2',
                  './data/cifar-10-batches-py/data_batch_3',
                  './data/cifar-10-batches-py/data_batch_4',
                  './data/cifar-10-batches-py/data_batch_5']
    test_path = './data/cifar-10-batches-py/test_batch'
    d0, d0_label, d0_name = cifar10unzip(train_path[0])
    d1, d1_label, d1_name = cifar10unzip(train_path[1])
    d2, d2_label, d2_name = cifar10unzip(train_path[2])
    d3, d3_label, d3_name = cifar10unzip(train_path[3])
    d4, d4_label, d4_name = cifar10unzip(train_path[4])

    test, test_label, test_name = cifar10unzip(test_path)
    train = np.concatenate((d0, d1, d2, d3, d4), axis=0)
    train_label = np.concatenate((d0_label, d1_label, d2_label, d3_label, d4_label), axis=0)
    train_name = np.concatenate((d0_name, d1_name, d2_name, d3_name, d4_name), axis=0)

    for index, pic in enumerate(train):
        cls = train_label[index]
        name = train_name[index]
        pic.reshape(32, 32, 3)
        img = Image.fromarray(pic.astype('uint8')).convert('RGB')
        path = osp.join('./data/cifar10', str(cls), str(name, encoding="utf-8"))
        img.save(path)


