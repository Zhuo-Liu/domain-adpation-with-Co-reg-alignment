import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from deep_model.model import build_small_model
from trainer import Trainer
from data.data_utils import get_mnist, get_svhn
from data.data_loader import get_data_loader, get_lmdb_loader
from config import Configs, MODEL_INDEX

import argparse

from matplotlib import pyplot as plt


import data.data_utils as dd
from scipy import io
import torchvision

import csv
import pandas as pd

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, required=True)
    parser.add_argument('--num_net', type=int, required=True)
    parser.add_argument('--test_only', action='store_true')
    return parser.parse_args()


def load_weights(net, path, gpu=True):
    if gpu:
        net.load_state_dict(torch.load(path))
    else:
        net.load_state_dict(torch.load(path, map_location='cpu'))

    return net
        

def main(model_index=MODEL_INDEX):
    #args = get_argument()
    config = Configs(model_index)
    net = build_small_model(config.ins_norm, False if config.lambda_div == 0 else True)

    #if config.mode in [0, -1] and not args.test_only:
    if config.mode in [0, -1]:
        config.dump_to_file(os.path.join(config.save_path, 'exp_config.txt'))

        train_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'data', 'label', batch_size=config.batch_size)
        val_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'vdata', 'vlabel', batch_size=config.batch_size)
        test_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'tdata', 'tlabel', batch_size=config.batch_size)

        net_list = [net]
        for idx in range(1, 2):
            tmp_net = build_small_model(config.ins_norm, False if config.lambda_div == 0 else True)
            net_list.append(tmp_net)


        trainer = Trainer(net_list, train_data_loader, val_data_loader, test_data_loader, config)

        trainer.train_all()

    # elif config.mode == 1 or args.test_only:
    #     #tdata, tlabel, tdata_test, tdata_test_label = get_svhn('D:/workspace/dataset/digits/SVHN/')
    #     #tdata, tlabel, tdata_test, tdata_test_label = get_mnist('D:/workspace/DA/dataset/MNIST/')
    #     test_data_loader = get_data_loader(tdata_test, tdata_test_label[:, 0], tdata_test, tdata_test_label[:, 0], shuffle=False, batch_size=32)
        
    #     load_weights(net, config.checkpoint, config.gpu)
        
    #     trainer = Trainer([net], None, None, test_data_loader, config)
    #     print('acc', trainer.val_(trainer.nets[0], 0, 0, 'test'))

def plot_samples():
    mnist_path = os.path.join('./data/dataset/MNIST/', 'train-images.idx3-ubyte')
    mnist_img = np.array(dd.load_mnist_image(mnist_path))
    mnist_label_path = os.path.join('./data/dataset/MNIST/', 'train-labels.idx1-ubyte')
    mnist_label = np.array(dd.load_mnist_label(mnist_label_path))

    svhn_path = os.path.join('./data/dataset/SVHN/', 'train_32x32.mat')
    svhn_img = np.transpose(io.loadmat(svhn_path)['X'], (3, 0, 1, 2))
    svhn_label = io.loadmat(svhn_path)['y']
    svhn_label[svhn_label == 10] = 0

    transform = torchvision.transforms.Resize(32)
    mnistm_path = os.path.join('./data/dataset/MNIST-M', 'mnist_m_train.pt')
    mnistm_img, mnistm_label = torch.load(mnistm_path)
    mnistm_img = mnistm_img.permute(0, 3, 1, 2)  # 60000, 28, 28, 3 -> 60000, 3, 28, 28
    mnistm_img = transform(mnistm_img)
    mnistm_img = mnistm_img.permute(0, 2, 3, 1)
    mnistm_img = mnistm_img.numpy()
    mnistm_label = mnistm_label.numpy()

    #MNIST
    figure = plt.figure(figsize=(8, 4))
    cols, rows = 4, 2
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if(i>=5):
            img = mnist_img[i-1-4]
            img = np.expand_dims(img, -1)
            img = instance_normalization(img)
            img = refine(img)
            plt.title(mnist_label[i-1-4])
        else:
            img = mnist_img[i-1]
            plt.title(mnist_label[i-1])
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.imshow(img.squeeze(),cmap='gray')
    plt.show()

    #SVHN
    figure = plt.figure(figsize=(8, 4))
    cols, rows = 4, 2
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if(i>=5):
            img = svhn_img[i-1-4]
            img = instance_normalization(img)
            img = refine(img)
            plt.title(svhn_label[i-1-4])
        else:
            img = svhn_img[i-1]
            plt.title(svhn_label[i-1])
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.imshow(img.squeeze(),cmap='gray')
    plt.show()

    #MNISTM
    figure = plt.figure(figsize=(8, 4))
    cols, rows = 4, 2
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if(i>=5):
            img = mnistm_img[i-1-4]
            img = instance_normalization(img)
            img = refine(img)
            plt.title(mnistm_label[i-1-4])
        else:
            img = mnistm_img[i-1]
            plt.title(mnistm_label[i-1])
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.imshow(img.squeeze(),cmap='gray')
    plt.show()

# input should be 32x32Xn
def instance_normalization(input):
    H,W,C = input.shape
    #mean
    mu = np.zeros(C)
    for i in range(C):
        mu[i] = np.average(input[:,:,i].flatten())
    
    #deviation
    sigma = np.zeros(C)
    for i in range(C):
        sigma[i] = np.average((input[:,:,i] - mu[i])**2)

    output = np.zeros((input.shape))
    for i in range(C):
        output[:,:,i] = (input[:,:,i] - mu[i]) / np.sqrt(sigma[i]*sigma[i] + 1e-5)
    
    return output

def refine(input):
    H,W,C = input.shape
    output =  np.zeros((input.shape))

    for i in range(C):
        mini = np.amin(input[:,:,i])
        maxi = np.amax(input[:,:,i])
        normalized = (input[:,:,i]-mini)/(maxi-mini) * 255.0
        output[:,:,i] = normalized

    output = output.astype(int)
    return output

def plot_loss():
    TSBOARD_SMOOTHING = 0.95

    m2s_coda_agreeloss = './m2s/run-model_0_tensorboard_exp_1637680592-tag-agree_loss.csv'
    m2s_coda_ccloss = './m2s/run-model_0_tensorboard_exp_1637680592-tag-net_0_CCloss.csv'
    m2mm_coda_agreeloss = './m2s/run-model_m_mm_tensorboard_exp_1638245006-tag-agree_loss.csv'
    m2mm_coda_ccloss = './m2s/run-model_m_mm_tensorboard_exp_1638245006-tag-net_0_CCloss.csv'

    m2s_vada_agreeloss = './m2s/run-model_2_tensorboard_exp_1638146765-tag-agree_loss.csv'

    df_m2s_coda_agreeloss = pd.read_csv(m2s_coda_agreeloss)
    df_m2s_coda_agreeloss = df_m2s_coda_agreeloss[df_m2s_coda_agreeloss['time'] <= 20000]
    smooth_m2s_coda_agreeloss = df_m2s_coda_agreeloss.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    df_m2s_coda_ccloss = pd.read_csv(m2s_coda_ccloss)
    df_m2s_coda_ccloss = df_m2s_coda_ccloss[df_m2s_coda_ccloss['time'] <= 20000]
    smooth_m2s_coda_ccloss = df_m2s_coda_ccloss.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

    df_m2s_vada_agreeloss = pd.read_csv(m2s_vada_agreeloss)
    df_m2s_vada_agreeloss = df_m2s_vada_agreeloss[df_m2s_vada_agreeloss['time'] <= 20000]
    smooth_m2s_vada_agreeloss = df_m2s_vada_agreeloss.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

    df_m2mm_coda_agreeloss = pd.read_csv(m2mm_coda_agreeloss)
    df_m2mm_coda_agreeloss = df_m2mm_coda_agreeloss[df_m2mm_coda_agreeloss['time'] <= 20000]
    smooth_m2mm_coda_agreeloss = df_m2mm_coda_agreeloss.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    df_m2mm_coda_ccloss = pd.read_csv(m2mm_coda_ccloss)
    df_m2mm_coda_ccloss = df_m2mm_coda_ccloss[df_m2mm_coda_ccloss['time'] <= 20000]
    smooth_m2mm_coda_ccloss = df_m2mm_coda_ccloss.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

    plt.figure(figsize=(10,8))
    plt.plot(df_m2s_coda_agreeloss["time"], df_m2s_coda_agreeloss["value"], alpha=0.4,color='red')
    plt.plot(df_m2s_coda_agreeloss["time"], smooth_m2s_coda_agreeloss["value"],label='Co-DA',color='red')
    plt.plot(df_m2s_vada_agreeloss["time"], df_m2s_vada_agreeloss["value"], alpha=0.4,color='blue')
    plt.plot(df_m2s_vada_agreeloss["time"], smooth_m2s_vada_agreeloss["value"],label='VADA',color='blue')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=24,loc='lower left')
    plt.xlim(0,20000)
    plt.ylim(0.5,1.42)
    plt.ylabel('Agree_loss', fontsize=22)
    plt.xlabel('Iterations', fontsize=22)
    plt.show()
    plt.figure(figsize=(10,8))
    plt.plot(df_m2s_coda_agreeloss["time"], df_m2s_coda_agreeloss["value"]/df_m2s_coda_ccloss["value"], alpha=0.4,color='red')
    plt.plot(df_m2s_coda_agreeloss["time"], smooth_m2s_coda_agreeloss["value"]/smooth_m2s_coda_ccloss["value"],label='M $\\rightarrow$ S',color='red')
    plt.plot(df_m2mm_coda_agreeloss["time"], df_m2mm_coda_agreeloss["value"]/df_m2mm_coda_ccloss["value"], alpha=0.4,color='blue')
    plt.plot(df_m2mm_coda_agreeloss["time"], smooth_m2mm_coda_agreeloss["value"]/smooth_m2mm_coda_ccloss["value"],label='M $\\rightarrow$ MM',color='blue')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=24,)
    plt.xlim(0,18000)
    #plt.ylim(0.5,1.)
    plt.ylabel('Agree_loss/Overall_loss', fontsize=22)
    plt.xlabel('Iterations', fontsize=22)
    plt.show()

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    if type(MODEL_INDEX) is list:
        for model_index in MODEL_INDEX:
            main(model_index)
    else:
        main(MODEL_INDEX)

    #plot_samples()
    #plot_loss()