#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dcgan.py
# @Author: Piston Yang
# @Date  : 18-8-20

import matplotlib as mpl
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
import logging
from datetime import datetime
import os
import time
mpl.use('Agg')
from matplotlib import pylab as plt
os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

logging.basicConfig(level=logging.DEBUG)
nz = 100
ngf = 64
ndf = 64
nc = 3
ctx = mx.gpu()
outf = './result'

batch_size = 64
lr = 0.0002
beta1 = 0.5
epochs = 100

if not os.path.exists(outf):
    os.makedirs(outf)


def fill_buf(buf, i, img, shape):
    n = buf.shape[0] // shape[1]
    m = buf.shape[1] // shape[0]

    sx = (i % m) * shape[0]
    sy = (i // m) * shape[1]
    buf[sy:sy + shape[1], sx:sx + shape[0], :] = img
    return None


def visual(title, X, name):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X - np.min(X)) * (255.0 / (np.max(X) - np.min(X))), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n * X.shape[1]), int(n * X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = buff[:, :, ::-1]
    plt.imshow(buff)
    plt.title(title)
    plt.savefig(name)


def transformer(data, label):
    data = mx.image.imresize(data, 64, 64)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128 - 1
    if data.shape[0] == 1:
        data = mx.nd.tile(data, (3, 1, 1))
    return data, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False)

# build the generator
netG = nn.HybridSequential()
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*4) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*2) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('sigmoid'))
    # state size. (nc) x 64 x 64

# build the discriminator
netD = nn.HybridSequential()
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf*2) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf*4) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf*2) x 4 x 4
    netD.add(nn.Conv2D(2, 4, 1, 0, use_bias=False))

loss = gluon.loss.SoftmaxCrossEntropyLoss()

netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

real_label = mx.nd.ones((batch_size,), ctx=ctx)
fake_label = mx.nd.zeros((batch_size,), ctx=ctx)

metric = mx.metric.Accuracy()
print('Training...')
stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

iter = 0
for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    for data, _ in train_data:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = data.as_in_context(ctx)
        noise = mx.nd.random.normal(0, 1, shape=(batch_size, nz, 1, 1), ctx=ctx)

        with autograd.record():
            output = netD(data)
            output = output.reshape((batch_size, 2))
            errD_real = loss(output, real_label)
            metric.update([real_label, ], [output, ])
            fake = netG(noise)
            output = netD(fake.detach())
            output = output.reshape((batch_size, 2))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label, ], [output, ])
        trainerD.step(batch_size)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        with autograd.record():
            output = netD(fake)
            output = output.reshape((-1, 2))
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size)

        name, acc = metric.get()
        logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d' %
                     (mx.nd.mean(errD).asscalar(), mx.nd.mean(errG).asscalar(), acc, iter, epoch))

        if iter % 1000 == 0:
            visual('gout', fake.asnumpy(), name=os.path.join(outf, 'fake_img_iter_%d.png' % iter))
            visual('data', data.asnumpy(), name=os.path.join(outf, 'real_img_iter_%d.png' % iter))

        iter = iter + 1
        btic = time.time()

    name, acc = metric.get()
    metric.reset()

    logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    logging.info('time: %f' % (time.time() - tic))

netG.save_parameters(os.path.join(outf, 'generator.params'))
netD.save_parameters(os.path.join(outf, 'discriminator.params'))