#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : wgan_gp_resnet.py
# @Author: Piston Yang
# @Date  : 18-8-24
from resnet import ResnetGenerator, ResnetDiscriminator
import matplotlib as mpl
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from mxnet import autograd
import logging
from datetime import datetime
import os
import time
from gan_datasets import tcy_dataset

mpl.use('Agg')
from matplotlib import pylab as plt

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

logging.basicConfig(level=logging.DEBUG)
nz = 128
ngf = 64
ndf = 64
ctx = mx.gpu()
outf = './result_w_resnet'

batch_size = 64
lr = 1e-4
beta1 = 0
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
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False)

dm_train = gluon.data.DataLoader(
    tcy_dataset(root_path='/home/piston/DataSets/faces', transform=transformer),
    batch_size=batch_size, shuffle=True, num_workers=3, last_batch='discard'
)

# build the generator
netG = ResnetGenerator(dim=ngf)

# build the discriminator
netD = ResnetDiscriminator(dim=ndf)

netG.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
netD.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

# lr_s = mx.lr_scheduler.FactorScheduler(5000, 0.1, 1e-8)

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': 0.9})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': 0.9})

metric = mx.metric.Loss('Wasserstein_D')
print('Training...')
stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, ctx):
    with autograd.pause():
        b_s = real_data.shape[0]
        alpha = nd.random.uniform(0, 1, shape=(b_s, 1, 1, 1), ctx=ctx)
        alpha = alpha.broadcast_to(real_data.shape)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = nd.array(interpolates, ctx=ctx)
    interpolates.attach_grad()
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(heads=disc_interpolates, variables=interpolates,
                              head_grads=nd.ones(shape=disc_interpolates.shape, ctx=ctx),
                              create_graph=True, retain_graph=True)[0]

    gradients = gradients.reshape((gradients.shape[0], -1))
    gradient_penalty = ((gradients.norm(2, axis=1) - 1) ** 2) * LAMBDA
    gradient_penalty.attach_grad()
    return gradient_penalty


iter = 0
for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    for data, _ in dm_train:
        ############################
        # (1) Update D network: maximize D(x) - D(G(z)) - lambda * gradient_penalty
        ###########################

        data = data.as_in_context(ctx)
        noise = mx.nd.random.normal(0, 1., shape=(batch_size, nz, 1, 1), ctx=ctx)

        with autograd.record():
            output = netD(data)
            output = output.reshape((batch_size, 1))
            errD_real = output

            fake = netG(noise)
            output = netD(fake.detach())
            output = output.reshape((batch_size, 1))
            errD_fake = output
            # errD = errD_fake - errD_real
            # errD = -errD.mean()
            # errD.backward()
            gradient_penalty = calc_gradient_penalty(netD, data.detach(), fake.detach(), 10, ctx)

            assert errD_real.shape == errD_fake.shape == gradient_penalty.shape, "Shape Error"
            errD = errD_real - errD_fake - gradient_penalty
            errD = -errD
            # gradient_penalty.backward()
            errD.backward()

        trainerD.step(batch_size)

        D_cost = errD_real.mean() - errD_fake.mean() - gradient_penalty.mean()
        Wasserstein_D = errD_real.mean() - errD_fake.mean()
        metric.update(0, D_cost)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        with autograd.record():
            output = netD(fake)
            output = output.reshape((-1, 1))
            errG = -output
            # errG = -errG
            errG.backward()
        trainerG.step(batch_size)
        name, dis = metric.get()
        logging.info(
            'discriminator loss = %f, generator loss = %f, gradient_penalty = %f, D_cost = %f at iter %d epoch %d' %
            (nd.mean(D_cost).asscalar(), nd.mean(errG).asscalar(), gradient_penalty.mean().asscalar(), D_cost.asscalar(), iter,
             epoch))

        if iter % 100 == 99:
            visual('gout', fake.asnumpy(), name=os.path.join(outf, 'fake_img_iter_%d.png' % iter))
            visual('data', data.asnumpy(), name=os.path.join(outf, 'real_img_iter_%d.png' % iter))

        iter = iter + 1
        btic = time.time()

    name, dis = metric.get()
    metric.reset()

    logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, dis))
    logging.info('time: %f' % (time.time() - tic))

netG.save_parameters(os.path.join(outf, 'generator.params'))
netD.save_parameters(os.path.join(outf, 'discriminator.params'))
