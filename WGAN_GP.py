#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : WGAN_GP.py
# @Author: Piston Yang
# @Date  : 18-8-20

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
# from inception_score import get_inception_score as gis
from lib.util import inception_score as icc

mpl.use('Agg')
from matplotlib import pylab as plt

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

logging.basicConfig(level=logging.DEBUG)
nz = 128
ngf = 64
ndf = 64
nc = 3
ctx = mx.gpu()
outf = './result_w_32'

batch_size = 64
lr = 1e-4
beta1 = 0.5
ITERS = 200000
CRITIC_ITERS = 1

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


# def get_inception_score_tf(G, ctx):
#     all_samples = []
#     for i in range(10):
#         samples_100 = nd.random_normal(0, 1, shape=(100, nz, 1, 1), ctx=ctx)
#         all_samples.append(G(samples_100).as_in_context(mx.cpu()).asnumpy())
#     all_samples = np.concatenate(all_samples, axis=0)
#     all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
#     all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
#     return gis(list(all_samples))


def get_inception_score_gl(G, ctx):
    all_samples = []
    for i in range(10):
        samples_100 = nd.random_normal(0, 1, shape=(100, nz, 1, 1), ctx=ctx)
        all_samples.append(G(samples_100).as_in_context(mx.cpu()).asnumpy())
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.add(np.multiply(all_samples, 0.5), 0.5)
    all_samples = all_samples.reshape((-1, 3, 32, 32))
    return icc(list(all_samples), resize=True, splits=10)


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, ctx):
    real_data = real_data.as_in_context(ctx)
    b_s = real_data.shape[0]
    alpha = nd.random.uniform(0, 1, shape=(b_s, 1, 1, 1), ctx=ctx)
    alpha = alpha.broadcast_to(real_data.shape)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = nd.array(interpolates)
    interpolates.attach_grad()
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(heads=disc_interpolates, variables=interpolates,
                              head_grads=nd.ones(shape=disc_interpolates.shape, ctx=ctx),
                              create_graph=True, retain_graph=True)[0]

    gradients = gradients.reshape((gradients.shape[0], -1))
    gradient_penalty = ((gradients.norm(2, axis=1, keepdims=True) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def transformer(data, label):
    # resize to 64x64
    data = mx.image.imresize(data, 32, 32)
    # transpose from (64, 64, 3) to (3, 64, 64)
    data = mx.nd.transpose(data, (2, 0, 1))
    # normalize to [-1, 1]
    data = (data.astype(np.float32) / 255 - 0.5) / 0.5
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = mx.nd.tile(data, (3, 1, 1))
    return data, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False)

dm_train = gluon.data.DataLoader(
    tcy_dataset(root_path='/home/piston/DataSets/faces', transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')


def inf_train_gen():
    while True:
        for img, _ in dm_train:
            yield img


gen = inf_train_gen()


# build the generator
class Generator(nn.HybridBlock):
    def __init__(self, dim, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.dim = dim
        self.preprocess = nn.HybridSequential()
        with self.preprocess.name_scope():
            self.preprocess.add(nn.Conv2D(4 * 4 * 4 * dim, kernel_size=1, use_bias=False))
            self.preprocess.add(nn.BatchNorm())
            self.preprocess.add(nn.Activation('relu'))

        self.block1 = nn.HybridSequential()
        with self.block1.name_scope():
            self.block1.add(nn.Conv2DTranspose(2 * dim, kernel_size=2, strides=2, use_bias=False))
            self.block1.add(nn.BatchNorm())
            self.block1.add((nn.Activation('relu')))

        self.block2 = nn.HybridSequential()
        with self.block2.name_scope():
            self.block2.add(nn.Conv2DTranspose(dim, kernel_size=2, strides=2, use_bias=False))
            self.block2.add(nn.BatchNorm())
            self.block2.add((nn.Activation('relu')))

        self.deconv_out = nn.Conv2DTranspose(3, kernel_size=2, strides=2)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.preprocess(x)
        x = F.reshape(x, (-1, 4 * self.dim, 4, 4))
        x = self.block1(x)
        x = self.block2(x)
        x = self.deconv_out(x)
        x = F.Activation(x, act_type='tanh')
        x = F.reshape(x, (-1, 3, 32, 32))
        return x


netG = Generator(dim=ngf)


# build the discriminator
class Discriminator(nn.HybridBlock):
    def __init__(self, dim, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.dim = dim
        self.main = nn.HybridSequential()
        with self.main.name_scope():
            self.main.add(nn.Conv2D(dim, kernel_size=3, strides=2, padding=1))
            self.main.add(nn.LeakyReLU(0.2))
            self.main.add(nn.Conv2D(2 * dim, kernel_size=3, strides=2, padding=1))
            self.main.add(nn.LeakyReLU(0.2))
            self.main.add(nn.Conv2D(4 * dim, kernel_size=3, strides=2, padding=1))
            self.main.add(nn.LeakyReLU(0.2))
        self.linear = nn.Dense(1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.main(x)
        x = F.reshape(x, (-1, 4 * 4 * 4 * self.dim))
        x = self.linear(x)
        return x


netD = Discriminator(ndf)

netG.initialize(init=mx.init.Normal(0.02), ctx=ctx)
netD.initialize(init=mx.init.Normal(0.02), ctx=ctx)

one = nd.array([1.], ctx=ctx)
mone = -1 * one

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.5, 'beta2': 0.9})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': 0.5, 'beta2': 0.9})

print('Training...')
stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
for iteration in range(ITERS):
    tic = time.time()

    for i in range(CRITIC_ITERS):
        ############################
        # (1) Update D network: maximize D(x) - D(G(z)) - gradient_penalty
        ###########################

        data = next(gen)
        data = data.as_in_context(ctx)
        noise = mx.nd.random_normal(0, 1, shape=(batch_size, nz, 1, 1), ctx=ctx)
        with autograd.record():
            output = netD(data)
            errD_real = output.mean()

            fake = netG(noise)

            output = netD(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = calc_gradient_penalty(netD, next(gen), fake.detach(), 10, ctx)
            errD = errD_fake - errD_real + gradient_penalty
            errD.backward()

            # assert errD_fake.shape == errD_real.shape == gradient_penalty.shape, \
            #     print(errD_fake.shape, errD_real.shape, gradient_penalty.shape)

        trainerD.step(1)

    D_cost = -(errD_fake - errD_real + gradient_penalty)
    Wasserstein_D = errD_real - errD_fake

    ############################
    # (2) Update G network: maximize D(G(z))
    ###########################

    with autograd.record():
        output = netD(fake)
        errG = -output.mean()
        errG.backward()

    trainerG.step(1)
    G_cost = -errG

    if iteration % 100 == 99:
        logging.info('errD_fake %f, errD_real %f ' % (
            mx.nd.mean(errD_fake).asscalar(), mx.nd.mean(errD_real).asscalar()))
        logging.info(
            'discriminator loss = %f, generator loss = %f, gradient_penalty = %f, Wasserstein_D = %f, at iter %d.' %
            (D_cost.mean().asscalar(), G_cost.mean().asscalar(), gradient_penalty.mean().asscalar(),
             Wasserstein_D.mean().asscalar(), iteration))

    if iteration % 1000 == 999:
        visual('gout', fake.asnumpy(), name=os.path.join(outf, 'fake_img_iter_%d.png' % iteration))
        visual('data', data.asnumpy(), name=os.path.join(outf, 'real_img_iter_%d.png' % iteration))
        inception_score = get_inception_score_gl(netG, ctx)[0]
        # i1 = get_inception_score_gl(netG, ctx)[0]
        print("\ninception_score: %f." % (inception_score))

netG.save_parameters(os.path.join(outf, 'generator.params'))
netD.save_parameters(os.path.join(outf, 'discriminator.params'))
