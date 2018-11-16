#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gan_toy.py
# @Author: Piston Yang
# @Date  : 18-9-11

import random

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import mxnet as mx
from mxnet import autograd, nd, gluon
from mxnet.gluon import nn
from lib.plot import tick as ltick, plot as lplot, flush as lflush

mx.random.seed(1)

DATASET = '8gaussians'
DIM = 512
FIXED_GENERATOR = False
LAMBDA = .1
CRITIC_ITERS = 5
BATCH_SIZE = 256
ITERS = 100000
ctx = mx.gpu()


class Generator(nn.HybridBlock):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.HybridSequential()
        self.main.add(nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(2))

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.HybridBlock):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.HybridSequential()
        self.main.add(nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(DIM),
                      nn.Activation('relu'),
                      nn.Dense(1))

    def forward(self, inputs):
        output = self.main(inputs)
        return output


frame_index = [0]


def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    :param true_dist:
    :return:
    """

    N_POINTS = 128
    RANGE = 3
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = nd.array(points, ctx=ctx)
    disc_map = netD(points_v).asnumpy()

    noise = nd.random.normal(shape=(BATCH_SIZE, 2), ctx=ctx)
    true_dist = nd.array(true_dist, ctx=ctx)
    samples = netG(noise, true_dist).asnumpy()

    plt.clf()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    true_dist = true_dist.asnumpy()
    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    if not FIXED_GENERATOR:
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig('tmp/' + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')
    frame_index[0] += 1


# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':
        dataset = []
        for i in range(int(1e5 // 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) // BATCH_SIZE):
                yield dataset[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':
        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5
            yield data

    elif DATASET == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414
            yield dataset


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, ctx):
    real_data = real_data.as_in_context(ctx)
    b_s = real_data.shape[0]
    alpha = nd.random.uniform(0, 1, shape=(b_s, 1), ctx=ctx)
    alpha = alpha.broadcast_to(real_data.shape)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = nd.array(interpolates, ctx=ctx)
    interpolates.attach_grad()

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(heads=disc_interpolates, variables=interpolates,
                              head_grads=nd.ones(shape=disc_interpolates.shape, ctx=ctx),
                              create_graph=True, retain_graph=True, train_mode=True)[0]

    gradients = gradients.reshape((gradients.shape[0], -1))
    gradient_penalty = ((gradients.norm(2, axis=1, keepdims=True) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


netG = Generator()
netD = Discriminator()
netG.initialize(init=mx.init.Normal(0.02), ctx=ctx)
netD.initialize(init=mx.init.Normal(0.02), ctx=ctx)

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9})

data = inf_train_gen()

for iteration in range(ITERS):
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        real_data = nd.array(_data, ctx=ctx)
        noise = nd.random.normal(shape=(BATCH_SIZE, 2), ctx=ctx)
        with autograd.record():
            D_real = netD(real_data).mean()
            fake = netG(noise, real_data.detach())
            D_fake = netD(fake.detach()).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data.detach(), fake.detach(), LAMBDA, ctx)
            errD = D_fake - D_real + gradient_penalty
            errD.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        trainerD.step(1)

    noise = nd.random.normal(shape=(BATCH_SIZE, 2), ctx=ctx)
    if not FIXED_GENERATOR:
        with autograd.record():
            fake = netG(noise, 0)
            output = netD(fake)
            errG = -output.mean()
            errG.backward()

        trainerG.step(1)
        G_cost = -errG

    # Write logs and save samples
    lplot('tmp/' + DATASET + '/' + 'disc cost', D_cost.mean().asnumpy())
    lplot('tmp/' + DATASET + '/' + 'wasserstein distance', Wasserstein_D.mean().asnumpy())
    if not FIXED_GENERATOR:
        lplot('tmp/' + DATASET + '/' + 'gen cost', G_cost.mean().asnumpy())
    if iteration % 100 == 99:
        lflush()
        generate_image(_data)
    ltick()
