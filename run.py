# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config
from corpus import Corpus

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Conditional Random Field(CRF) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--anneal', '-a', action='store_true', default=False,
                    dest='anneal', help='use simulated annealing')
parser.add_argument('--optimize', '-o', action='store_true', default=False,
                    dest='optimize', help='use feature extracion optimization')
parser.add_argument('--regularize', '-r', action='store_true', default=False,
                    dest='regularize', help='use L2 regularization')
parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                    dest='shuffle', help='shuffle the data at each epoch')
parser.add_argument('--file', '-f', action='store', dest='file',
                    help='set where to store the model')
args = parser.parse_args()

if args.optimize:
    from ocrf import CRF
else:
    from crf import CRF

# 根据参数读取配置
config = Config(args.bigdata)

print("Preprocess the data")
corpus = Corpus(config.ftrain)
train = corpus.load(config.ftrain)
dev = corpus.load(config.fdev)
file = args.file if args.file else config.crfpkl

start = datetime.now()

print("Create Conditional Random Field with %d tags" % corpus.nt)
if args.optimize:
    print("\tuse feature extracion optimization")
if args.anneal:
    print("\tuse simulated annealing")
if args.regularize:
    print("\tuse L2 regularization")
if args.shuffle:
    print("\tshuffle the data at each epoch")
crf = CRF(corpus.nt)

print("Use %d sentences to create the feature space" % corpus.ns)
crf.create_feature_space(train)
print("The size of the feature space is %d" % crf.d)

print("Use SGD algorithm to train the model")
print("\tepochs: %d\n\tbatch_size: %d\n\tinterval: %d\t\n\teta: %f" %
      (config.epochs, config.batch_size,  config.interval, config.eta))
if args.anneal:
    print("\tdacay: %f" % config.decay)
if args.regularize:
    print("\tlmbda: %f" % config.lmbda)
crf.SGD(train, dev, file,
        epochs=config.epochs,
        batch_size=config.batch_size,
        interval=config.interval,
        eta=config.eta,
        decay=config.decay,
        lmbda=config.lmbda,
        anneal=args.anneal,
        regularize=args.regularize,
        shuffle=args.shuffle)

if args.bigdata:
    test = corpus.load(config.ftest)
    crf = CRF.load(file)
    print("Precision of test: %d / %d = %4f" % crf.evaluate(test))

print("%ss elapsed\n" % (datetime.now() - start))
