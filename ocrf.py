# -*- coding: utf-8 -*-

import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.misc import logsumexp


class CRF(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wiseq, tiseq in data
            for f in set(self.instantiate(wiseq, 0, -1)).union(
                *[self.instantiate(wiseq, i, tiseq[i - 1])
                  for i, ti in enumerate(tiseq[1:], 1)]
            )
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.nt))
        # Bigram特征及对应权重分值
        self.BF = [self.bigram(prev_ti) for prev_ti in range(self.nt)]
        self.BS = np.array([self.score(bfv) for bfv in self.BF])

    def SGD(self, train, dev, file,
            epochs, batch_size, interval, eta, decay, lmbda,
            anneal, regularize):
        # 训练集大小
        n = len(train)
        # 记录更新次数
        count = 0
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_precision = 0, 0.0

        # 迭代指定次数训练模型
        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 随机打乱数据
            random.shuffle(train)
            # 设置L2正则化系数
            if not regularize:
                lmbda = 0
            # 按照指定大小对数据分割批次
            batches = [train[i:i + batch_size]
                       for i in range(0, len(train), batch_size)]
            nb = len(batches)
            # 根据批次数据更新权重
            for batch in batches:
                if not anneal:
                    self.update(batch, lmbda, n, eta)
                # 设置学习速率的指数衰减
                else:
                    self.update(batch, lmbda, n, eta * decay ** (count / nb))
                count += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            # 保存效果最好的模型
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))
        print("mean time of each epoch is %ss" % (total_time / epoch))

    def update(self, batch, lmbda, n, eta):
        gradients = defaultdict(float)

        for wiseq, tiseq in batch:
            prev_ti = -1
            for i, ti in enumerate(tiseq):
                fiseq = (self.fdict[f]
                         for f in self.instantiate(wiseq, i, prev_ti)
                         if f in self.fdict)
                for fi in fiseq:
                    gradients[fi, ti] += 1
                prev_ti = ti

            alpha = self.forward(wiseq)
            beta = self.backward(wiseq)
            logZ = logsumexp(alpha[-1])

            fv = self.instantiate(wiseq, 0, -1)
            fiseq = (self.fdict[f] for f in fv if f in self.fdict)
            p = np.exp(self.score(fv) + beta[0] - logZ)
            for fi in fiseq:
                gradients[fi] -= p

            for i in range(1, len(tiseq)):
                ufv = self.unigram(wiseq, i)
                ufiseq = [self.fdict[f] for f in ufv if f in self.fdict]
                scores = self.BS + self.score(ufv)
                probs = np.exp(scores + alpha[i - 1][:, None] + beta[i] - logZ)

                for bfv, p in zip(self.BF, probs):
                    bfiseq = [self.fdict[f] for f in bfv if f in self.fdict]
                    for fi in bfiseq + ufiseq:
                        gradients[fi] -= p

        if lmbda != 0:
            self.W *= (1 - eta * lmbda / n)
        for k, v in gradients.items():
            self.W[k] += eta * v
        self.BS = np.array([self.score(bfv) for bfv in self.BF])

    def forward(self, wiseq):
        T = len(wiseq)
        alpha = np.zeros((T, self.nt))

        fv = self.instantiate(wiseq, 0, -1)
        alpha[0] = self.score(fv)

        for i in range(1, T):
            uscores = self.score(self.unigram(wiseq, i))
            scores = np.transpose(self.BS + uscores)
            alpha[i] = logsumexp(scores + alpha[i - 1], axis=1)
        return alpha

    def backward(self, wiseq):
        T = len(wiseq)
        beta = np.zeros((T, self.nt))

        for i in reversed(range(T - 1)):
            uscores = self.score(self.unigram(wiseq, i + 1))
            scores = self.BS + uscores
            beta[i] = logsumexp(scores + beta[i + 1], axis=1)
        return beta

    def predict(self, wiseq):
        T = len(wiseq)
        delta = np.zeros((T, self.nt))
        paths = np.zeros((T, self.nt), dtype='int')

        fv = self.instantiate(wiseq, 0, -1)
        delta[0] = self.score(fv)

        for i in range(1, T):
            uscores = self.score(self.unigram(wiseq, i))
            scores = np.transpose(self.BS + uscores) + delta[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            delta[i] = scores[np.arange(self.nt), paths[i]]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return predict

    def score(self, fvector):
        scores = np.array([self.W[self.fdict[f]]
                           for f in fvector if f in self.fdict])
        return np.sum(scores, axis=0)

    def bigram(self, prev_ti):
        return [('01', prev_ti)]

    def unigram(self, wiseq, index):
        word = wiseq[index]
        prev_word = wiseq[index - 1] if index > 0 else '^^'
        next_word = wiseq[index + 1] if index < len(wiseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', word))
        fvector.append(('03', prev_word))
        fvector.append(('04', next_word))
        fvector.append(('05', word, prev_char))
        fvector.append(('06', word, next_char))
        fvector.append(('07', first_char))
        fvector.append(('08', last_char))

        for char in word[1:-1]:
            fvector.append(('09', char))
            fvector.append(('10', first_char, char))
            fvector.append(('11', last_char, char))
        if len(word) == 1:
            fvector.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', word[:i]))
                fvector.append(('15', word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', word))
            fvector.append(('15', word))
        return fvector

    def instantiate(self, wiseq, index, prev_ti):
        bigram = self.bigram(prev_ti)
        unigram = self.unigram(wiseq, index)
        return bigram + unigram

    def evaluate(self, data):
        tp, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            piseq = np.array(self.predict(wiseq))
            tp += np.sum(tiseq == piseq)
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            crf = pickle.load(f)
        return crf
