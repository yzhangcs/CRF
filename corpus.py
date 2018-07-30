# -*- coding: utf-8 -*-

import numpy as np


class Corpus(object):
    UNK = '<UNK>'

    def __init__(self, fdata):
        self.sentences = self.preprocess(fdata)
        self.wordseqs, self.tagseqs = zip(*self.sentences)
        self.words = sorted(set(np.hstack(self.wordseqs)))
        self.tags = sorted(set(np.hstack(self.tagseqs)))
        self.chars = sorted({c for w in self.words for c in w})
        self.chars.append(self.UNK)

        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.ui = self.cdict[self.UNK]
        self.ns = len(self.sentences)
        self.nw = len(self.words)
        self.nt = len(self.tags)

    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wordseq = [
                tuple(self.cdict[c]
                      if c in self.cdict else self.ui
                      for c in w)
                for w in wordseq
            ]
            tiseq = [self.tdict[t] for t in tagseq]
            data.append((wordseq, tiseq))
        return data

    def size(self):
        return self.nw - 1, self.nt

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as train:
            lines = [line for line in train]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences
