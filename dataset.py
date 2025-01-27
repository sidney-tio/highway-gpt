#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Code modified from https://github.com/facebookresearch/transformer-sequential/blob/main/data.py

import os
import torch
import numpy as np
from typing import List


class Dictionary(object):
    def __init__(self):
        self.UNK = "<unk>"
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []
        self.add_word(self.UNK)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 0

    def add_count(self, word):
        self.add_word(word)
        self.word2count[word] += 1

    def build_indices(self):
        sorted_dict = sorted(self.word2count.items(), key=lambda kv: kv[1])[::-1]
        for i in range(len(sorted_dict)):
            word = sorted_dict[i][0]
            self.word2idx[word] = i
            self.idx2word.append(word)

    @staticmethod
    def _split_line(line: str) -> List[str]:
        return line.split()

    def build(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = type(self)._split_line(line) + ["<eos>"]
                for word in words:
                    self.add_count(word)
        # Sort dictionary by count and build indices accordingly:
        self.build_indices()
        # self.__check__()

    def getidx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx[self.UNK]

    def __len__(self):
        return len(self.idx2word)

    def __check__(self):
        for i in range(min(26, self.__len__())):
            word = self.idx2word[i]
            print(i, word, self.word2count[word])


class CharDictionary(Dictionary):
    @staticmethod
    def _split_line(line: str) -> List[str]:
        return [c for c in line]


class Corpus(object):
    def __init__(self, path, include_eos=False):
        self.include_eos = include_eos
        self.dictionary = self._make_dictionary()

        print("building dictionary")
        self.dictionary.build(os.path.join(path, "train.txt"))

        print("tokenizing dataset")
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

        if os.path.exists(os.path.join(path, "train.txt.labels")):
            self.train_labels = self.tokenize(os.path.join(path, "train.txt.labels"))
            self.valid_labels = self.tokenize(os.path.join(path, "valid.txt.labels"))
            self.test_labels = self.tokenize(os.path.join(path, "test.txt.labels"))

    def _make_dictionary(self):
        return Dictionary()

    def _split_line(self, line):
        return line.split()

    def tokenize(self, path):
        print("tokenizing " + path)
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = self._split_line(line)
                if self.include_eos:
                    words += ["<eos>"]
                tokens += len(words)
        ids = torch.IntTensor(tokens)
        with open(path, "r", encoding="utf8") as f:
            token = 0
            for line in f:
                words = self._split_line(line)
                if self.include_eos:
                    words += ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.getidx(word)
                    token += 1
        return ids.numpy().astype(np.uint16)


class CharCorpus(Corpus):
    def _make_dictionary(self):
        return CharDictionary()

    def _split_line(self, line):
        return [c for c in line]


def get_data(args, logger, include_eos: bool = False):
    corpus_path = os.path.join(args.training.data, "corpus.pt")
    if os.path.exists(corpus_path):
        corpus = torch.load(corpus_path)
        if include_eos:
            assert corpus.include_eos
    else:
        corpus = Corpus(args.training.data, include_eos)
        torch.save(corpus, corpus_path)

    vocab_sz = len(corpus.dictionary)
    logger.info("Dictionary contains %d words (including the unk token)" % vocab_sz)

    data_path = os.path.join(args.training.data, "train.bin")
    if not os.path.exists(data_path):
        corpus.train.tofile(os.path.join(args.training.data, "train.bin"))
        corpus.valid.tofile(os.path.join(args.training.data, "val.bin"))
        corpus.test.tofile(os.path.join(args.training.data, "test.bin"))
    return vocab_sz


def get_batch(args, split, device):
    data = np.memmap(
        os.path.join(args.training.data, f"{split}.bin"), dtype=np.uint16, mode="r"
    )
    ix = torch.randint(len(data) - args.model.block_size, (args.training.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + args.model.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + args.model.block_size]).astype(np.int64)
            )
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model, args, ctx, device):
    model.eval()
    data = np.memmap(
        os.path.join(args.training.data, f"test.bin"), dtype=np.uint16, mode="r"
    )
    n = len(data)
    iters = (n - args.model.block_size) // args.training.batch_size
    losses = torch.zeros(iters)
    for k in range(iters):
        idx = list(
            range(k, min(k + args.training.batch_size, n - args.model.block_size))
        )
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + args.model.block_size]).astype(np.int64))
                for i in idx
            ]
        )
        Y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + args.model.block_size]).astype(np.int64)
                )
                for i in idx
            ]
        )
        if device == "cuda":
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            X, Y = X.to(device), Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def evaluate(model, args, ctx, device):
    model.eval()
    data = np.memmap(
        os.path.join(args.training.data, f"test.bin"), dtype=np.uint16, mode="r"
    )
    n = len(data)
    iters = (n - args.model.block_size) // args.training.batch_size
    losses = torch.zeros(iters)
    for k in range(iters):
        idx = list(
            range(k, min(k + args.training.batch_size, n - args.model.block_size))
        )
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + args.model.block_size]).astype(np.int64))
                for i in idx
            ]
        )
        Y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + args.model.block_size]).astype(np.int64)
                )
                for i in idx
            ]
        )
        if device == "cuda":
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            X, Y = X.to(device), Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean().item()
