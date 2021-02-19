# Copyright (c) 2020-     UofT-EcoSystem,
# Copyright 2018 - 2019 Junseong Kim, Scatter Lab, respective BERT contributors
# Copyright (c) 2018 Alexander Rush : The Annotated Trasnformer
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from io import open
import torch


class Dictionary(object):

  def __init__(self):
    self.word2idx = {}
    self.idx2word = []

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


class Corpus(object):

  def __init__(self, path, max_token=None, with_pos=False):
    self.dictionary = Dictionary()
    self.last = set()
    self.counter = {}
    self.add_dict(os.path.join(path, 'wiki.train.tokens'))
    self.add_dict(os.path.join(path, 'wiki.valid.tokens'))
    self.add_dict(os.path.join(path, 'wiki.test.tokens'))
    words = [(v, k) for k, v in self.counter.items()]
    words.sort(reverse=True)

    if max_token is not None:
      if max_token < len(words):
        max_token -= 1
      for i, item in enumerate(words):
        if (i < max_token):
          self.dictionary.add_word(item[1])
        else:
          self.last.add(item[1])
      self.dictionary.add_word("<ONCE WORD>")
    else:
      for i, item in enumerate(words):
        self.dictionary.add_word(item[1])

    self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'),
                               with_pos)
    self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'),
                               with_pos)
    self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'), with_pos)

  def add_dict(self, path):
    assert os.path.exists(path)
    # Add words to the dictionary

    with open(path, 'r', encoding="utf8") as f:
      for line in f:
        words = line.split() + ['<eos>']
        for word in words:
          if word in self.counter:
            self.counter[word] += 1
          else:
            self.counter[word] = 1

  def tokenize(self, path, with_pos):
    """Tokenizes a text file."""
    assert os.path.exists(path)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
      idss = []
      for lines in f:
        lines = lines.split(" . ")
        for line in lines:
          words = line.split() + ['<eos>']
          ids = []
          for word in words:
            if word in self.last:
              word = "<ONCE WORD>"
            ids.append(self.dictionary.word2idx[word])
          idss.append(torch.tensor(ids).type(torch.int64))

      if with_pos:
        poss = []
        for ids in idss:
          pos = torch.range(0, len(ids))
          poss.append(torch.tensor(pos).type(torch.int64))
        poss = torch.cat(poss)
        ids = torch.cat(idss)
        return ids, poss
      else:
        return torch.cat(idss)
